from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from ..config import get_default_policy
from ..exceptions import (
    BlockedByPolicyError,
    GuardError,
    OutputBlockedError,
    PromptBlockedError,
    ProviderError,
)
from ..guards.outputs import scan_and_redact
from ..guards.prompts import aggregate_prompt_scans, scan_messages
from ..guards.tools import aggregate_tool_scans, validate_tool_calls
from ..logging import LogFormat, SecurityEventLogger
from ..policies import Policy
from ..types import GuardDecision, GuardType, PolicyAction, ScanResult, ToolCall

_logger = logging.getLogger(__name__)

ChatMessage = Dict[str, str]

@dataclass
class ProviderConfig:
    timeout_seconds: float          = 30.0
    max_retries:     int            = 0
    log_format:      LogFormat      = LogFormat.JSON
    extra_headers:   Dict[str, str] = field(default_factory=dict)
    default_extra:   Dict[str, Any] = field(default_factory=dict)
    roles_to_scan:   List[str]      = field(default_factory=lambda: ["user"])

class BaseProvider(ABC):
    #: Must be overridden in every concrete subclass.
    provider_name: str = "base"

    def __init__(
        self,
        policy: Optional[Policy] = None,
        config: Optional[ProviderConfig] = None,
    ) -> None:
        self._policy = policy or get_default_policy()
        self._config = config or ProviderConfig()
        self._sec_logger = SecurityEventLogger(
            policy        = self._policy,
            provider_name = self.provider_name,
            fmt           = self._config.log_format,
            default_extra = self._config.default_extra,
        )

    @property
    def policy(self) -> Policy:
        return self._policy

    @policy.setter
    def policy(self, value: Policy) -> None:
        if not isinstance(value, Policy):
            raise TypeError(
                f"policy must be a Policy instance, got {type(value).__name__!r}"
            )
        self._policy = value
        self._sec_logger.policy = value

    @property
    def config(self) -> ProviderConfig:
        return self._config

    @abstractmethod
    def _call_model(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        """
        Make the raw API call to the LLM provider.

        This is called by ``chat()`` after all pre-call guards pass.
        The return value is passed directly to ``_extract_text()``.

        Args:
            messages: The full conversation in chat-message format.
            **kwargs: Any extra keyword arguments forwarded from chat().

        Returns:
            The raw SDK response object (provider-specific).

        Raises:
            ProviderError: Concrete providers should wrap SDK-specific
                           exceptions in ProviderError or ProviderTimeoutError
                           before re-raising.
        """

    @abstractmethod
    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's text content from the SDK response.

        Args:
            response: The raw SDK response returned by _call_model().

        Returns:
            The assistant's reply as a plain string. Return "" if the
            response contained no text (e.g. a tool-only response).
        """

    def _extract_tool_calls(self, response: Any) -> List[ToolCall]:
        return []

    def _on_blocked(
        self,
        exc: BlockedByPolicyError,
        messages: List[ChatMessage],
    ) -> None:
        """
        Optional hook called just before a BlockedByPolicyError is raised.

        Use this to attach provider-specific context (e.g. request IDs)
        to the exception, or to trigger custom alerting.

        The default implementation does nothing.

        Args:
            exc:      The exception that is about to be raised.
            messages: The original message list that was blocked.
        """

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[ToolCall]] = None,
        policy: Optional[Policy] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GuardDecision:
        active_policy = policy or self._policy

        if not messages:
            raise ValueError("messages must be a non-empty list")

        start_time = time.perf_counter()
        scan_results: List[ScanResult] = []

        prompt_scan = self._run_prompt_guard(messages, active_policy)
        scan_results.append(prompt_scan)

        if not prompt_scan.allowed:
            decision = self._build_blocked_decision(
                scan_results=scan_results,
                policy=active_policy,
                guard_type=GuardType.PROMPT,
            )
            self._log_and_maybe_raise(
                decision=decision,
                policy=active_policy,
                messages=messages,
                start_time=start_time,
                extra=extra,
                exc_class=PromptBlockedError,
                exc_kwargs={
                    "prompt_snippet": _snippet(messages),
                },
            )
            return decision

        if tools:
            tool_scan = self._run_tool_guard(tools, active_policy)
            scan_results.append(tool_scan)

            if not tool_scan.allowed:
                decision = self._build_blocked_decision(
                    scan_results=scan_results,
                    policy=active_policy,
                    guard_type=GuardType.TOOL,
                )
                self._log_and_maybe_raise(
                    decision=decision,
                    policy=active_policy,
                    messages=messages,
                    start_time=start_time,
                    extra=extra,
                    exc_class=BlockedByPolicyError,
                    exc_kwargs={},
                )
                return decision

        try:
            response = self._call_model(messages, **kwargs)
            raw_text = self._extract_text(response)
        except (ProviderError, BlockedByPolicyError):
            raise  # already wrapped — let it propagate
        except Exception as exc:
            raise ProviderError(
                message=str(exc),
                provider_name=self.provider_name,
                original_error=exc,
            ) from exc

        response_tools = self._extract_tool_calls(response)
        if response_tools and active_policy.is_guard_enabled(GuardType.TOOL):
            resp_tool_scan = self._run_tool_guard(response_tools, active_policy)
            scan_results.append(resp_tool_scan)

            if not resp_tool_scan.allowed:
                decision = self._build_blocked_decision(
                    scan_results=scan_results,
                    policy=active_policy,
                    guard_type=GuardType.TOOL,
                )
                self._log_and_maybe_raise(
                    decision=decision,
                    policy=active_policy,
                    messages=messages,
                    start_time=start_time,
                    extra=extra,
                    exc_class=BlockedByPolicyError,
                    exc_kwargs={},
                )
                return decision

        if raw_text:
            output_scan = self._run_output_guard(raw_text, active_policy)
            scan_results.append(output_scan)

            if not output_scan.allowed:
                decision = self._build_blocked_decision(
                    scan_results=scan_results,
                    policy=active_policy,
                    guard_type=GuardType.OUTPUT,
                    safe_output=output_scan.safe_output,
                )
                self._log_and_maybe_raise(
                    decision=decision,
                    policy=active_policy,
                    messages=messages,
                    start_time=start_time,
                    extra=extra,
                    exc_class=OutputBlockedError,
                    exc_kwargs={
                        "output_snippet": (raw_text or "")[:120],
                    },
                )
                return decision
        else:
            # Empty text response — treat as clean
            output_scan = ScanResult(
                allowed=True, score=0.0, reasons=[],
                guard_type=GuardType.OUTPUT,
                metadata={"skipped": True, "reason": "empty model response"},
            )
            scan_results.append(output_scan)

        decision = self._build_clean_decision(
            scan_results=scan_results,
            policy=active_policy,
            safe_output=output_scan.safe_output or raw_text,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._sec_logger.log_decision(decision, duration_ms=elapsed_ms, extra=extra)

        return decision

    def _run_prompt_guard(
        self,
        messages: List[ChatMessage],
        policy: Policy,
    ) -> ScanResult:
        try:
            per_message = scan_messages(
                messages,
                policy,
                roles_to_scan=self._config.roles_to_scan,
            )
            return aggregate_prompt_scans(per_message, policy)
        except Exception as exc:
            raise GuardError(
                guard_name="prompt_guard",
                message=str(exc),
                original_error=exc,
            ) from exc

    def _run_tool_guard(
        self,
        tools: List[ToolCall],
        policy: Policy,
    ) -> ScanResult:
        try:
            per_call = validate_tool_calls(tools, policy)
            return aggregate_tool_scans(per_call, policy)
        except Exception as exc:
            raise GuardError(
                guard_name="tool_guard",
                message=str(exc),
                original_error=exc,
            ) from exc

    def _run_output_guard(
        self,
        text: str,
        policy: Policy,
    ) -> ScanResult:
        try:
            return scan_and_redact(text, policy)
        except Exception as exc:
            raise GuardError(
                guard_name="output_guard",
                message=str(exc),
                original_error=exc,
            ) from exc

    def _build_blocked_decision(
        self,
        scan_results: List[ScanResult],
        policy: Policy,
        guard_type: GuardType,
        safe_output: Optional[str] = None,
    ) -> GuardDecision:
        max_score = max((r.score for r in scan_results), default=0.0)
        all_reasons = [reason for sr in scan_results for reason in sr.reasons]

        return GuardDecision(
            allowed=False,
            score=round(max_score, 4),
            reasons=all_reasons,
            safe_output=safe_output,
            warned=False,
            scan_results=list(scan_results),
            action=PolicyAction.BLOCK,
        )

    def _build_clean_decision(
        self,
        scan_results: List[ScanResult],
        policy: Policy,
        safe_output: str,
    ) -> GuardDecision:
        max_score = max((r.score for r in scan_results), default=0.0)
        all_reasons = [reason for sr in scan_results for reason in sr.reasons]

        warn_threshold = policy.effective_warn_threshold(GuardType.OUTPUT)
        warned = max_score >= warn_threshold
        action = PolicyAction.WARN if warned else PolicyAction.LOG

        return GuardDecision(
            allowed=True,
            score=round(max_score, 4),
            reasons=all_reasons,
            safe_output=safe_output,
            warned=warned,
            scan_results=list(scan_results),
            action=action,
        )

    def _log_and_maybe_raise(
        self,
        decision: GuardDecision,
        policy: Policy,
        messages: List[ChatMessage],
        start_time: float,
        extra: Optional[Dict[str, Any]],
        exc_class: type,
        exc_kwargs: Dict[str, Any],
    ) -> None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._sec_logger.log_decision(decision, duration_ms=elapsed_ms, extra=extra)

        if policy.raise_on_block:
            exc = exc_class(
                reasons=decision.reasons,
                score=decision.score,
                decision=decision,
                policy_name=policy.name,
                **exc_kwargs,
            )
            self._on_blocked(exc, messages)
            raise exc

def _snippet(messages: List[ChatMessage], max_len: int = 120) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content[:max_len]
    return ""

__all__ = [
    "BaseProvider",
    "ProviderConfig",
    "ChatMessage",
]