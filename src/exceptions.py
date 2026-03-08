from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .types import GuardDecision, ScanResult

class LLMSecurityError(Exception):
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = context or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"

class BlockedByPolicyError(LLMSecurityError):
    def __init__(
        self,
        reasons: List[str],
        score: float = 1.0,
        decision: Optional["GuardDecision"] = None,
        policy_name: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reasons = reasons
        self.score = score
        self.decision = decision
        self.policy_name = policy_name

        reason_summary = "; ".join(reasons) if reasons else "no details provided"
        message = (
            f"Request blocked by policy {policy_name!r} "
            f"(score={score:.3f}): {reason_summary}"
        )
        super().__init__(message, context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error":       type(self).__name__,
            "reasons":     self.reasons,
            "score":       round(self.score, 4),
            "policy_name": self.policy_name,
        }

class PromptBlockedError(BlockedByPolicyError):
    def __init__(
        self,
        reasons: List[str],
        score: float = 1.0,
        prompt_snippet: str = "",
        decision: Optional["GuardDecision"] = None,
        policy_name: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt_snippet = prompt_snippet[:120]
        super().__init__(
            reasons=reasons,
            score=score,
            decision=decision,
            policy_name=policy_name,
            context=context,
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["prompt_snippet"] = self.prompt_snippet
        return base


class OutputBlockedError(BlockedByPolicyError):
    def __init__(
        self,
        reasons: List[str],
        score: float = 1.0,
        output_snippet: str = "",
        decision: Optional["GuardDecision"] = None,
        policy_name: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_snippet = output_snippet[:120]
        super().__init__(
            reasons=reasons,
            score=score,
            decision=decision,
            policy_name=policy_name,
            context=context,
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["output_snippet"] = self.output_snippet
        return base

class InvalidToolCallError(LLMSecurityError):
    def __init__(
        self,
        tool_name: str,
        reason: str,
        scan_result: Optional["ScanResult"] = None,
        call_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tool_name = tool_name
        self.reason = reason
        self.scan_result = scan_result
        self.call_id = call_id

        call_id_part = f" [call_id={call_id!r}]" if call_id else ""
        message = (
            f"Invalid tool call for {tool_name!r}{call_id_part}: {reason}"
        )
        super().__init__(message, context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error":     type(self).__name__,
            "tool_name": self.tool_name,
            "reason":    self.reason,
            "call_id":   self.call_id,
        }

class PolicyNotFoundError(LLMSecurityError):
    def __init__(
        self,
        policy_name: str,
        available: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy_name = policy_name
        self.available = available or []

        if self.available:
            avail_str = ", ".join(sorted(self.available))
            message = (
                f"Policy {policy_name!r} not found. "
                f"Available policies: {avail_str}"
            )
        else:
            message = f"Policy {policy_name!r} not found. No policies are registered."

        super().__init__(message, context)

class ProviderError(LLMSecurityError):
    def __init__(
        self,
        message: str,
        provider_name: str = "unknown",
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.provider_name = provider_name
        self.status_code = status_code
        self.original_error = original_error

        status_part = f" (HTTP {status_code})" if status_code else ""
        full_message = f"[{provider_name}]{status_part} {message}"
        super().__init__(full_message, context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error":         type(self).__name__,
            "provider_name": self.provider_name,
            "status_code":   self.status_code,
            "message":       self.message,
        }

class ProviderTimeoutError(ProviderError):
    def __init__(
        self,
        provider_name: str = "unknown",
        timeout_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds

        timeout_part = (
            f" after {timeout_seconds}s" if timeout_seconds is not None else ""
        )
        message = f"Provider request timed out{timeout_part}."
        super().__init__(
            message=message,
            provider_name=provider_name,
            status_code=408,
            original_error=original_error,
            context=context,
        )

class GuardError(LLMSecurityError):
    def __init__(
        self,
        guard_name: str,
        message: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.guard_name = guard_name
        self.original_error = original_error

        full_message = f"Guard fault in {guard_name!r}: {message}"
        super().__init__(full_message, context)

__all__ = [
    # Base
    "LLMSecurityError",
    # Policy blocks
    "BlockedByPolicyError",
    "PromptBlockedError",
    "OutputBlockedError",
    # Tool validation
    "InvalidToolCallError",
    # Policy registry
    "PolicyNotFoundError",
    # Provider errors
    "ProviderError",
    "ProviderTimeoutError",
    # Guard faults
    "GuardError",
]