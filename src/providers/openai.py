from __future__ import annotations
import logging
import os
from typing import Any, Dict, Generator, Iterable, List, Optional
from ..exceptions import ProviderError, ProviderTimeoutError
from ..policies import Policy
from ..types import GuardDecision, ToolCall
from .base import BaseProvider, ChatMessage, ProviderConfig

_logger = logging.getLogger(__name__)

def _get_openai():
    try:
        import openai
        return openai
    except ImportError as exc:
        raise ProviderError(
            message=(
                "The openai package is required to use OpenAIProvider. "
                "Install it with: pip install openai>=1.0.0"
            ),
            provider_name="openai",
        ) from exc

def _parse_openai_tool_calls(
    raw_tool_calls: Any,
) -> List[ToolCall]:
    if not raw_tool_calls:
        return []

    result: List[ToolCall] = []
    for tc in raw_tool_calls:
        try:
            import json as _json

            if hasattr(tc, "function"):
                fn   = tc.function
                name = fn.name or ""
                try:
                    args = _json.loads(fn.arguments or "{}")
                except (_json.JSONDecodeError, TypeError):
                    args = {}
                result.append(ToolCall(
                    name    = name,
                    args    = args,
                    call_id = getattr(tc, "id", None),
                ))

            elif hasattr(tc, "name"):
                try:
                    args = _json.loads(getattr(tc, "arguments", "{}") or "{}")
                except (_json.JSONDecodeError, TypeError):
                    args = {}
                result.append(ToolCall(
                    name    = tc.name,
                    args    = args,
                    call_id = None,
                ))
        except Exception:
            _logger.warning(
                "OpenAIProvider: could not parse tool call %r — skipping", tc
            )
    return result

def _build_openai_tool_schema(tool_call: ToolCall) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name":        tool_call.name,
            "description": tool_call.schema.get("description", ""),
            "parameters":  tool_call.schema,
        },
    }

class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(
        self,
        api_key:        Optional[str]         = None,
        model:          str                   = "gpt-4o",
        organization:   Optional[str]         = None,
        base_url:       Optional[str]         = None,
        policy:         Optional[Policy]      = None,
        config:         Optional[ProviderConfig] = None,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(policy=policy, config=config)

        self._model          = model
        self._default_params = default_params or {}

        # Resolve credentials from args → environment
        resolved_key  = api_key      or os.environ.get("OPENAI_API_KEY", "")
        resolved_org  = organization or os.environ.get("OPENAI_ORG_ID")

        if not resolved_key:
            raise ProviderError(
                message=(
                    "No OpenAI API key provided. Pass api_key= or set the "
                    "OPENAI_API_KEY environment variable."
                ),
                provider_name="openai",
            )

        # Build the OpenAI client (lazy import)
        openai = _get_openai()
        client_kwargs: Dict[str, Any] = {
            "api_key": resolved_key,
            "timeout": self._config.timeout_seconds,
        }
        if resolved_org:
            client_kwargs["organization"] = resolved_org
        if base_url:
            client_kwargs["base_url"] = base_url
        if self._config.extra_headers:
            client_kwargs["default_headers"] = self._config.extra_headers

        self._client = openai.OpenAI(**client_kwargs)

    def _call_model(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        openai = _get_openai()

        params = {**self._default_params, **kwargs}
        params.setdefault("model", self._model)

        # Convert ToolCall objects in tools= to OpenAI tool dicts
        if "tools" in params and isinstance(params["tools"], list):
            converted: List[Dict[str, Any]] = []
            for item in params["tools"]:
                if isinstance(item, ToolCall):
                    converted.append(_build_openai_tool_schema(item))
                else:
                    converted.append(item)  # already a dict
            params["tools"] = converted

        try:
            return self._client.chat.completions.create(
                messages=messages,
                **params,
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(
                provider_name="openai",
                timeout_seconds=self._config.timeout_seconds,
                original_error=exc,
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderError(
                message="OpenAI rate limit exceeded. Retry after backing off.",
                provider_name="openai",
                status_code=429,
                original_error=exc,
            ) from exc
        except openai.AuthenticationError as exc:
            raise ProviderError(
                message="OpenAI API key is invalid or has expired.",
                provider_name="openai",
                status_code=401,
                original_error=exc,
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise ProviderError(
                message="OpenAI request denied (permission error).",
                provider_name="openai",
                status_code=403,
                original_error=exc,
            ) from exc
        except openai.NotFoundError as exc:
            raise ProviderError(
                message=f"OpenAI model or resource not found: {params.get('model')}",
                provider_name="openai",
                status_code=404,
                original_error=exc,
            ) from exc
        except openai.BadRequestError as exc:
            raise ProviderError(
                message=f"OpenAI rejected the request: {exc.message}",
                provider_name="openai",
                status_code=400,
                original_error=exc,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(
                message="Could not connect to the OpenAI API. Check your network.",
                provider_name="openai",
                original_error=exc,
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                message=f"OpenAI API error: {exc.message}",
                provider_name="openai",
                status_code=exc.status_code,
                original_error=exc,
            ) from exc

    def _extract_text(self, response: Any) -> str:
        try:
            choice  = response.choices[0]
            message = choice.message
            content = message.content
            return content if isinstance(content, str) else ""
        except (IndexError, AttributeError):
            return ""

    def _extract_tool_calls(self, response: Any) -> List[ToolCall]:
        try:
            raw = response.choices[0].message.tool_calls
            return _parse_openai_tool_calls(raw)
        except (IndexError, AttributeError):
            return []

    def stream_chat(
        self,
        messages: List[ChatMessage],
        *,
        policy: Optional[Policy] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, GuardDecision]:
        import time as _time
        from ..guards.prompts import scan_messages, aggregate_prompt_scans
        from ..guards.outputs import scan_and_redact
        from ..types import PolicyAction, GuardType, ScanResult
        from ..exceptions import PromptBlockedError
        from .base import _snippet

        active_policy = policy or self._policy
        start = _time.perf_counter()

        per_msg  = scan_messages(messages, active_policy, roles_to_scan=self._config.roles_to_scan)
        combined = aggregate_prompt_scans(per_msg, active_policy)

        if not combined.allowed:
            decision = GuardDecision(
                allowed=False,
                score=round(combined.score, 4),
                reasons=combined.reasons,
                safe_output=None,
                warned=False,
                scan_results=[combined],
                action=PolicyAction.BLOCK,
            )
            elapsed_ms = (_time.perf_counter() - start) * 1000
            self._sec_logger.log_decision(decision, duration_ms=elapsed_ms, extra=extra)

            if active_policy.raise_on_block:
                raise PromptBlockedError(
                    reasons=combined.reasons,
                    score=combined.score,
                    decision=decision,
                    policy_name=active_policy.name,
                    prompt_snippet=_snippet(messages),
                )
            return  # generator returns immediately

        openai = _get_openai()
        params = {**self._default_params, **kwargs}
        params.setdefault("model", self._model)
        params["stream"] = True
        try:
            stream = self._client.chat.completions.create(
                messages=messages,
                **params,
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(
                provider_name="openai",
                timeout_seconds=self._config.timeout_seconds,
                original_error=exc,
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                message=f"OpenAI API error: {exc.message}",
                provider_name="openai",
                status_code=exc.status_code,
                original_error=exc,
            ) from exc

        # Accumulate full response while yielding chunks
        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            text  = (delta.content or "") if delta else ""
            if text:
                full_text += text
                yield text

        output_scan = scan_and_redact(full_text, active_policy)
        scan_results = [combined, output_scan]

        if not output_scan.allowed:
            from ..exceptions import OutputBlockedError
            decision = GuardDecision(
                allowed=False,
                score=round(output_scan.score, 4),
                reasons=output_scan.reasons,
                safe_output=output_scan.safe_output,
                warned=False,
                scan_results=scan_results,
                action=PolicyAction.BLOCK,
            )
            elapsed_ms = (_time.perf_counter() - start) * 1000
            self._sec_logger.log_decision(decision, duration_ms=elapsed_ms, extra=extra)

            if active_policy.raise_on_block:
                raise OutputBlockedError(
                    reasons=output_scan.reasons,
                    score=output_scan.score,
                    output_snippet=full_text[:120],
                    decision=decision,
                    policy_name=active_policy.name,
                )
            yield "[BLOCKED]"
            return

        # Clean / warned decision
        max_score = max(combined.score, output_scan.score)
        warn_threshold = active_policy.effective_warn_threshold(GuardType.OUTPUT)
        warned = max_score >= warn_threshold
        action = PolicyAction.WARN if warned else PolicyAction.LOG

        decision = GuardDecision(
            allowed=True,
            score=round(max_score, 4),
            reasons=combined.reasons + output_scan.reasons,
            safe_output=output_scan.safe_output or full_text,
            warned=warned,
            scan_results=scan_results,
            action=action,
        )
        elapsed_ms = (_time.perf_counter() - start) * 1000
        self._sec_logger.log_decision(decision, duration_ms=elapsed_ms, extra=extra)

    def chat_with_tool_definitions(
        self,
        messages: List[ChatMessage],
        tool_definitions: List[Dict[str, Any]],
        *,
        policy: Optional[Policy] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GuardDecision:
        return self.chat(
            messages=messages,
            policy=policy,
            extra=extra,
            tools=tool_definitions,  # forwarded as-is in kwargs
            **kwargs,
        )

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model must be a non-empty string")
        self._model = value

    @property
    def client(self):
        return self._client

__all__ = [
    "OpenAIProvider",
]