from __future__ import annotations
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import ASGIApp
from ..config import get_default_policy
from ..exceptions import BlockedByPolicyError, PromptBlockedError, OutputBlockedError
from ..guards.outputs import scan_and_redact
from ..guards.prompts import aggregate_prompt_scans, scan_messages
from ..logging import LogFormat, SecurityEventLogger
from ..policies import Policy
from ..types import GuardDecision, GuardType, PolicyAction, ScanResult

_logger = logging.getLogger(__name__)

#: Request state key where GuardedRoute stores the GuardDecision.
GUARD_DECISION_KEY = "llm_security_decision"

#: Request state key where GuardedRoute stores the active policy.
GUARD_POLICY_KEY   = "llm_security_policy"

#: Default paths the middleware inspects for message content.
DEFAULT_SCAN_PATHS = ("/chat", "/v1/chat", "/v1/messages", "/api/chat")

#: JSON field names the middleware looks for messages in request bodies.
MESSAGE_FIELD_NAMES = ("messages", "prompt", "input", "query", "text")

#: JSON field names the middleware looks for text in response bodies.
OUTPUT_FIELD_NAMES  = ("content", "output", "text", "response", "message",
                       "choices")

def _blocked_json(
    decision: GuardDecision,
    status_code: int = 400,
    include_reasons: bool = True,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error":   "request_blocked",
        "message": "This request was blocked by the LLM security policy.",
        "score":   decision.score,
        "action":  decision.action.value,
    }
    if include_reasons:
        body["reasons"] = decision.reasons
    return JSONResponse(content=body, status_code=status_code)


def decision_response(
    decision: GuardDecision,
    *,
    blocked_status_code: int  = 400,
    include_reasons:     bool = True,
) -> Optional[JSONResponse]:
    if not decision.allowed:
        return _blocked_json(
            decision,
            status_code=blocked_status_code,
            include_reasons=include_reasons,
        )
    return None

def guard_messages(
    messages: List[Dict[str, str]],
    policy:   Optional[Policy] = None,
    *,
    roles_to_scan: List[str]            = ("user",),
    raise_on_block: Optional[bool]      = None,
    extra:          Optional[Dict[str, Any]] = None,
) -> GuardDecision:
    active_policy = policy or get_default_policy()
    per_msg  = scan_messages(messages, active_policy, roles_to_scan=list(roles_to_scan))
    combined = aggregate_prompt_scans(per_msg, active_policy)

    should_raise = raise_on_block if raise_on_block is not None else active_policy.raise_on_block

    scan_results = list(per_msg)
    max_score    = combined.score
    all_reasons  = combined.reasons
    action       = active_policy.action_for_score(max_score, GuardType.PROMPT)

    decision = GuardDecision(
        allowed      = combined.allowed,
        score        = round(max_score, 4),
        reasons      = all_reasons,
        safe_output  = None,
        warned       = action == PolicyAction.WARN,
        scan_results = scan_results,
        action       = action,
    )

    if not combined.allowed and should_raise:
        raise PromptBlockedError(
            reasons     = all_reasons,
            score       = max_score,
            decision    = decision,
            policy_name = active_policy.name,
            prompt_snippet = _extract_last_user_content(messages),
        )

    return decision


def guard_output(
    text:   str,
    policy: Optional[Policy] = None,
    *,
    raise_on_block: Optional[bool]      = None,
    extra:          Optional[Dict[str, Any]] = None,
) -> GuardDecision:
    active_policy = policy or get_default_policy()
    scan  = scan_and_redact(text, active_policy)
    action = active_policy.action_for_score(scan.score, GuardType.OUTPUT)

    decision = GuardDecision(
        allowed      = scan.allowed,
        score        = round(scan.score, 4),
        reasons      = scan.reasons,
        safe_output  = scan.safe_output,
        warned       = action == PolicyAction.WARN,
        scan_results = [scan],
        action       = action,
    )

    should_raise = raise_on_block if raise_on_block is not None else active_policy.raise_on_block

    if not scan.allowed and should_raise:
        raise OutputBlockedError(
            reasons        = scan.reasons,
            score          = scan.score,
            output_snippet = text[:120],
            decision       = decision,
            policy_name    = active_policy.name,
        )

    return decision

class GuardedRoute:
    def __init__(
        self,
        policy:          Optional[Policy]   = None,
        roles_to_scan:   Sequence[str]      = ("user",),
        messages_field:  Optional[str]      = None,
        block_status:    int                = 400,
        include_reasons: bool               = True,
    ) -> None:
        self._policy          = policy or get_default_policy()
        self._roles_to_scan   = list(roles_to_scan)
        self._messages_field  = messages_field
        self._block_status    = block_status
        self._include_reasons = include_reasons
        self._sec_logger      = SecurityEventLogger(
            policy        = self._policy,
            provider_name = "fastapi",
        )

    async def __call__(self, request: Request) -> GuardDecision:
        start = time.perf_counter()

        messages = await self._extract_messages(request)
        if not messages:
            # No messages found — pass through with a clean decision
            decision = GuardDecision.clean(None)
            request.state.__dict__[GUARD_DECISION_KEY] = decision
            request.state.__dict__[GUARD_POLICY_KEY]   = self._policy
            return decision

        per_msg  = scan_messages(messages, self._policy, roles_to_scan=self._roles_to_scan)
        combined = aggregate_prompt_scans(per_msg, self._policy)

        action  = self._policy.action_for_score(combined.score, GuardType.PROMPT)
        warned  = action == PolicyAction.WARN

        decision = GuardDecision(
            allowed      = combined.allowed,
            score        = round(combined.score, 4),
            reasons      = combined.reasons,
            safe_output  = None,
            warned       = warned,
            scan_results = list(per_msg),
            action       = action,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._sec_logger.log_decision(decision, duration_ms=elapsed_ms)

        # Store on request state for handler access
        request.state.__dict__[GUARD_DECISION_KEY] = decision
        request.state.__dict__[GUARD_POLICY_KEY]   = self._policy

        if not decision.allowed and self._policy.raise_on_block:
            raise PromptBlockedError(
                reasons        = decision.reasons,
                score          = decision.score,
                decision       = decision,
                policy_name    = self._policy.name,
                prompt_snippet = _extract_last_user_content(messages),
            )

        return decision

    async def _extract_messages(
        self,
        request: Request,
    ) -> List[Dict[str, str]]:
        try:
            body = await request.json()
        except Exception:
            return []

        if not isinstance(body, dict):
            return []

        # Try configured field first
        if self._messages_field:
            value = body.get(self._messages_field)
            if isinstance(value, list):
                return _normalise_messages(value)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]

        # Auto-detect
        for field in MESSAGE_FIELD_NAMES:
            value = body.get(field)
            if isinstance(value, list) and value:
                return _normalise_messages(value)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]

        return []

class LLMSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app:             ASGIApp,
        policy:          Optional[Policy]         = None,
        scan_paths:      Sequence[str]            = DEFAULT_SCAN_PATHS,
        scan_output:     bool                     = True,
        roles_to_scan:   Sequence[str]            = ("user",),
        block_status:    int                      = 400,
        include_reasons: bool                     = True,
        log_format:      LogFormat                = LogFormat.JSON,
    ) -> None:
        super().__init__(app)
        self._policy          = policy or get_default_policy()
        self._scan_paths      = list(scan_paths)
        self._scan_output     = scan_output
        self._roles_to_scan   = list(roles_to_scan)
        self._block_status    = block_status
        self._include_reasons = include_reasons
        self._sec_logger      = SecurityEventLogger(
            policy        = self._policy,
            provider_name = "fastapi-middleware",
            fmt           = log_format,
        )

    def _should_scan(self, path: str) -> bool:
        return any(path.startswith(p) for p in self._scan_paths)

    async def dispatch(
        self,
        request:  Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not self._should_scan(request.url.path):
            return await call_next(request)

        start = time.perf_counter()
        messages  = await self._extract_messages(request)
        prompt_ok = True
        p_decision: Optional[GuardDecision] = None

        if messages:
            per_msg  = scan_messages(
                messages, self._policy, roles_to_scan=self._roles_to_scan
            )
            combined = aggregate_prompt_scans(per_msg, self._policy)
            action   = self._policy.action_for_score(combined.score, GuardType.PROMPT)

            p_decision = GuardDecision(
                allowed      = combined.allowed,
                score        = round(combined.score, 4),
                reasons      = combined.reasons,
                safe_output  = None,
                warned       = action == PolicyAction.WARN,
                scan_results = list(per_msg),
                action       = action,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._sec_logger.log_decision(p_decision, duration_ms=elapsed_ms)

            if not combined.allowed:
                return _blocked_json(
                    p_decision,
                    status_code      = self._block_status,
                    include_reasons  = self._include_reasons,
                )

        response = await call_next(request)

        if not self._scan_output:
            return response

        # Only scan buffered JSON responses — skip streams / binary / etc.
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        body_bytes = b""
        # Support both sync iterators (stdlib/tests) and async iterators (real starlette)
        iterator = response.body_iterator
        if hasattr(iterator, "__aiter__"):
            async for chunk in iterator:
                body_bytes += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        else:
            for chunk in iterator:
                body_bytes += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")

        text = self._extract_output_text(body_bytes)
        if not text:
            # Rebuild response unchanged
            return Response(
                content    = body_bytes,
                status_code = response.status_code,
                headers     = dict(response.headers),
                media_type  = content_type,
            )

        out_scan = scan_and_redact(text, self._policy)
        out_action = self._policy.action_for_score(out_scan.score, GuardType.OUTPUT)

        o_decision = GuardDecision(
            allowed      = out_scan.allowed,
            score        = round(out_scan.score, 4),
            reasons      = out_scan.reasons,
            safe_output  = out_scan.safe_output,
            warned       = out_action == PolicyAction.WARN,
            scan_results = [out_scan],
            action       = out_action,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._sec_logger.log_decision(o_decision, duration_ms=elapsed_ms)

        if not out_scan.allowed:
            return _blocked_json(
                o_decision,
                status_code     = self._block_status,
                include_reasons = self._include_reasons,
            )

        # Output passed — rebuild the response with the (possibly redacted) body
        safe_text = out_scan.safe_output or text
        new_body  = self._rebuild_output_body(body_bytes, text, safe_text)

        return Response(
            content     = new_body,
            status_code = response.status_code,
            headers     = dict(response.headers),
            media_type  = content_type,
        )

    async def _extract_messages(
        self,
        request: Request,
    ) -> List[Dict[str, str]]:
        try:
            body_bytes = await request.body()
        except Exception:
            return []

        # Stash bytes so the handler can still call request.body()
        request.state.__dict__["_body"] = body_bytes

        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

        if not isinstance(body, dict):
            return []

        for field in MESSAGE_FIELD_NAMES:
            value = body.get(field)
            if isinstance(value, list) and value:
                return _normalise_messages(value)
            if isinstance(value, str) and value.strip():
                return [{"role": "user", "content": value}]

        return []

    def _extract_output_text(self, body_bytes: bytes) -> str:
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            return ""

        if not isinstance(body, dict):
            return ""

        # OpenAI-style choices array
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content
                # delta style (streaming partial — skip)
                delta = first.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content

        # Generic field scan
        for field in OUTPUT_FIELD_NAMES:
            value = body.get(field)
            if isinstance(value, str) and value.strip():
                return value

        return ""

    def _rebuild_output_body(
        self,
        original_bytes: bytes,
        original_text:  str,
        safe_text:      str,
    ) -> bytes:
        if original_text == safe_text:
            return original_bytes

        try:
            body = json.loads(original_bytes.decode("utf-8"))

            # OpenAI choices style
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and msg.get("content") == original_text:
                    choices[0]["message"]["content"] = safe_text
                    return json.dumps(body).encode("utf-8")

            # Generic field replacement
            for field in OUTPUT_FIELD_NAMES:
                if body.get(field) == original_text:
                    body[field] = safe_text
                    return json.dumps(body).encode("utf-8")
        except Exception:
            pass

        # Plain text substitution fallback
        return original_bytes.replace(
            original_text.encode("utf-8"),
            safe_text.encode("utf-8"),
        )

async def _handle_prompt_blocked(
    request: Request,
    exc:     PromptBlockedError,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error":   "prompt_blocked",
        "message": "The request was blocked by the prompt security policy.",
        "score":   exc.score,
        "reasons": exc.reasons,
    }
    return JSONResponse(content=body, status_code=400)

async def _handle_output_blocked(
    request: Request,
    exc:     OutputBlockedError,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error":   "output_blocked",
        "message": "The model response was blocked by the output security policy.",
        "score":   exc.score,
        "reasons": exc.reasons,
    }
    return JSONResponse(content=body, status_code=400)

async def _handle_blocked_by_policy(
    request: Request,
    exc:     BlockedByPolicyError,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error":   "blocked_by_policy",
        "message": "The request was blocked by the security policy.",
        "score":   exc.score,
        "reasons": exc.reasons,
    }
    return JSONResponse(content=body, status_code=400)

def add_exception_handlers(app: Any) -> None:
    app.add_exception_handler(PromptBlockedError,    _handle_prompt_blocked)
    app.add_exception_handler(OutputBlockedError,    _handle_output_blocked)
    app.add_exception_handler(BlockedByPolicyError,  _handle_blocked_by_policy)

def _normalise_messages(raw: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            content = item.get("content", "")
            role    = item.get("role", "user")
            if isinstance(content, str):
                out.append({"role": str(role), "content": content})
        elif isinstance(item, str) and item.strip():
            out.append({"role": "user", "content": item})
    return out


def _extract_last_user_content(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")[:120]
    return ""

__all__ = [
    "LLMSecurityMiddleware",
    "GuardedRoute",
    "guard_messages",
    "guard_output",
    "decision_response",
    "add_exception_handlers",
    "GUARD_DECISION_KEY",
    "GUARD_POLICY_KEY",
]