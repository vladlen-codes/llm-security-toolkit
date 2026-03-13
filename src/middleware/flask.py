from __future__ import annotations
import functools
import io
import json
import logging
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from ..config import get_default_policy
from ..exceptions import BlockedByPolicyError, OutputBlockedError, PromptBlockedError
from ..guards.outputs import scan_and_redact
from ..guards.prompts import aggregate_prompt_scans, scan_messages
from ..logging import LogFormat, SecurityEventLogger
from ..policies import Policy
from ..types import GuardDecision, GuardType, PolicyAction, ScanResult

_logger = logging.getLogger(__name__)
#: Flask ``g`` attribute where ``guard_route`` stores the GuardDecision.
GUARD_DECISION_KEY = "llm_security_decision"

#: Flask ``g`` attribute where ``guard_route`` stores the active policy.
GUARD_POLICY_KEY = "llm_security_policy"

#: Default path prefixes the WSGI middleware inspects.
DEFAULT_SCAN_PATHS = ("/chat", "/v1/chat", "/v1/messages", "/api/chat")

#: JSON field names to look for messages in request bodies.
MESSAGE_FIELD_NAMES = ("messages", "prompt", "input", "query", "text")

#: JSON field names to look for text in response bodies.
OUTPUT_FIELD_NAMES = ("content", "output", "text", "response", "message", "choices")

# WSGI status strings considered JSON-safe for output scanning.
_JSON_CONTENT_TYPES = ("application/json",)

def _blocked_json_bytes(
    decision: GuardDecision,
    status_code: int = 400,
    include_reasons: bool = True,
) -> Tuple[bytes, str, int]:
    body: Dict[str, Any] = {
        "error":   "request_blocked",
        "message": "This request was blocked by the LLM security policy.",
        "score":   decision.score,
        "action":  decision.action.value,
    }
    if include_reasons:
        body["reasons"] = decision.reasons
    return json.dumps(body).encode("utf-8"), "application/json", status_code

def decision_response(
    decision: GuardDecision,
    *,
    blocked_status_code: int  = 400,
    include_reasons:     bool = True,
) -> Optional[Any]:
    if not decision.allowed:
        try:
            from flask import jsonify
            body: Dict[str, Any] = {
                "error":   "request_blocked",
                "message": "This request was blocked by the LLM security policy.",
                "score":   decision.score,
                "action":  decision.action.value,
            }
            if include_reasons:
                body["reasons"] = decision.reasons
            resp = jsonify(body)
            resp.status_code = blocked_status_code
            return resp
        except ImportError:
            raise RuntimeError(
                "Flask is required to use decision_response(). "
                "Install it with: pip install flask"
            )
    return None

def guard_messages(
    messages: List[Dict[str, str]],
    policy:   Optional[Policy] = None,
    *,
    roles_to_scan:  List[str]             = ("user",),
    raise_on_block: Optional[bool]        = None,
    extra:          Optional[Dict[str, Any]] = None,
) -> GuardDecision:
    active_policy = policy or get_default_policy()
    per_msg       = scan_messages(messages, active_policy, roles_to_scan=list(roles_to_scan))
    combined      = aggregate_prompt_scans(per_msg, active_policy)
    action        = active_policy.action_for_score(combined.score, GuardType.PROMPT)

    decision = GuardDecision(
        allowed      = combined.allowed,
        score        = round(combined.score, 4),
        reasons      = combined.reasons,
        safe_output  = None,
        warned       = action == PolicyAction.WARN,
        scan_results = list(per_msg),
        action       = action,
    )

    should_raise = raise_on_block if raise_on_block is not None else active_policy.raise_on_block

    if not combined.allowed and should_raise:
        raise PromptBlockedError(
            reasons        = combined.reasons,
            score          = combined.score,
            decision       = decision,
            policy_name    = active_policy.name,
            prompt_snippet = _extract_last_user_content(messages),
        )

    return decision

def guard_output(
    text:   str,
    policy: Optional[Policy] = None,
    *,
    raise_on_block: Optional[bool]        = None,
    extra:          Optional[Dict[str, Any]] = None,
) -> GuardDecision:
    active_policy = policy or get_default_policy()
    scan          = scan_and_redact(text, active_policy)
    action        = active_policy.action_for_score(scan.score, GuardType.OUTPUT)

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

def get_decision() -> Optional[GuardDecision]:
    try:
        from flask import g
        return getattr(g, GUARD_DECISION_KEY, None)
    except RuntimeError:
        return None

def guard_route(
    policy:          Optional[Policy]   = None,
    roles_to_scan:   Sequence[str]      = ("user",),
    messages_field:  Optional[str]      = None,
    block_status:    int                = 400,
    include_reasons: bool               = True,
) -> Callable:
    active_policy = policy or get_default_policy()
    _roles        = list(roles_to_scan)
    _sec_logger   = SecurityEventLogger(
        policy        = active_policy,
        provider_name = "flask",
    )

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from flask import g, request

            start    = time.perf_counter()
            messages = _extract_messages_from_flask_request(request, messages_field)

            if not messages:
                clean = GuardDecision.clean(None)
                setattr(g, GUARD_DECISION_KEY, clean)
                setattr(g, GUARD_POLICY_KEY, active_policy)
                return fn(*args, **kwargs)

            per_msg  = scan_messages(messages, active_policy, roles_to_scan=_roles)
            combined = aggregate_prompt_scans(per_msg, active_policy)
            action   = active_policy.action_for_score(combined.score, GuardType.PROMPT)

            decision = GuardDecision(
                allowed      = combined.allowed,
                score        = round(combined.score, 4),
                reasons      = combined.reasons,
                safe_output  = None,
                warned       = action == PolicyAction.WARN,
                scan_results = list(per_msg),
                action       = action,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            _sec_logger.log_decision(decision, duration_ms=elapsed_ms)

            setattr(g, GUARD_DECISION_KEY, decision)
            setattr(g, GUARD_POLICY_KEY, active_policy)

            if not decision.allowed and active_policy.raise_on_block:
                raise PromptBlockedError(
                    reasons        = decision.reasons,
                    score          = decision.score,
                    decision       = decision,
                    policy_name    = active_policy.name,
                    prompt_snippet = _extract_last_user_content(messages),
                )

            return fn(*args, **kwargs)

        return wrapper
    return decorator

class LLMSecurityMiddleware:
    def __init__(
        self,
        app:             Any,
        policy:          Optional[Policy]   = None,
        scan_paths:      Sequence[str]      = DEFAULT_SCAN_PATHS,
        scan_output:     bool               = True,
        roles_to_scan:   Sequence[str]      = ("user",),
        block_status:    int                = 400,
        include_reasons: bool               = True,
        log_format:      LogFormat          = LogFormat.JSON,
    ) -> None:
        self._app             = app
        self._policy          = policy or get_default_policy()
        self._scan_paths      = list(scan_paths)
        self._scan_output     = scan_output
        self._roles_to_scan   = list(roles_to_scan)
        self._block_status    = block_status
        self._include_reasons = include_reasons
        self._sec_logger      = SecurityEventLogger(
            policy        = self._policy,
            provider_name = "flask-middleware",
            fmt           = log_format,
        )

    def __call__(
        self,
        environ:        Dict[str, Any],
        start_response: Callable,
    ) -> Iterable[bytes]:
        path = environ.get("PATH_INFO", "")

        if not self._should_scan(path):
            return self._app(environ, start_response)

        start = time.perf_counter()

        body_bytes, environ = self._read_body(environ)
        messages = _extract_messages_from_bytes(body_bytes)

        if messages:
            per_msg  = scan_messages(messages, self._policy, roles_to_scan=self._roles_to_scan)
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
                return self._blocked_response(p_decision, start_response)

        if not self._scan_output:
            return self._app(environ, start_response)

        response_started: Dict[str, Any] = {}

        def capturing_start_response(
            status: str,
            response_headers: List[Tuple[str, str]],
            exc_info: Any = None,
        ) -> Callable:
            response_started["status"]  = status
            response_started["headers"] = list(response_headers)
            return _noop_write

        app_iter = self._app(environ, capturing_start_response)

        if not response_started:
            return app_iter

        headers      = response_started.get("headers", [])
        content_type = _get_header(headers, "content-type")
        if not any(ct in content_type for ct in _JSON_CONTENT_TYPES):
            _call_start_response(start_response, response_started)
            return app_iter

        # Buffer full body
        buffered = b"".join(
            chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            for chunk in app_iter
        )
        if hasattr(app_iter, "close"):
            app_iter.close()

        text = _extract_output_text(buffered)

        if not text:
            _call_start_response(start_response, response_started)
            return [buffered]

        out_scan   = scan_and_redact(text, self._policy)
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
            return self._blocked_response(o_decision, start_response)

        safe_text = out_scan.safe_output or text
        new_body  = _rebuild_output_body(buffered, text, safe_text)
        new_headers = _update_content_length(headers, len(new_body))
        response_started["headers"] = new_headers
        _call_start_response(start_response, response_started)
        return [new_body]

    def _should_scan(self, path: str) -> bool:
        return any(path.startswith(p) for p in self._scan_paths)

    def _read_body(
        self,
        environ: Dict[str, Any],
    ) -> Tuple[bytes, Dict[str, Any]]:
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or 0)
        except (ValueError, TypeError):
            content_length = 0

        try:
            wsgi_input = environ.get("wsgi.input") or io.BytesIO(b"")
            body = wsgi_input.read(content_length) if content_length > 0 else wsgi_input.read()
        except Exception:
            body = b""

        environ = dict(environ)
        environ["wsgi.input"]     = io.BytesIO(body)
        environ["CONTENT_LENGTH"] = str(len(body))
        return body, environ

    def _blocked_response(
        self,
        decision:       GuardDecision,
        start_response: Callable,
    ) -> List[bytes]:
        body, content_type, status_code = _blocked_json_bytes(
            decision,
            status_code     = self._block_status,
            include_reasons = self._include_reasons,
        )
        status_str = f"{status_code} {'Bad Request' if status_code == 400 else 'Error'}"
        start_response(status_str, [
            ("Content-Type",   content_type),
            ("Content-Length", str(len(body))),
        ])
        return [body]

def register_error_handlers(app: Any) -> None:
    from flask import jsonify

    @app.errorhandler(PromptBlockedError)
    def handle_prompt_blocked(exc: PromptBlockedError):  # type: ignore[misc]
        resp = jsonify({
            "error":   "prompt_blocked",
            "message": "The request was blocked by the prompt security policy.",
            "score":   exc.score,
            "reasons": exc.reasons,
        })
        resp.status_code = 400
        return resp

    @app.errorhandler(OutputBlockedError)
    def handle_output_blocked(exc: OutputBlockedError):  # type: ignore[misc]
        resp = jsonify({
            "error":   "output_blocked",
            "message": "The model response was blocked by the output security policy.",
            "score":   exc.score,
            "reasons": exc.reasons,
        })
        resp.status_code = 400
        return resp

    @app.errorhandler(BlockedByPolicyError)
    def handle_blocked_by_policy(exc: BlockedByPolicyError):  # type: ignore[misc]
        resp = jsonify({
            "error":   "blocked_by_policy",
            "message": "The request was blocked by the security policy.",
            "score":   exc.score,
            "reasons": exc.reasons,
        })
        resp.status_code = 400
        return resp

def _extract_messages_from_flask_request(
    request: Any,
    messages_field: Optional[str] = None,
) -> List[Dict[str, str]]:
    try:
        body = request.get_json(silent=True, force=True)
    except Exception:
        body = None

    if not isinstance(body, dict):
        try:
            body = json.loads(request.data or b"")
        except Exception:
            return []

    if not isinstance(body, dict):
        return []

    if messages_field:
        value = body.get(messages_field)
        if isinstance(value, list):
            return _normalise_messages(value)
        if isinstance(value, str) and value.strip():
            return [{"role": "user", "content": value}]

    for field in MESSAGE_FIELD_NAMES:
        value = body.get(field)
        if isinstance(value, list) and value:
            return _normalise_messages(value)
        if isinstance(value, str) and value.strip():
            return [{"role": "user", "content": value}]

    return []

def _extract_messages_from_bytes(body_bytes: bytes) -> List[Dict[str, str]]:
    if not body_bytes:
        return []
    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except Exception:
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

def _normalise_messages(raw: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            content = item.get("content", "")
            role    = item.get("role", "user")
            if isinstance(content, str) and content.strip():
                out.append({"role": str(role), "content": content})
        elif isinstance(item, str) and item.strip():
            out.append({"role": "user", "content": item})
    return out

def _extract_last_user_content(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")[:120]
    return ""

def _extract_output_text(body_bytes: bytes) -> str:
    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return ""

    if not isinstance(body, dict):
        return ""

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content
            delta = first.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content

    for field in OUTPUT_FIELD_NAMES:
        value = body.get(field)
        if isinstance(value, str) and value.strip():
            return value

    return ""

def _rebuild_output_body(
    original_bytes: bytes,
    original_text:  str,
    safe_text:      str,
) -> bytes:
    if original_text == safe_text:
        return original_bytes

    try:
        body = json.loads(original_bytes.decode("utf-8"))
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict) and msg.get("content") == original_text:
                choices[0]["message"]["content"] = safe_text
                return json.dumps(body).encode("utf-8")
        for field in OUTPUT_FIELD_NAMES:
            if body.get(field) == original_text:
                body[field] = safe_text
                return json.dumps(body).encode("utf-8")
    except Exception:
        pass

    return original_bytes.replace(
        original_text.encode("utf-8"),
        safe_text.encode("utf-8"),
    )

def _get_header(headers: List[Tuple[str, str]], name: str) -> str:
    name_lower = name.lower()
    for k, v in headers:
        if k.lower() == name_lower:
            return v.lower()
    return ""

def _update_content_length(
    headers: List[Tuple[str, str]],
    new_length: int,
) -> List[Tuple[str, str]]:
    updated = [(k, v) for k, v in headers if k.lower() != "content-length"]
    updated.append(("Content-Length", str(new_length)))
    return updated

def _call_start_response(
    start_response:   Callable,
    response_started: Dict[str, Any],
) -> None:
    start_response(response_started["status"], response_started["headers"])

def _noop_write(data: bytes) -> None:
    pass

__all__ = [
    "LLMSecurityMiddleware",
    "guard_route",
    "guard_messages",
    "guard_output",
    "decision_response",
    "get_decision",
    "register_error_handlers",
    "GUARD_DECISION_KEY",
    "GUARD_POLICY_KEY",
]