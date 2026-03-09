from __future__ import annotations
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .policies import Policy
    from .types import GuardDecision, GuardType, ScanResult, ToolCall

_internal_logger = logging.getLogger(__name__)

class LogFormat(str, Enum):
    JSON = "json"
    TEXT = "text"

class EventType(str, Enum):
    DECISION  = "decision"
    SCAN      = "scan"
    TOOL_CALL = "tool_call"

@dataclass
class SecurityEvent:
    event_type:          EventType
    timestamp:           str
    allowed:             bool
    score:               float
    risk_level:          str
    reasons:             List[str]
    action:              str
    guard_type:          Optional[str]         = None
    policy_name:         str                   = "unknown"
    provider_name:       str                   = "unknown"
    tool_name:           Optional[str]         = None
    tool_call_id:        Optional[str]         = None
    safe_output_present: bool                  = False
    duration_ms:         Optional[float]       = None
    trace_id:            Optional[str]         = None
    span_id:             Optional[str]         = None
    extra:               Dict[str, Any]        = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def to_text(self) -> str:
        parts = [
            f"event={self.event_type.value}",
            f"ts={self.timestamp[:19]}Z",
            f"allowed={self.allowed}",
            f"score={self.score:.4f}",
            f"risk={self.risk_level}",
            f"action={self.action}",
            f"policy={self.policy_name}",
            f"provider={self.provider_name}",
        ]
        if self.guard_type:
            parts.append(f"guard={self.guard_type}")
        if self.tool_name:
            parts.append(f"tool={self.tool_name}")
        if self.tool_call_id:
            parts.append(f"call_id={self.tool_call_id}")
        if self.duration_ms is not None:
            parts.append(f"duration_ms={self.duration_ms:.1f}")
        if self.trace_id:
            parts.append(f"trace_id={self.trace_id}")
        if self.reasons:
            joined = "; ".join(self.reasons)
            parts.append(f'reasons="{joined}"')
        for k, v in self.extra.items():
            parts.append(f"{k}={v!r}")
        return " ".join(parts)

# produced by any log_* call in this module.
_handlers: List[Callable[[SecurityEvent], None]] = []

def add_handler(handler: Callable[[SecurityEvent], None]) -> None:
    _handlers.append(handler)

def remove_handler(handler: Callable[[SecurityEvent], None]) -> None:
    try:
        _handlers.remove(handler)
    except ValueError:
        pass

def clear_handlers() -> None:
    _handlers.clear()

def _dispatch(event: SecurityEvent) -> None:
    for handler in _handlers:
        try:
            handler(event)
        except Exception:  # noqa: BLE001
            _internal_logger.error(
                "LLM Security Toolkit: custom handler %r raised an exception:\n%s",
                handler,
                traceback.format_exc(),
            )

def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def _stdlib_level(allowed: bool, warned: bool) -> int:
    if not allowed:
        return logging.ERROR
    if warned:
        return logging.WARNING
    return logging.INFO

def _write_to_logger(
    event: SecurityEvent,
    logger: logging.Logger,
    fmt: LogFormat,
) -> None:
    level = _stdlib_level(
        allowed=event.allowed,
        warned=(event.action == "warn"),
    )
    message = event.to_json() if fmt == LogFormat.JSON else event.to_text()
    logger.log(level, message)

def log_decision(
    decision: "GuardDecision",
    *,
    policy: Optional["Policy"] = None,
    provider_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    fmt: LogFormat = LogFormat.JSON,
    duration_ms: Optional[float] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> SecurityEvent:
    target_logger = logger or _internal_logger
    event = SecurityEvent(
        event_type          = EventType.DECISION,
        timestamp           = _utcnow_iso(),
        allowed             = decision.allowed,
        score               = round(decision.score, 4),
        risk_level          = decision.risk_level.value,
        reasons             = list(decision.reasons),
        action              = decision.action.value,
        policy_name         = policy.name if policy else "unknown",
        provider_name       = provider_name,
        safe_output_present = decision.safe_output is not None,
        duration_ms         = duration_ms,
        trace_id            = trace_id,
        span_id             = span_id,
        extra               = extra or {},
    )

    _write_to_logger(event, target_logger, fmt)
    _dispatch(event)
    return event


def log_scan_result(
    result: "ScanResult",
    *,
    policy: Optional["Policy"] = None,
    provider_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    fmt: LogFormat = LogFormat.JSON,
    extra: Optional[Dict[str, Any]] = None,
) -> SecurityEvent:
    target_logger = logger or _internal_logger
    event = SecurityEvent(
        event_type    = EventType.SCAN,
        timestamp     = _utcnow_iso(),
        allowed       = result.allowed,
        score         = round(result.score, 4),
        risk_level    = result.risk_level.value,
        reasons       = list(result.reasons),
        action        = "block" if not result.allowed else "log",
        guard_type    = result.guard_type.value,
        policy_name   = policy.name if policy else "unknown",
        provider_name = provider_name,
        extra         = extra or {},
    )

    _write_to_logger(event, target_logger, fmt)
    _dispatch(event)
    return event

def log_tool_call(
    tool_call: "ToolCall",
    result: Optional["ScanResult"] = None,
    *,
    policy: Optional["Policy"] = None,
    provider_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    fmt: LogFormat = LogFormat.JSON,
    extra: Optional[Dict[str, Any]] = None,
) -> SecurityEvent:
    target_logger = logger or _internal_logger

    allowed    = result.allowed if result else True
    score      = round(result.score, 4) if result else 0.0
    risk_level = result.risk_level.value if result else "none"
    reasons    = list(result.reasons) if result else []
    action     = ("block" if not allowed else "log") if result else "log"

    event = SecurityEvent(
        event_type    = EventType.TOOL_CALL,
        timestamp     = _utcnow_iso(),
        allowed       = allowed,
        score         = score,
        risk_level    = risk_level,
        reasons       = reasons,
        action        = action,
        guard_type    = "tool",
        policy_name   = policy.name if policy else "unknown",
        provider_name = provider_name,
        tool_name     = tool_call.name,
        tool_call_id  = tool_call.call_id,
        extra         = extra or {},
    )

    _write_to_logger(event, target_logger, fmt)
    _dispatch(event)
    return event

class SecurityEventLogger:
    def __init__(
        self,
        policy: Optional["Policy"] = None,
        provider_name: str = "unknown",
        fmt: LogFormat = LogFormat.JSON,
        logger: Optional[logging.Logger] = None,
        default_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy         = policy
        self.provider_name  = provider_name
        self.fmt            = fmt
        self.logger         = logger or _internal_logger
        self.default_extra  = default_extra or {}

    def _merge_extra(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self.default_extra)
        if extra:
            merged.update(extra)
        return merged

    def log_decision(
        self,
        decision: "GuardDecision",
        *,
        duration_ms: Optional[float] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        return log_decision(
            decision,
            policy        = self.policy,
            provider_name = self.provider_name,
            logger        = self.logger,
            fmt           = self.fmt,
            duration_ms   = duration_ms,
            trace_id      = trace_id,
            span_id       = span_id,
            extra         = self._merge_extra(extra),
        )

    def log_scan_result(
        self,
        result: "ScanResult",
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        return log_scan_result(
            result,
            policy        = self.policy,
            provider_name = self.provider_name,
            logger        = self.logger,
            fmt           = self.fmt,
            extra         = self._merge_extra(extra),
        )

    def log_tool_call(
        self,
        tool_call: "ToolCall",
        result: Optional["ScanResult"] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        return log_tool_call(
            tool_call,
            result,
            policy        = self.policy,
            provider_name = self.provider_name,
            logger        = self.logger,
            fmt           = self.fmt,
            extra         = self._merge_extra(extra),
        )

__all__ = [
    # Enums
    "LogFormat",
    "EventType",
    # Event dataclass
    "SecurityEvent",
    # Module-level functions
    "log_decision",
    "log_scan_result",
    "log_tool_call",
    # Handler registry
    "add_handler",
    "remove_handler",
    "clear_handlers",
    # Stateful logger
    "SecurityEventLogger",
]