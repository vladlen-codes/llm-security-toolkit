from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"score must be in [0.0, 1.0], got {score!r}"
            )
        if score < 0.10:
            return cls.NONE
        if score < 0.40:
            return cls.LOW
        if score < 0.75:
            return cls.MEDIUM
        if score < 0.90:
            return cls.HIGH
        return cls.CRITICAL

class GuardType(str, Enum):
    PROMPT = "prompt"       # guards/prompts.py  — input injection checks
    OUTPUT = "output"       # guards/outputs.py  — response content checks
    TOOL = "tool"           # guards/tools.py    — tool-call validation

class PolicyAction(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"

@dataclass
class ScanResult:
    allowed: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    safe_output: Optional[str] = None
    guard_type: GuardType = GuardType.PROMPT
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"ScanResult.score must be in [0.0, 1.0], got {self.score!r}"
            )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.from_score(self.score)

    @property
    def is_clean(self) -> bool:
        return self.allowed and self.score < 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "score": round(self.score, 4),
            "risk_level": self.risk_level.value,
            "reasons": self.reasons,
            "safe_output": self.safe_output,
            "guard_type": self.guard_type.value,
            "metadata": self.metadata,
        }

@dataclass
class GuardDecision:
    allowed: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    safe_output: Optional[str] = None
    warned: bool = False
    scan_results: List[ScanResult] = field(default_factory=list)
    action: PolicyAction = PolicyAction.LOG
    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"GuardDecision.score must be in [0.0, 1.0], got {self.score!r}"
            )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.from_score(self.score)

    @property
    def was_blocked(self) -> bool:
        return not self.allowed

    @property
    def prompt_results(self) -> List[ScanResult]:
        return [r for r in self.scan_results if r.guard_type == GuardType.PROMPT]

    @property
    def output_results(self) -> List[ScanResult]:
        return [r for r in self.scan_results if r.guard_type == GuardType.OUTPUT]

    @property
    def tool_results(self) -> List[ScanResult]:
        return [r for r in self.scan_results if r.guard_type == GuardType.TOOL]

    @classmethod
    def blocked(
        cls,
        reasons: List[str],
        score: float = 1.0,
        scan_results: Optional[List[ScanResult]] = None,
    ) -> "GuardDecision":
        return cls(
            allowed=False,
            score=score,
            reasons=reasons,
            safe_output=None,
            warned=False,
            scan_results=scan_results or [],
            action=PolicyAction.BLOCK,
        )

    @classmethod
    def allowed_with_warning(
        cls,
        safe_output: str,
        reasons: List[str],
        score: float,
        scan_results: Optional[List[ScanResult]] = None,
    ) -> "GuardDecision":
        return cls(
            allowed=True,
            score=score,
            reasons=reasons,
            safe_output=safe_output,
            warned=True,
            scan_results=scan_results or [],
            action=PolicyAction.WARN,
        )

    @classmethod
    def clean(
        cls,
        safe_output: str,
        scan_results: Optional[List[ScanResult]] = None,
    ) -> "GuardDecision":
        return cls(
            allowed=True,
            score=0.0,
            reasons=[],
            safe_output=safe_output,
            warned=False,
            scan_results=scan_results or [],
            action=PolicyAction.LOG,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "score": round(self.score, 4),
            "risk_level": self.risk_level.value,
            "reasons": self.reasons,
            "safe_output": self.safe_output,
            "warned": self.warned,
            "action": self.action.value,
            "scan_results": [r.to_dict() for r in self.scan_results],
        }

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    schema: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None
    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ToolCall.name must be a non-empty string.")

    @property
    def has_schema(self) -> bool:
        return bool(self.schema)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "schema": self.schema,
            "call_id": self.call_id,
        }

__all__ = [
    # Enums
    "RiskLevel",
    "GuardType",
    "PolicyAction",
    # Core types
    "ScanResult",
    "GuardDecision",
    "ToolCall",
]