from __future__ import annotations
import copy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Set
from .types import GuardType, PolicyAction

@dataclass
class GuardConfig:
    enabled: bool = True
    block_threshold: Optional[float] = None
    warn_threshold: Optional[float] = None
    def __post_init__(self) -> None:
        for attr, label in (
            (self.block_threshold, "block_threshold"),
            (self.warn_threshold, "warn_threshold"),
        ):
            if attr is not None and not (0.0 <= attr <= 1.0):
                raise ValueError(
                    f"GuardConfig.{label} must be in [0.0, 1.0], got {attr!r}"
                )
        if (
            self.block_threshold is not None
            and self.warn_threshold is not None
            and self.warn_threshold >= self.block_threshold
        ):
            raise ValueError(
                "GuardConfig.warn_threshold must be strictly less than block_threshold. "
                f"Got warn={self.warn_threshold}, block={self.block_threshold}"
            )

@dataclass
class Policy:
    # Identity
    name: str = "default"

    # Global thresholds
    block_threshold: float = 0.75
    warn_threshold: float = 0.40

    # Failure behaviour
    raise_on_block: bool = True

    # Per-guard configs
    prompt_guard: GuardConfig = field(default_factory=GuardConfig)
    output_guard: GuardConfig = field(default_factory=GuardConfig)
    tool_guard: GuardConfig = field(default_factory=GuardConfig)

    # Tool allowlist / denylist
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Set[str] = field(default_factory=set)

    # Output handling
    redact_on_warn: bool = True

    # Logging
    log_clean_requests: bool = False

    # Arbitrary metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validation
    def __post_init__(self) -> None:
        if not (0.0 <= self.block_threshold <= 1.0):
            raise ValueError(
                f"Policy.block_threshold must be in [0.0, 1.0], "
                f"got {self.block_threshold!r}"
            )
        if not (0.0 <= self.warn_threshold <= 1.0):
            raise ValueError(
                f"Policy.warn_threshold must be in [0.0, 1.0], "
                f"got {self.warn_threshold!r}"
            )
        if self.warn_threshold >= self.block_threshold:
            raise ValueError(
                "Policy.warn_threshold must be strictly less than block_threshold. "
                f"Got warn={self.warn_threshold}, block={self.block_threshold}"
            )
        if not self.name or not self.name.strip():
            raise ValueError("Policy.name must be a non-empty string.")

        # Normalise allowed_tools to lowercase for case-insensitive matching
        if self.allowed_tools is not None:
            self.allowed_tools = [t.lower().strip() for t in self.allowed_tools]

        # Normalise blocked_tools to lowercase
        self.blocked_tools = {t.lower().strip() for t in self.blocked_tools}

    # Per-guard threshold resolution
    def effective_block_threshold(self, guard: GuardType) -> float:
        cfg = self._guard_config(guard)
        return cfg.block_threshold if cfg.block_threshold is not None \
            else self.block_threshold

    def effective_warn_threshold(self, guard: GuardType) -> float:
        cfg = self._guard_config(guard)
        return cfg.warn_threshold if cfg.warn_threshold is not None \
            else self.warn_threshold

    def is_guard_enabled(self, guard: GuardType) -> bool:
        return self._guard_config(guard).enabled

    def _guard_config(self, guard: GuardType) -> GuardConfig:
        mapping = {
            GuardType.PROMPT: self.prompt_guard,
            GuardType.OUTPUT: self.output_guard,
            GuardType.TOOL:   self.tool_guard,
        }
        return mapping[guard]

    # Tool allowlist helpers
    def is_tool_allowed(self, tool_name: str) -> bool:
        normalised = tool_name.lower().strip()
        if normalised in self.blocked_tools:
            return False
        if self.allowed_tools is None:
            return True
        return normalised in self.allowed_tools

    # Action resolution
    def action_for_score(self, score: float, guard: GuardType) -> PolicyAction:
        if score >= self.effective_block_threshold(guard):
            return PolicyAction.BLOCK
        if score >= self.effective_warn_threshold(guard):
            return PolicyAction.WARN
        return PolicyAction.LOG

    # Mutation helpers
    def replace(self, **kwargs: Any) -> "Policy":
        valid = {f.name for f in fields(self)}
        unknown = set(kwargs) - valid
        if unknown:
            raise TypeError(
                f"Policy.replace() got unknown field(s): {sorted(unknown)}"
            )
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(kwargs)
        return Policy(**current)

    def with_allowed_tools(self, tools: List[str]) -> "Policy":
        return self.replace(allowed_tools=tools)

    def with_blocked_tools(self, tools: List[str]) -> "Policy":
        return self.replace(blocked_tools=set(tools))

    # Serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":               self.name,
            "block_threshold":    self.block_threshold,
            "warn_threshold":     self.warn_threshold,
            "raise_on_block":     self.raise_on_block,
            "prompt_guard": {
                "enabled":         self.prompt_guard.enabled,
                "block_threshold": self.prompt_guard.block_threshold,
                "warn_threshold":  self.prompt_guard.warn_threshold,
            },
            "output_guard": {
                "enabled":         self.output_guard.enabled,
                "block_threshold": self.output_guard.block_threshold,
                "warn_threshold":  self.output_guard.warn_threshold,
            },
            "tool_guard": {
                "enabled":         self.tool_guard.enabled,
                "block_threshold": self.tool_guard.block_threshold,
                "warn_threshold":  self.tool_guard.warn_threshold,
            },
            "allowed_tools":      self.allowed_tools,
            "blocked_tools":      sorted(self.blocked_tools),
            "redact_on_warn":     self.redact_on_warn,
            "log_clean_requests": self.log_clean_requests,
            "metadata":           self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"Policy(name={self.name!r}, "
            f"block={self.block_threshold}, "
            f"warn={self.warn_threshold}, "
            f"raise_on_block={self.raise_on_block})"
        )

def StrictPolicy(
    name: str = "strict",
    **overrides: Any,
) -> Policy:
    defaults: Dict[str, Any] = dict(
        name=name,
        block_threshold=0.40,
        warn_threshold=0.15,
        raise_on_block=True,
        redact_on_warn=True,
        log_clean_requests=True,
    )
    defaults.update(overrides)
    return Policy(**defaults)

def BalancedPolicy(
    name: str = "balanced",
    **overrides: Any,
) -> Policy:
    defaults: Dict[str, Any] = dict(
        name=name,
        block_threshold=0.75,
        warn_threshold=0.40,
        raise_on_block=True,
        redact_on_warn=True,
        log_clean_requests=False,
    )
    defaults.update(overrides)
    return Policy(**defaults)

def LoggingOnlyPolicy(
    name: str = "logging-only",
    **overrides: Any,
) -> Policy:
    defaults: Dict[str, Any] = dict(
        name=name,
        block_threshold=1.0,
        warn_threshold=0.0,
        raise_on_block=False,
        redact_on_warn=False,
        log_clean_requests=True,
    )
    defaults.update(overrides)
    return Policy(**defaults)

def load_policy_from_dict(data: Dict[str, Any]) -> Policy:
    data = copy.deepcopy(data)

    # Coerce nested guard dicts into GuardConfig instances
    for guard_key in ("prompt_guard", "output_guard", "tool_guard"):
        if guard_key in data and isinstance(data[guard_key], dict):
            data[guard_key] = GuardConfig(**data[guard_key])

    # Coerce blocked_tools list → set
    if "blocked_tools" in data and isinstance(data["blocked_tools"], list):
        data["blocked_tools"] = set(data["blocked_tools"])

    # Drop unknown keys to stay forward-compatible
    valid_fields = {f.name for f in fields(Policy)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return Policy(**filtered)

def load_policy_from_yaml(path: str) -> Policy:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load policies from YAML files. "
            "Install it with: pip install pyyaml"
        ) from exc

    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Policy YAML file not found: {path!r}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Policy YAML file must contain a mapping at the top level. "
            f"Got {type(data).__name__!r} in {path!r}"
        )

    return load_policy_from_dict(data)

__all__ = [
    # Core class
    "Policy",
    "GuardConfig",
    # Preset factories
    "StrictPolicy",
    "BalancedPolicy",
    "LoggingOnlyPolicy",
    # Loaders
    "load_policy_from_dict",
    "load_policy_from_yaml",
]