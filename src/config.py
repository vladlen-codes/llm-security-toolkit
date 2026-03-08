from __future__ import annotations
import copy
import logging
import os
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple
from .policies import (
    BalancedPolicy,
    GuardConfig,
    Policy,
    load_policy_from_dict,
    load_policy_from_yaml,
)
from .types import GuardType

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """
    Base exception for all configuration errors.

    Catch this to handle any config-related failure generically,
    or catch the subclasses for more targeted error handling.
    """


class ConfigValidationError(ConfigError):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        bullet_list = "\n  - ".join(errors)
        super().__init__(
            f"Config validation failed with {len(errors)} error(s):\n  - {bullet_list}"
        )


class ConfigSourceError(ConfigError):
    """
    Raised when a config source (file path, env var) cannot be resolved.

    Example:
        try:
            load_config(yaml_path="missing.yaml")
        except ConfigSourceError as exc:
            print(exc)
    """

def _parse_bool(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ConfigValidationError(
        [f"Cannot parse {value!r} as a boolean. "
         "Use one of: 1/0, true/false, yes/no, on/off"]
    )


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_set(value: str) -> set:
    return set(_parse_csv_list(value))

_ENV_PREFIX = "LLM_SECURITY_"

_ENV_FIELD_MAP: Dict[str, Tuple[str, Any]] = {
    "NAME":                ("name",                str),
    "BLOCK_THRESHOLD":     ("block_threshold",     float),
    "WARN_THRESHOLD":      ("warn_threshold",      float),
    "RAISE_ON_BLOCK":      ("raise_on_block",      _parse_bool),
    "REDACT_ON_WARN":      ("redact_on_warn",      _parse_bool),
    "LOG_CLEAN_REQUESTS":  ("log_clean_requests",  _parse_bool),
    # Tool allowlist: comma-separated list
    "ALLOWED_TOOLS":       ("allowed_tools",       _parse_csv_list),
    # Tool denylist: comma-separated list
    "BLOCKED_TOOLS":       ("blocked_tools",       _parse_csv_set),
    # Per-guard enable toggles
    "PROMPT_GUARD_ENABLED": ("_prompt_guard_enabled", _parse_bool),
    "OUTPUT_GUARD_ENABLED": ("_output_guard_enabled", _parse_bool),
    "TOOL_GUARD_ENABLED":   ("_tool_guard_enabled",   _parse_bool),
}

_VALID_POLICY_FIELDS = {f.name for f in fields(Policy)}

_THRESHOLD_FIELDS = {"block_threshold", "warn_threshold"}

_BOOL_FIELDS = {"raise_on_block", "redact_on_warn", "log_clean_requests"}

_GUARD_KEYS = {"prompt_guard", "output_guard", "tool_guard"}

_VALID_GUARD_FIELDS = {"enabled", "block_threshold", "warn_threshold"}


def validate_config_dict(data: Dict[str, Any]) -> None:
    errors: List[str] = []
    for field_name in _THRESHOLD_FIELDS:
        if field_name in data:
            val = data[field_name]
            if not isinstance(val, (int, float)):
                errors.append(
                    f"{field_name} must be a number, got {type(val).__name__!r}"
                )
            elif not (0.0 <= float(val) <= 1.0):
                errors.append(
                    f"{field_name} must be in [0.0, 1.0], got {val!r}"
                )

    block = data.get("block_threshold", 0.75)
    warn  = data.get("warn_threshold",  0.40)
    try:
        if float(warn) >= float(block):
            errors.append(
                f"warn_threshold ({warn}) must be strictly less than "
                f"block_threshold ({block})"
            )
    except (TypeError, ValueError):
        pass

    for field_name in _BOOL_FIELDS:
        if field_name in data and not isinstance(data[field_name], bool):
            errors.append(
                f"{field_name} must be a boolean (True/False), "
                f"got {type(data[field_name]).__name__!r}"
            )

    if "name" in data:
        if not isinstance(data["name"], str) or not data["name"].strip():
            errors.append("name must be a non-empty string")

    if "allowed_tools" in data and data["allowed_tools"] is not None:
        if not isinstance(data["allowed_tools"], list):
            errors.append(
                f"allowed_tools must be a list of strings or null, "
                f"got {type(data['allowed_tools']).__name__!r}"
            )
        elif not all(isinstance(t, str) for t in data["allowed_tools"]):
            errors.append("allowed_tools must contain only strings")

    if "blocked_tools" in data:
        bt = data["blocked_tools"]
        if not isinstance(bt, (list, set)):
            errors.append(
                f"blocked_tools must be a list or set of strings, "
                f"got {type(bt).__name__!r}"
            )
        elif not all(isinstance(t, str) for t in bt):
            errors.append("blocked_tools must contain only strings")

    for guard_key in _GUARD_KEYS:
        if guard_key not in data:
            continue
        gval = data[guard_key]
        if isinstance(gval, GuardConfig):
            continue
        if not isinstance(gval, dict):
            errors.append(
                f"{guard_key} must be a dict or GuardConfig, "
                f"got {type(gval).__name__!r}"
            )
            continue
        # Check for unknown sub-keys
        unknown = set(gval.keys()) - _VALID_GUARD_FIELDS
        for uk in sorted(unknown):
            errors.append(f"{guard_key}: unknown field {uk!r}")
        # Validate sub-thresholds
        for sub_field in ("block_threshold", "warn_threshold"):
            if sub_field in gval and gval[sub_field] is not None:
                sv = gval[sub_field]
                if not isinstance(sv, (int, float)):
                    errors.append(
                        f"{guard_key}.{sub_field} must be a number, "
                        f"got {type(sv).__name__!r}"
                    )
                elif not (0.0 <= float(sv) <= 1.0):
                    errors.append(
                        f"{guard_key}.{sub_field} must be in [0.0, 1.0], "
                        f"got {sv!r}"
                    )
        # warn < block cross-check inside guard
        g_block = gval.get("block_threshold")
        g_warn  = gval.get("warn_threshold")
        if (
            g_block is not None
            and g_warn is not None
            and isinstance(g_block, (int, float))
            and isinstance(g_warn, (int, float))
            and float(g_warn) >= float(g_block)
        ):
            errors.append(
                f"{guard_key}.warn_threshold ({g_warn}) must be strictly "
                f"less than {guard_key}.block_threshold ({g_block})"
            )

    if errors:
        raise ConfigValidationError(errors)

def _resolve_env_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    guard_toggles: Dict[str, bool] = {}

    for env_suffix, (field_name, coerce_fn) in _ENV_FIELD_MAP.items():
        env_key = f"{_ENV_PREFIX}{env_suffix}"
        raw = os.environ.get(env_key)
        if raw is None:
            continue

        try:
            value = coerce_fn(raw) if coerce_fn is not None else raw
        except (ValueError, TypeError) as exc:
            raise ConfigValidationError(
                [f"Environment variable {env_key}={raw!r} is invalid: {exc}"]
            ) from exc

        if field_name.startswith("_") and field_name.endswith("_enabled"):
            guard_key = field_name[1:-8]
            guard_toggles[guard_key] = value
        else:
            overrides[field_name] = value
            logger.debug("Config override from env: %s=%r", env_key, value)

    for guard_field, enabled in guard_toggles.items():
        if guard_field not in overrides:
            overrides[guard_field] = {}
        if isinstance(overrides[guard_field], dict):
            overrides[guard_field]["enabled"] = enabled

    return overrides

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def load_config(
    *,
    yaml_path: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    use_env: bool = True,
    validate: bool = True,
    base_policy: Optional[Policy] = None,
) -> Policy:
    merged: Dict[str, Any] = (base_policy or BalancedPolicy()).to_dict()

    if yaml_path is not None:
        if not os.path.isfile(yaml_path):
            raise ConfigSourceError(
                f"Config YAML file not found: {yaml_path!r}. "
                "Check the path and ensure the file exists."
            )
        try:
            import yaml
        except ImportError as exc:
            raise ConfigSourceError(
                "PyYAML is required to load config from YAML. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(yaml_path, "r", encoding="utf-8") as fh:
            file_data = yaml.safe_load(fh) or {}

        if not isinstance(file_data, dict):
            raise ConfigSourceError(
                f"YAML file {yaml_path!r} must contain a mapping at the top "
                f"level, got {type(file_data).__name__!r}"
            )

        merged = _deep_merge(merged, file_data)
        logger.debug("Loaded config from YAML: %s", yaml_path)

    if data is not None:
        merged = _deep_merge(merged, data)

    if use_env:
        env_overrides = _resolve_env_overrides()
        if env_overrides:
            merged = _deep_merge(merged, env_overrides)
            logger.debug(
                "Applied %d environment variable override(s)", len(env_overrides)
            )

    if validate:
        validate_config_dict(merged)

    return load_policy_from_dict(merged)

_default_policy: Optional[Policy] = None


def get_default_policy() -> Policy:
    global _default_policy
    if _default_policy is None:
        _default_policy = BalancedPolicy()
    return _default_policy


def set_default_policy(policy: Policy) -> None:
    global _default_policy
    if not isinstance(policy, Policy):
        raise TypeError(
            f"set_default_policy() expects a Policy instance, "
            f"got {type(policy).__name__!r}"
        )
    _default_policy = policy
    logger.info("Default policy set: %r", policy)


def reset_default_policy() -> None:
    global _default_policy
    _default_policy = None

__all__ = [
    # Exceptions
    "ConfigError",
    "ConfigValidationError",
    "ConfigSourceError",
    # Primary loader
    "load_config",
    # Validation
    "validate_config_dict",
    # Process-wide default registry
    "get_default_policy",
    "set_default_policy",
    "reset_default_policy",
    # Re-exported for convenience (one import covers all config needs)
    "load_policy_from_dict",
    "load_policy_from_yaml",
]