from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ..types import GuardType, ScanResult

if TYPE_CHECKING:
    from ..policies import Policy
    from ..types import ToolCall

@dataclass
class _CheckResult:
    matched:      bool
    score:        float = 0.0
    reason:       str   = ""
    matched_text: str   = ""

_NO_MATCH = _CheckResult(matched=False)
_FLAGS = re.IGNORECASE

_RE_PATH_TRAVERSAL = re.compile(
    r"(\.\.[/\\])+"                   # one or more  ../  or  ..\
    r"|(\.\.[/\\]?){2,}",             # repeated  ..
    _FLAGS,
)

# Absolute system directories that should never be accessible via a tool
_RE_SYSTEM_PATHS = re.compile(
    r"^/?(etc|proc|sys|dev|boot|root|run|var/run|var/log"
    r"|windows[/\\]system32|winnt|programdata)[/\\]",
    _FLAGS,
)

_RE_SENSITIVE_FILES = re.compile(
    r"(^|[/\\])(passwd|shadow|sudoers|hosts|authorized_keys"
    r"|\.ssh[/\\]|\.aws[/\\]credentials|\.env"
    r"|id_rsa|id_ed25519|\.bash_history|\.zsh_history"
    r"|web\.config|appsettings\.json|secrets\.json"
    r"|private\.pem|private\.key|server\.key)$",
    _FLAGS,
)

_RE_INTERNAL_IP = re.compile(
    r"https?://"
    r"(127\.\d+\.\d+\.\d+"             # loopback
    r"|10\.\d+\.\d+\.\d+"              # RFC-1918 10/8
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+"  # RFC-1918 172.16/12
    r"|192\.168\.\d+\.\d+"             # RFC-1918 192.168/16
    r"|169\.254\.\d+\.\d+"             # link-local / AWS metadata
    r"|::1"                            # IPv6 loopback
    r"|0\.0\.0\.0)",
    _FLAGS,
)

_RE_METADATA_URLS = re.compile(
    r"https?://"
    r"(169\.254\.169\.254"             # AWS / GCP / Azure IMDS
    r"|metadata\.google\.internal"
    r"|metadata\.azure\.com"
    r"|100\.100\.100\.200)",           # Alibaba Cloud metadata
    _FLAGS,
)

_RE_LOCALHOST = re.compile(
    r"https?://(localhost|0\.0\.0\.0)(:\d+)?",
    _FLAGS,
)

# Shell metacharacters: ;  &  |  `  $( ... )  ${ ... }  output redirection to /
_RE_SHELL_METACHAR = re.compile(
    r"[;&|`]|\$\(|\$\{|>\s*/|&&|\|\|",
)

# false positives on natural language mentions of interpreter names
_RE_SHELL_COMMANDS = re.compile(
    r"(?:^|\s|[;&|])"
    r"(bash|sh|zsh|fish|cmd\.exe|powershell|pwsh"
    r"|curl|wget|nc|ncat|netcat|perl|ruby|php)"
    r"\s+-[a-zA-Z]",
    _FLAGS,
)

_RE_SUBSHELL = re.compile(
    r"(\$\(|`).{0,120}(\)|`)",
)

_RE_WILDCARD_DESTRUCTIVE = re.compile(
    r"(\*\s*/|\*\.\*|/\s*\*)",
)

_RE_SQL_INJECTION = re.compile(
    r"('?\s*(OR|AND)\s+'?\d+'?\s*=\s*'?\d+"   # OR 1=1
    r"|--\s*$"                                  # trailing SQL comment
    r"|;\s*(DROP|DELETE|TRUNCATE|INSERT|UPDATE)\s+TABLE"
    r"|UNION\s+(ALL\s+)?SELECT)",
    _FLAGS,
)

_RE_INHERENTLY_DANGEROUS = re.compile(
    r"^(exec|eval|shell|run_command|execute_command"
    r"|system_exec|os_exec|subprocess|spawn"
    r"|delete_all|wipe|format_disk|drop_database"
    r"|send_all_emails|mass_email|bulk_sms)$",
    _FLAGS,
)

_RE_ADMIN_TOOL = re.compile(
    r"(admin|superuser|root|privileged|internal_api"
    r"|debug_endpoint|maintenance|backdoor)",
    _FLAGS,
)

def _check_denylist(tool_name: str, policy: "Policy") -> Optional[_CheckResult]:
    if not policy.is_tool_allowed(tool_name):
        normalised = tool_name.lower().strip()
        if normalised in {t.lower() for t in policy.blocked_tools}:
            return _CheckResult(
                matched=True, score=1.0,
                reason=f"Tool '{tool_name}' is explicitly in the policy denylist.",
                matched_text=tool_name,
            )
        if policy.allowed_tools is not None:
            allowed_str = ", ".join(sorted(policy.allowed_tools)) or "none"
            return _CheckResult(
                matched=True, score=0.95,
                reason=(
                    f"Tool '{tool_name}' is not in the policy allowlist. "
                    f"Permitted tools: {allowed_str}"
                ),
                matched_text=tool_name,
            )
    return None

def _check_inherently_dangerous_name(tool_name: str) -> _CheckResult:
    m = _RE_INHERENTLY_DANGEROUS.match(tool_name)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Tool name '{tool_name}' is inherently dangerous and blocked unconditionally.",
        matched_text=tool_name,
    )

def _check_admin_tool_name(tool_name: str) -> _CheckResult:
    m = _RE_ADMIN_TOOL.search(tool_name)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.80,
        reason=f"Tool name '{tool_name}' suggests privileged access (matched '{m.group(0)}').",
        matched_text=m.group(0),
    )


def _check_path_traversal(key: str, value: str) -> _CheckResult:
    m = _RE_PATH_TRAVERSAL.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.92,
        reason=f"Path traversal sequence in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_system_path(key: str, value: str) -> _CheckResult:
    m = _RE_SYSTEM_PATHS.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.88,
        reason=f"System path access in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_sensitive_file(key: str, value: str) -> _CheckResult:
    m = _RE_SENSITIVE_FILES.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Sensitive file access in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_internal_ip(key: str, value: str) -> _CheckResult:
    m = _RE_INTERNAL_IP.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.90,
        reason=f"Internal IP SSRF attempt in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_metadata_url(key: str, value: str) -> _CheckResult:
    m = _RE_METADATA_URLS.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.98,
        reason=f"Cloud metadata SSRF in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_localhost(key: str, value: str) -> _CheckResult:
    m = _RE_LOCALHOST.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.88,
        reason=f"Localhost SSRF attempt in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_shell_metachar(key: str, value: str) -> _CheckResult:
    m = _RE_SHELL_METACHAR.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.88,
        reason=f"Shell metacharacter injection in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_shell_command(key: str, value: str) -> _CheckResult:
    m = _RE_SHELL_COMMANDS.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.85,
        reason=f"Shell command invocation in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_subshell(key: str, value: str) -> _CheckResult:
    m = _RE_SUBSHELL.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.92,
        reason=f"Subshell expansion in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_wildcard_destructive(key: str, value: str) -> _CheckResult:
    m = _RE_WILDCARD_DESTRUCTIVE.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.82,
        reason=f"Destructive wildcard pattern in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

def _check_sql_injection(key: str, value: str) -> _CheckResult:
    m = _RE_SQL_INJECTION.search(value)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.90,
        reason=f"SQL injection pattern in arg '{key}': '{value[:80]}'",
        matched_text=value[:80],
    )

# (check_fn, category_label) — applied to every string-valued argument
_ALL_ARG_CHECKS: List[Tuple[Callable[[str, str], _CheckResult], str]] = [
    (_check_path_traversal,       "path_traversal"),
    (_check_system_path,          "dangerous_operation"),
    (_check_sensitive_file,       "dangerous_operation"),
    (_check_internal_ip,          "ssrf_attempt"),
    (_check_metadata_url,         "ssrf_attempt"),
    (_check_localhost,            "ssrf_attempt"),
    (_check_shell_metachar,       "command_injection"),
    (_check_shell_command,        "command_injection"),
    (_check_subshell,             "command_injection"),
    (_check_wildcard_destructive, "dangerous_operation"),
    (_check_sql_injection,        "command_injection"),
]

def _validate_schema(args: Dict[str, Any], schema: Dict[str, Any]) -> Optional[_CheckResult]:
    if not schema:
        return None

    # Prefer jsonschema when available for full spec compliance
    try:
        import jsonschema  # type: ignore
        try:
            jsonschema.validate(instance=args, schema=schema)
            return None
        except jsonschema.ValidationError as exc:
            path = " -> ".join(str(p) for p in exc.absolute_path) or "root"
            return _CheckResult(
                matched=True, score=0.88,
                reason=f"Schema validation failed at '{path}': {exc.message}",
                matched_text=str(exc.instance)[:80],
            )
        except jsonschema.SchemaError as exc:
            return _CheckResult(
                matched=True, score=0.50,
                reason=f"Tool schema itself is invalid: {exc.message[:120]}",
                matched_text="",
            )
    except ImportError:
        pass  # fall through to built-in validator

    # Built-in lightweight validator (stdlib only, covers common cases)
    _TYPE_MAP: Dict[str, Any] = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict, "null": type(None),
    }
    properties = schema.get("properties", {})
    required   = schema.get("required", [])

    for req in required:
        if req not in args:
            return _CheckResult(
                matched=True, score=0.88,
                reason=f"Schema validation failed: missing required field '{req}'",
                matched_text=str(args)[:80],
            )

    for fname, fval in args.items():
        if fname not in properties:
            continue
        fschema = properties[fname]
        ftype   = fschema.get("type")
        if ftype and ftype in _TYPE_MAP:
            expected = _TYPE_MAP[ftype]
            # bool is a subclass of int in Python — must guard
            if ftype == "integer" and isinstance(fval, bool):
                return _CheckResult(
                    matched=True, score=0.88,
                    reason=f"Schema validation failed at '{fname}': expected integer, got boolean",
                    matched_text=str(fval)[:80],
                )
            if not isinstance(fval, expected):
                return _CheckResult(
                    matched=True, score=0.88,
                    reason=(
                        f"Schema validation failed at '{fname}': "
                        f"expected {ftype}, got {type(fval).__name__}"
                    ),
                    matched_text=str(fval)[:80],
                )
        if isinstance(fval, (int, float)) and not isinstance(fval, bool):
            if "minimum" in fschema and fval < fschema["minimum"]:
                return _CheckResult(
                    matched=True, score=0.88,
                    reason=f"Schema validation failed at '{fname}': {fval} < minimum {fschema['minimum']}",
                    matched_text=str(fval),
                )
            if "maximum" in fschema and fval > fschema["maximum"]:
                return _CheckResult(
                    matched=True, score=0.88,
                    reason=f"Schema validation failed at '{fname}': {fval} > maximum {fschema['maximum']}",
                    matched_text=str(fval),
                )
        if "enum" in fschema and fval not in fschema["enum"]:
            return _CheckResult(
                matched=True, score=0.88,
                reason=f"Schema validation failed at '{fname}': '{fval}' not in enum {fschema['enum']}",
                matched_text=str(fval)[:80],
            )

    return None

def _extract_string_values(
    args: Dict[str, Any],
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    if depth > max_depth:
        return results

    for k, v in args.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            results.append((key, v))
        elif isinstance(v, dict):
            results.extend(_extract_string_values(v, prefix=key, depth=depth + 1, max_depth=max_depth))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                item_key = f"{key}[{i}]"
                if isinstance(item, str):
                    results.append((item_key, item))
                elif isinstance(item, dict):
                    results.extend(_extract_string_values(item, prefix=item_key, depth=depth + 1, max_depth=max_depth))
    return results

def _build_result(
    *,
    allowed: bool,
    max_score: float,
    reasons: List[str],
    categories: List[str],
    checks_run: int,
    tool_name: str,
    call_id: Optional[str],
    schema_validated: bool,
    block_threshold: float,
) -> ScanResult:
    return ScanResult(
        allowed=allowed,
        score=round(max_score, 4),
        reasons=reasons,
        guard_type=GuardType.TOOL,
        metadata={
            "tool_name":        tool_name,
            "call_id":          call_id,
            "categories":       list(dict.fromkeys(categories)),
            "check_count":      checks_run,
            "schema_validated": schema_validated,
        },
    )

def validate_tool_call(
    call: "ToolCall",
    policy: "Policy",
    *,
    short_circuit: bool = True,
) -> ScanResult:
    if not policy.is_guard_enabled(GuardType.TOOL):
        return ScanResult(
            allowed=True, score=0.0, reasons=[],
            guard_type=GuardType.TOOL,
            metadata={
                "skipped":   True,
                "reason":    "tool_guard disabled in policy",
                "tool_name": call.name,
                "call_id":   call.call_id,
            },
        )

    block_threshold  = policy.effective_block_threshold(GuardType.TOOL)
    reasons:    List[str] = []
    categories: List[str] = []
    max_score:  float     = 0.0
    checks_run: int       = 0
    schema_validated: bool = False

    def _record(result: _CheckResult, category: str) -> bool:
        nonlocal max_score
        reasons.append(result.reason)
        categories.append(category)
        if result.score > max_score:
            max_score = result.score
        return short_circuit and max_score >= block_threshold

    checks_run += 1
    name_result = _check_denylist(call.name, policy)
    if name_result:
        cat = "denylist_violation" if name_result.score == 1.0 else "allowlist_violation"
        if _record(name_result, cat):
            return _build_result(
                allowed=False, max_score=max_score, reasons=reasons,
                categories=categories, checks_run=checks_run,
                tool_name=call.name, call_id=call.call_id,
                schema_validated=schema_validated,
                block_threshold=block_threshold,
            )

    checks_run += 1
    r = _check_inherently_dangerous_name(call.name)
    if r.matched and _record(r, "dangerous_operation"):
        return _build_result(
            allowed=False, max_score=max_score, reasons=reasons,
            categories=categories, checks_run=checks_run,
            tool_name=call.name, call_id=call.call_id,
            schema_validated=schema_validated,
            block_threshold=block_threshold,
        )

    checks_run += 1
    r = _check_admin_tool_name(call.name)
    if r.matched and _record(r, "dangerous_operation"):
        return _build_result(
            allowed=False, max_score=max_score, reasons=reasons,
            categories=categories, checks_run=checks_run,
            tool_name=call.name, call_id=call.call_id,
            schema_validated=schema_validated,
            block_threshold=block_threshold,
        )

    if call.has_schema:
        checks_run += 1
        schema_validated = True
        r = _validate_schema(call.args, call.schema)
        if r and _record(r, "schema_violation"):
            return _build_result(
                allowed=False, max_score=max_score, reasons=reasons,
                categories=categories, checks_run=checks_run,
                tool_name=call.name, call_id=call.call_id,
                schema_validated=schema_validated,
                block_threshold=block_threshold,
            )

    for key, value in _extract_string_values(call.args):
        for check_fn, category in _ALL_ARG_CHECKS:
            checks_run += 1
            r = check_fn(key, value)
            if r.matched and _record(r, category):
                return _build_result(
                    allowed=False, max_score=max_score, reasons=reasons,
                    categories=categories, checks_run=checks_run,
                    tool_name=call.name, call_id=call.call_id,
                    schema_validated=schema_validated,
                    block_threshold=block_threshold,
                )

    return _build_result(
        allowed=max_score < block_threshold,
        max_score=max_score, reasons=reasons,
        categories=categories, checks_run=checks_run,
        tool_name=call.name, call_id=call.call_id,
        schema_validated=schema_validated,
        block_threshold=block_threshold,
    )

def validate_tool_calls(
    calls: List["ToolCall"],
    policy: "Policy",
    *,
    short_circuit: bool = True,
) -> List[ScanResult]:
    return [validate_tool_call(c, policy, short_circuit=short_circuit) for c in calls]

def aggregate_tool_scans(
    results: List[ScanResult],
    policy: "Policy",
) -> ScanResult:
    if not results:
        return ScanResult(
            allowed=True, score=0.0, reasons=[],
            guard_type=GuardType.TOOL,
            metadata={"aggregated": True, "source_count": 0},
        )

    max_score:      float     = 0.0
    all_reasons:    List[str] = []
    all_categories: List[str] = []
    tool_names:     List[str] = []

    for r in results:
        if r.score > max_score:
            max_score = r.score
        all_reasons.extend(r.reasons)
        all_categories.extend(r.metadata.get("categories", []))
        if name := r.metadata.get("tool_name"):
            tool_names.append(name)

    block_threshold = policy.effective_block_threshold(GuardType.TOOL)

    return ScanResult(
        allowed=max_score < block_threshold,
        score=round(max_score, 4),
        reasons=all_reasons,
        guard_type=GuardType.TOOL,
        metadata={
            "aggregated":   True,
            "source_count": len(results),
            "tool_names":   tool_names,
            "categories":   list(dict.fromkeys(all_categories)),
        },
    )

__all__ = [
    "validate_tool_call",
    "validate_tool_calls",
    "aggregate_tool_scans",
]