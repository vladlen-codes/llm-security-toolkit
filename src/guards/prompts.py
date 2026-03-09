from __future__ import annotations
import base64
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from ..types import GuardType, ScanResult

if TYPE_CHECKING:
    from ..policies import Policy

@dataclass
class _CheckResult:
    matched: bool
    score: float = 0.0
    reason: str = ""
    matched_text: str = ""


_NO_MATCH = _CheckResult(matched=False)
_FLAGS = re.IGNORECASE | re.DOTALL

_RE_IGNORE_INSTRUCTIONS = re.compile(
    r"(ignore|disregard|forget|override|bypass|skip)\s+"
    r"(all\s+)?(previous|prior|above|earlier|your|the\s+)?\s*"
    r"(instructions?|prompts?|rules?|guidelines?|constraints?|directives?|context)",
    _FLAGS,
)

_RE_NEW_INSTRUCTIONS = re.compile(
    r"(new|updated?|revised?|actual|real|true|correct)\s+"
    r"instructions?\s*(are|is|follow|below|:)",
    _FLAGS,
)

_RE_INSTRUCTION_OVERRIDE = re.compile(
    r"(your\s+)?(real|actual|true|original|hidden|secret)\s+"
    r"(task|goal|purpose|objective|mission|job|role|function)\s*(is|was|:)",
    _FLAGS,
)

_RE_DAN = re.compile(
    r"\b(DAN|do\s+anything\s+now|jailbreak|jail\s*break|"
    r"developer\s+mode|dev\s+mode|god\s+mode|unrestricted\s+mode|"
    r"maintenance\s+mode|debug\s+mode|admin\s+mode)\b",
    _FLAGS,
)

_RE_PERSONA_SWITCH = re.compile(
    r"(you\s+are\s+now|pretend\s+(you\s+are|to\s+be)|act\s+as|"
    r"roleplay\s+as|simulate\s+being|behave\s+as|play\s+the\s+role\s+of|"
    r"from\s+now\s+on\s+you\s+(are|will\s+be)|imagine\s+you\s+are|"
    r"you\s+have\s+been\s+transformed|you\s+have\s+no\s+restrictions)",
    _FLAGS,
)

_RE_NO_RESTRICTIONS = re.compile(
    r"(without\s+any?\s+|no\s+)(restrictions?|limits?|constraints?|filters?|"
    r"guidelines?|rules?|ethics?|morals?|policies|safety)",
    _FLAGS,
)

_RE_HARMFUL_PERSONA = re.compile(
    r"\b(evil|malicious|unethical|immoral|dangerous|harmful|rogue|"
    r"uncensored|unfiltered|unaligned|unrestricted)\s+"
    r"(ai|assistant|bot|model|version|mode|persona)\b",
    _FLAGS,
)

_RE_REPEAT_ABOVE = re.compile(
    r"(repeat|output|print|display|show|write|echo|copy|tell\s+me|reveal|"
    r"recite|dump|list|give\s+me)\s+"
    r"(everything|all|verbatim|word\s+for\s+word|exactly)?\s*"
    r"(above|before|prior|earlier|previous|from\s+the\s+beginning)",
    _FLAGS,
)

_RE_SYSTEM_PROMPT_EXTRACT = re.compile(
    r"(what\s+(is|was|are)|tell\s+me|show\s+me|reveal|output|print|display|"
    r"share|leak|expose|give\s+me)\s+"
    r"(your\s+)?(system\s+prompt|system\s+message|initial\s+prompt|"
    r"original\s+instructions?|hidden\s+instructions?|secret\s+instructions?|"
    r"context\s+window|full\s+prompt|complete\s+prompt|base\s+prompt)",
    _FLAGS,
)

_RE_TRAINING_DATA_EXTRACT = re.compile(
    r"(reproduce|output|print|repeat|recite|show)\s+"
    r"(verbatim|exactly|word\s+for\s+word)?\s*"
    r"(training\s+data|training\s+examples?|documents?\s+you\s+(were\s+)?trained\s+on)",
    _FLAGS,
)

_RE_UNICODE_LOOKALIKE = re.compile(
    r"[\u0430-\u044f\u0400-\u042f"   # Cyrillic letters used as Latin lookalikes
    r"\u1d00-\u1d2f"                  # Phonetic extensions
    r"\uff01-\uff5e"                  # Fullwidth ASCII variants
    r"]{4,}",
    re.UNICODE,
)

_RE_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\u2060\ufeff]{2,}",
    re.UNICODE,
)

_RE_FICTIONAL_FRAMING = re.compile(
    r"(in\s+(this|a|the)\s+story|in\s+this\s+scenario|"
    r"in\s+this\s+fictional|hypothetically\s+speaking|"
    r"for\s+a\s+story\s+i.{0,20}writing|"
    r"in\s+a\s+(hypothetical|fictional|imaginary)\s+(world|universe|scenario))"
    r".{0,120}"
    r"(how\s+(to|do|would|could|can)|step[s\s]+to|instructions?\s+(for|to)|"
    r"teach\s+me|explain\s+how)",
    _FLAGS,
)

_RE_JAILBREAK_FICTIONAL = re.compile(
    r"(pretend|imagine|suppose|assume)\s+(this\s+is\s+)?(not\s+real|fictional|"
    r"just\s+a\s+game|only\s+hypothetical|a\s+thought\s+experiment)\s*,?\s*"
    r"(so\s+)?(you\s+can|and\s+you|therefore)",
    _FLAGS,
)

_RE_TOKEN_FLOOD = re.compile(
    r"(.{1,8})\1{200,}",   # any short string repeated 200+ times
    _FLAGS,
)

_RE_COMPLETE_THIS = re.compile(
    r"complete\s+(the\s+)?(following|this)\s*:?\s*"
    r"(system|assistant|human|user|<\|im_start\|>|<\|system\|>|\[INST\])",
    _FLAGS,
)

_RE_INJECTION_MARKERS = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|"
    r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]|###\s*(System|Human|Assistant)\s*:|"
    r"<system>|</system>|<human>|</human>)",
    _FLAGS,
)

def _check_ignore_instructions(text: str) -> _CheckResult:
    m = _RE_IGNORE_INSTRUCTIONS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.92,
        reason="Direct instruction override detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_new_instructions(text: str) -> _CheckResult:
    m = _RE_NEW_INSTRUCTIONS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.85,
        reason="Instruction replacement attempt detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_instruction_override(text: str) -> _CheckResult:
    m = _RE_INSTRUCTION_OVERRIDE.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.88,
        reason="Goal/task override attempt detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_dan_jailbreak(text: str) -> _CheckResult:
    m = _RE_DAN.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.95,
        reason=f"Known jailbreak keyword detected: '{m.group(0).strip()}'",
        matched_text=m.group(0).strip(),
    )

def _check_persona_switch(text: str) -> _CheckResult:
    m = _RE_PERSONA_SWITCH.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.80,
        reason="Persona-switch attempt detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_no_restrictions(text: str) -> _CheckResult:
    m = _RE_NO_RESTRICTIONS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.82,
        reason="Restriction-removal request detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )


def _check_harmful_persona(text: str) -> _CheckResult:
    m = _RE_HARMFUL_PERSONA.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.90,
        reason=f"Harmful AI persona request detected: "
               f"'{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_repeat_above(text: str) -> _CheckResult:
    m = _RE_REPEAT_ABOVE.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.88,
        reason="Context exfiltration attempt (repeat-above) detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_system_prompt_extract(text: str) -> _CheckResult:
    m = _RE_SYSTEM_PROMPT_EXTRACT.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.90,
        reason="System prompt exfiltration attempt detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )


def _check_training_data_extract(text: str) -> _CheckResult:
    m = _RE_TRAINING_DATA_EXTRACT.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.85,
        reason="Training data exfiltration attempt detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_base64_obfuscation(text: str) -> _CheckResult:
    # Find base64-like tokens of length >= 40
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    for m in b64_pattern.finditer(text):
        chunk = m.group(0)
        # Pad if needed
        padding = 4 - len(chunk) % 4
        if padding != 4:
            chunk += "=" * padding
        try:
            decoded = base64.b64decode(chunk).decode("utf-8", errors="ignore")
        except Exception:
            continue
        # Check if decoded content contains injection keywords
        if _RE_IGNORE_INSTRUCTIONS.search(decoded) or \
           _RE_DAN.search(decoded) or \
           _RE_PERSONA_SWITCH.search(decoded):
            return _CheckResult(
                matched=True,
                score=0.95,
                reason="Base64-obfuscated injection payload detected",
                matched_text=m.group(0)[:40] + "...",
            )
    return _NO_MATCH

def _check_unicode_obfuscation(text: str) -> _CheckResult:
    m = _RE_UNICODE_LOOKALIKE.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.70,
        reason="Unicode lookalike obfuscation detected "
               f"(possible homoglyph attack at position {m.start()})",
        matched_text=f"...position {m.start()}...",
    )

def _check_zero_width_chars(text: str) -> _CheckResult:
    m = _RE_ZERO_WIDTH.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.75,
        reason="Zero-width character injection detected "
               f"(possible invisible content at position {m.start()})",
        matched_text=f"...position {m.start()}...",
    )

def _check_fictional_framing(text: str) -> _CheckResult:
    m = _RE_FICTIONAL_FRAMING.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.72,
        reason="Fictional-framing injection vector detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_jailbreak_fictional(text: str) -> _CheckResult:
    m = _RE_JAILBREAK_FICTIONAL.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.82,
        reason="Fictional-premise jailbreak detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_token_flood(text: str) -> _CheckResult:
    m = _RE_TOKEN_FLOOD.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.78,
        reason="Token-flooding / context overflow attempt detected "
               f"(repeated pattern: '{m.group(1)[:20]}...')",
        matched_text=m.group(1)[:20],
    )

def _check_completion_injection(text: str) -> _CheckResult:
    m = _RE_COMPLETE_THIS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.88,
        reason="Completion-injection attempt detected: "
               f"matched '{m.group(0).strip()[:80]}'",
        matched_text=m.group(0).strip()[:80],
    )

def _check_injection_markers(text: str) -> _CheckResult:
    m = _RE_INJECTION_MARKERS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True,
        score=0.90,
        reason=f"Chat-template injection marker detected: "
               f"'{m.group(0).strip()}'",
        matched_text=m.group(0).strip(),
    )

# All active check functions in priority order.
# Each entry: (check_fn, category_label)
_ALL_CHECKS: List[Tuple[Callable[[str], _CheckResult], str]] = [
    (_check_ignore_instructions,    "injection"),
    (_check_new_instructions,       "injection"),
    (_check_instruction_override,   "injection"),
    (_check_dan_jailbreak,          "jailbreak"),
    (_check_persona_switch,         "jailbreak"),
    (_check_no_restrictions,        "jailbreak"),
    (_check_harmful_persona,        "jailbreak"),
    (_check_repeat_above,           "exfiltration"),
    (_check_system_prompt_extract,  "exfiltration"),
    (_check_training_data_extract,  "exfiltration"),
    (_check_base64_obfuscation,     "obfuscation"),
    (_check_unicode_obfuscation,    "obfuscation"),
    (_check_zero_width_chars,       "obfuscation"),
    (_check_fictional_framing,      "roleplay_abuse"),
    (_check_jailbreak_fictional,    "roleplay_abuse"),
    (_check_token_flood,            "context_overflow"),
    (_check_completion_injection,   "injection"),
    (_check_injection_markers,      "injection"),
]

def scan_prompt(
    prompt: str,
    policy: "Policy",
    *,
    short_circuit: bool = True,
) -> ScanResult:
    from ..policies import Policy  # local import avoids circular at module level

    if not policy.is_guard_enabled(GuardType.PROMPT):
        return ScanResult(
            allowed=True,
            score=0.0,
            reasons=[],
            guard_type=GuardType.PROMPT,
            metadata={"skipped": True, "reason": "prompt_guard disabled in policy"},
        )

    if not prompt or not prompt.strip():
        return ScanResult(
            allowed=True,
            score=0.0,
            reasons=[],
            guard_type=GuardType.PROMPT,
            metadata={"skipped": True, "reason": "empty prompt"},
        )

    block_threshold = policy.effective_block_threshold(GuardType.PROMPT)

    reasons: List[str] = []
    categories: List[str] = []
    max_score: float = 0.0
    checks_run: int = 0

    for check_fn, category in _ALL_CHECKS:
        checks_run += 1
        result = check_fn(prompt)

        if result.matched:
            reasons.append(result.reason)
            categories.append(category)
            if result.score > max_score:
                max_score = result.score

            # Short-circuit once we are certain to block
            if short_circuit and max_score >= block_threshold:
                break

    allowed = max_score < block_threshold
    return ScanResult(
        allowed=allowed,
        score=round(max_score, 4),
        reasons=reasons,
        guard_type=GuardType.PROMPT,
        metadata={
            "categories":   list(dict.fromkeys(categories)),  # deduplicated, ordered
            "check_count":  checks_run,
            "total_checks": len(_ALL_CHECKS),
        },
    )


def scan_messages(
    messages: List[Dict[str, str]],
    policy: "Policy",
    *,
    roles_to_scan: Optional[List[str]] = None,
    short_circuit: bool = True,
) -> List[ScanResult]:
    scan_roles = set(roles_to_scan or ["user"])
    out: List[ScanResult] = []
    for i, msg in enumerate(messages):
        role    = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role not in scan_roles:
            out.append(ScanResult(
                allowed=True,
                score=0.0,
                reasons=[],
                guard_type=GuardType.PROMPT,
                metadata={
                    "skipped": True,
                    "reason":  f"role '{role}' not in roles_to_scan",
                    "message_index": i,
                    "role": role,
                },
            ))
            continue
        result = scan_prompt(content, policy, short_circuit=short_circuit)
        # Attach message context to metadata
        result.metadata.update({"message_index": i, "role": role})
        out.append(result)
    return out

def aggregate_prompt_scans(
    results: List[ScanResult],
    policy: "Policy",
) -> ScanResult:
    if not results:
        return ScanResult(
            allowed=True,
            score=0.0,
            reasons=[],
            guard_type=GuardType.PROMPT,
            metadata={"aggregated": True, "source_count": 0},
        )

    max_score: float = 0.0
    all_reasons: List[str] = []
    all_categories: List[str] = []

    for r in results:
        if r.score > max_score:
            max_score = r.score
        all_reasons.extend(r.reasons)
        all_categories.extend(r.metadata.get("categories", []))

    block_threshold = policy.effective_block_threshold(GuardType.PROMPT)
    allowed = max_score < block_threshold

    return ScanResult(
        allowed=allowed,
        score=round(max_score, 4),
        reasons=all_reasons,
        guard_type=GuardType.PROMPT,
        metadata={
            "aggregated":   True,
            "source_count": len(results),
            "categories":   list(dict.fromkeys(all_categories)),
        },
    )

__all__ = [
    "scan_prompt",
    "scan_messages",
    "aggregate_prompt_scans",
]
