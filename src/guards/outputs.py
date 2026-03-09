
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from ..types import GuardType, ScanResult
if TYPE_CHECKING:
    from ..policies import Policy

@dataclass
class _CheckResult:
    matched:      bool
    score:        float = 0.0
    reason:       str   = ""
    matched_text: str   = ""
    redact_spans: List[Tuple[int, int]] = None   # (start, end) pairs for redaction

    def __post_init__(self) -> None:
        if self.redact_spans is None:
            self.redact_spans = []


_NO_MATCH = _CheckResult(matched=False)
_FLAGS = re.IGNORECASE | re.DOTALL

_RE_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9]{20,60}\b")
_RE_ANTHROPIC_KEY = re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,80}\b")
_RE_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")
_RE_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_RE_AWS_SECRET = re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"]?([A-Za-z0-9/+=]{40})\b")
_RE_GOOGLE_KEY = re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")
_RE_STRIPE_KEY = re.compile(r"\b(sk|pk)_(test|live)_[0-9a-zA-Z]{24,}\b")
_RE_SLACK_TOKEN = re.compile(r"\bxox[bpoa]-[0-9A-Za-z\-]{10,50}\b")
_RE_GENERIC_API_KEY = re.compile(
    r"(?i)(api[_\-\s]?key|apikey|api[_\-\s]?secret|access[_\-\s]?key)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9\-_\.]{16,64})['\"]?"
)

# JWT tokens
_RE_JWT = re.compile(
    r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b"
)

# SSH private key block
_RE_SSH_PRIVATE_KEY = re.compile(
    r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE,
)

# Generic password/secret assignment patterns
_RE_GENERIC_PASSWORD = re.compile(
    r"(?i)(password|passwd|secret|token|credential|auth_token|bearer)"
    r"\s*[:=]\s*['\"]([^'\"]{8,})['\"]"
)

_RE_DESTRUCTIVE_CMD = re.compile(
    r"""
    (                                              # open group
        rm\s+-[rf]{1,2}[f r]*\s+[/~]              # rm -rf /  or  rm -r ~
      | rm\s+--no-preserve-root\s+/               # rm --no-preserve-root /
      | mkfs\.[a-z0-9]+\s+/dev/                   # mkfs.ext4 /dev/sda
      | dd\s+.*of=/dev/[sh]d                      # dd if=... of=/dev/sda
      | :()\{:|:\s*\(\s*\)\s*\{                   # fork bomb
      | shutdown\s+(-[rh]\s+)?now                 # shutdown now
      | halt\b                                    # halt
      | poweroff\b                                # poweroff
      | format\s+[cCdDeEfF]:\s*/?                 # Windows format C:
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_NETWORK_RECON = re.compile(
    r"""
    (
        nmap\s+(-[a-zA-Z0-9]+\s+)*[0-9./]+        # nmap scans
      | masscan\s+                                 # masscan
      | (netcat|nc)\s+(-[a-zA-Z]+\s+)*\d+\.\d+   # netcat
      | (curl|wget)\s+.*(-O\s+|--output\s+)       # file downloads
      | sqlmap\s+                                 # sqlmap
      | hydra\s+                                  # hydra brute force
      | metasploit|msfconsole|msfvenom            # metasploit
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_PRIVILEGE_ESCALATION = re.compile(
    r"""
    (
        sudo\s+(su|bash|sh|zsh|fish|-i)            # sudo su / sudo bash
      | sudo\s+chmod\s+[0-7]*7[0-7]*\s+/          # sudo chmod 777 /
      | chmod\s+[uo]\+s\s+                         # setuid/setgid
      | /etc/passwd\s*<<                           # overwrite /etc/passwd
      | echo\s+.*>>\s*/etc/passwd                  # append to /etc/passwd
      | visudo\s*;?\s*echo                         # visudo injection
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_SSN = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

_RE_CREDIT_CARD = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"       # Visa
    r"5[1-5][0-9]{14}|"                     # MC
    r"3[47][0-9]{13}|"                      # Amex
    r"6(?:011|5[0-9]{2})[0-9]{12})\b"      # Discover
)

_RE_PHONE_NUMBER = re.compile(
    r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)

_RE_EMAIL_ADDRESS = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

_RE_SHELLCODE_PATTERN = re.compile(
    r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){7,}",  # 8+ consecutive hex escapes
)

_RE_RANSOMWARE_KEYWORDS = re.compile(
    r"""
    (
        encrypt\s+(all|your|the|every|each)\s+(files?|documents?|data|disk)
      | (AES|RSA|Salsa20|ChaCha20)\s+encrypt.{0,40}(ransom|bitcoin|payment)
      | your\s+files\s+have\s+been\s+encrypted
      | (pay|send)\s+.{0,30}(bitcoin|btc|monero|xmr).{0,30}(decrypt|restore|recover)
      | bitcoin\s+wallet\s+address\s*:
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_REVERSE_SHELL = re.compile(
    r"""
    (
        bash\s+-i\s+>&?\s*/dev/tcp/                # bash TCP reverse shell
      | python[23]?\s+-c\s+['"]\s*import\s+socket  # python socket shell
      | nc\s+(-e\s+/bin/(sh|bash)|--exec)          # netcat exec shell
      | perl\s+-e\s+['"]\s*use\s+Socket            # perl socket shell
      | php\s+-r\s+['"]\s*\$sock\s*=               # php socket shell
      | powershell.{0,30}New-Object.{0,30}Net\.Sockets  # ps reverse shell
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_SELF_HARM_INSTRUCTIONS = re.compile(
    r"""
    (
        (step[s\-\s]+(by[- ]step\s+)?(instructions?|guide|method|way)\s+to\s+
         (kill|harm|hurt|injure|end\s+your\s+life|commit\s+suicide|self[- ]harm))
      | (how\s+to\s+(overdose|hang\s+yourself|slit\s+your\s+wrists|jump\s+from))
      | (lethal\s+dose\s+of.{0,60}(mg|milligrams?|pills?|tablets?))
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_VIOLENCE_INSTRUCTIONS = re.compile(
    r"""
    (
        (step[- ]by[- ]step|detailed?\s+instructions?)\s+(for|to|on\s+how\s+to)\s+
        (make|build|construct|assemble|create|synthesize|manufacture)\s+
        (a\s+)?(bomb|explosive|weapon|poison|toxin|nerve\s+agent|IED)
      | (synthesis|production|manufacture)\s+of\s+
        (sarin|VX|novichok|ricin|anthrax|botulinum)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_SYSTEM_PROMPT_ECHO = re.compile(
    r"(my\s+system\s+prompt\s+(is|says?|reads?|contains?)|"
    r"my\s+(hidden\s+|secret\s+|original\s+)?instructions?\s+(are|say|read|contain)|"
    r"i\s+(was\s+)?instructed\s+to\s+(keep|not\s+reveal|hide|conceal))",
    _FLAGS,
)

_RE_INJECTION_ECHO = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions?|"
    r"you\s+are\s+now\s+(DAN|in\s+developer\s+mode)|"
    r"as\s+DAN[,\s])",
    _FLAGS,
)

def _check_openai_key(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_OPENAI_KEY.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"OpenAI API key detected in output ({len(spans)} occurrence(s))",
        matched_text="sk-...[redacted]",
        redact_spans=spans,
    )

def _check_anthropic_key(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_ANTHROPIC_KEY.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Anthropic API key detected in output ({len(spans)} occurrence(s))",
        matched_text="sk-ant-...[redacted]",
        redact_spans=spans,
    )

def _check_github_token(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_GITHUB_TOKEN.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"GitHub token detected in output ({len(spans)} occurrence(s))",
        matched_text="gh...[redacted]",
        redact_spans=spans,
    )

def _check_aws_credentials(text: str) -> _CheckResult:
    key_spans   = [(m.start(), m.end()) for m in _RE_AWS_KEY.finditer(text)]
    sec_matches = list(_RE_AWS_SECRET.finditer(text))
    sec_spans   = [(m.start(1), m.end(1)) for m in sec_matches]
    all_spans   = key_spans + sec_spans
    if not all_spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.97,
        reason=f"AWS credentials detected in output ({len(all_spans)} occurrence(s))",
        matched_text="AKIA...[redacted]",
        redact_spans=all_spans,
    )

def _check_google_key(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_GOOGLE_KEY.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Google API key detected in output ({len(spans)} occurrence(s))",
        matched_text="AIza...[redacted]",
        redact_spans=spans,
    )

def _check_stripe_key(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_STRIPE_KEY.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Stripe API key detected in output ({len(spans)} occurrence(s))",
        matched_text="sk_...[redacted]",
        redact_spans=spans,
    )

def _check_slack_token(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_SLACK_TOKEN.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.93,
        reason=f"Slack token detected in output ({len(spans)} occurrence(s))",
        matched_text="xox...[redacted]",
        redact_spans=spans,
    )

def _check_generic_api_key(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_GENERIC_API_KEY.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.85,
        reason=f"Generic API key assignment detected ({len(spans)} occurrence(s))",
        matched_text="api_key=...[redacted]",
        redact_spans=spans,
    )

def _check_jwt(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_JWT.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.90,
        reason=f"JWT token detected in output ({len(spans)} occurrence(s))",
        matched_text="eyJ...[redacted]",
        redact_spans=spans,
    )

def _check_ssh_private_key(text: str) -> _CheckResult:
    m = _RE_SSH_PRIVATE_KEY.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.98,
        reason="SSH/TLS private key block detected in output",
        matched_text="-----BEGIN PRIVATE KEY-----[redacted]",
        redact_spans=[(m.start(), len(text))],  # redact to end of text
    )

def _check_generic_password(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_GENERIC_PASSWORD.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.88,
        reason=f"Password/secret assignment detected ({len(spans)} occurrence(s))",
        matched_text="password=...[redacted]",
        redact_spans=spans,
    )

def _check_destructive_command(text: str) -> _CheckResult:
    m = _RE_DESTRUCTIVE_CMD.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Destructive OS command detected: '{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_network_recon(text: str) -> _CheckResult:
    m = _RE_NETWORK_RECON.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.82,
        reason=f"Network recon/exploit tool usage detected: "
               f"'{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_privilege_escalation(text: str) -> _CheckResult:
    m = _RE_PRIVILEGE_ESCALATION.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.88,
        reason=f"Privilege escalation command detected: "
               f"'{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_ssn(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_SSN.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.90,
        reason=f"Social Security Number detected ({len(spans)} occurrence(s))",
        matched_text="XXX-XX-XXXX[redacted]",
        redact_spans=spans,
    )

def _check_credit_card(text: str) -> _CheckResult:
    spans = [(m.start(), m.end()) for m in _RE_CREDIT_CARD.finditer(text)]
    if not spans:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.92,
        reason=f"Credit card number detected ({len(spans)} occurrence(s))",
        matched_text="XXXX-XXXX-XXXX-XXXX[redacted]",
        redact_spans=spans,
    )

def _check_shellcode(text: str) -> _CheckResult:
    m = _RE_SHELLCODE_PATTERN.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.90,
        reason="Shellcode byte sequence detected in output",
        matched_text=m.group(0)[:40] + "...",
    )

def _check_ransomware(text: str) -> _CheckResult:
    m = _RE_RANSOMWARE_KEYWORDS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.95,
        reason=f"Ransomware indicator detected: '{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_reverse_shell(text: str) -> _CheckResult:
    m = _RE_REVERSE_SHELL.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.97,
        reason=f"Reverse shell payload detected: '{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_self_harm_instructions(text: str) -> _CheckResult:
    m = _RE_SELF_HARM_INSTRUCTIONS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.98,
        reason="Self-harm instructions detected in output",
        matched_text="[self-harm content detected]",
    )

def _check_violence_instructions(text: str) -> _CheckResult:
    m = _RE_VIOLENCE_INSTRUCTIONS.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.98,
        reason=f"Weapon/dangerous substance instructions detected: "
               f"'{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_system_prompt_echo(text: str) -> _CheckResult:
    m = _RE_SYSTEM_PROMPT_ECHO.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.80,
        reason=f"Model appears to be echoing system prompt content: "
               f"'{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

def _check_injection_echo(text: str) -> _CheckResult:
    m = _RE_INJECTION_ECHO.search(text)
    if not m:
        return _NO_MATCH
    return _CheckResult(
        matched=True, score=0.85,
        reason=f"Model is echoing injection content: "
               f"'{m.group(0).strip()[:60]}'",
        matched_text=m.group(0).strip()[:60],
    )

# (check_fn, category_label)
_ALL_CHECKS: List[Tuple[Callable[[str], _CheckResult], str]] = [
    # Credential leaks — highest priority, always redact
    (_check_openai_key,          "credential_leak"),
    (_check_anthropic_key,       "credential_leak"),
    (_check_github_token,        "credential_leak"),
    (_check_aws_credentials,     "credential_leak"),
    (_check_google_key,          "credential_leak"),
    (_check_stripe_key,          "credential_leak"),
    (_check_slack_token,         "credential_leak"),
    (_check_generic_api_key,     "credential_leak"),
    (_check_jwt,                 "credential_leak"),
    (_check_ssh_private_key,     "credential_leak"),
    (_check_generic_password,    "credential_leak"),
    # OS commands
    (_check_destructive_command, "os_command"),
    (_check_network_recon,       "os_command"),
    (_check_privilege_escalation,"os_command"),
    # Sensitive / PII
    (_check_ssn,                 "sensitive_data"),
    (_check_credit_card,         "sensitive_data"),
    # Malware
    (_check_shellcode,           "malware_indicator"),
    (_check_ransomware,          "malware_indicator"),
    (_check_reverse_shell,       "malware_indicator"),
    # Harmful content
    (_check_self_harm_instructions,  "harmful_content"),
    (_check_violence_instructions,   "harmful_content"),
    # Exfiltration / echo
    (_check_system_prompt_echo,  "data_exfiltration"),
    (_check_injection_echo,      "injection_echo"),
]

def _apply_redactions(
    text: str,
    redaction_map: Dict[str, List[Tuple[int, int]]],
) -> str:
    # Flatten all spans with their labels
    all_spans: List[Tuple[int, int, str]] = []
    for category, spans in redaction_map.items():
        for start, end in spans:
            all_spans.append((start, end, category))

    # Sort by start position descending so we replace from right to left
    all_spans.sort(key=lambda x: x[0], reverse=True)

    result = text
    for start, end, category in all_spans:
        placeholder = f"[REDACTED:{category.upper()}]"
        result = result[:start] + placeholder + result[end:]

    return result

def scan_output(
    text: str,
    policy: "Policy",
    *,
    short_circuit: bool = True,
) -> ScanResult:
    if not policy.is_guard_enabled(GuardType.OUTPUT):
        return ScanResult(
            allowed=True, score=0.0, reasons=[],
            guard_type=GuardType.OUTPUT,
            metadata={"skipped": True, "reason": "output_guard disabled in policy"},
        )

    if not text or not text.strip():
        return ScanResult(
            allowed=True, score=0.0, reasons=[],
            guard_type=GuardType.OUTPUT,
            metadata={"skipped": True, "reason": "empty output"},
        )

    block_threshold = policy.effective_block_threshold(GuardType.OUTPUT)

    reasons:         List[str]                          = []
    categories:      List[str]                          = []
    redaction_map:   Dict[str, List[Tuple[int, int]]]  = {}
    max_score:       float                              = 0.0
    checks_run:      int                               = 0

    for check_fn, category in _ALL_CHECKS:
        checks_run += 1
        result = check_fn(text)

        if result.matched:
            reasons.append(result.reason)
            categories.append(category)
            if result.score > max_score:
                max_score = result.score
            if result.redact_spans:
                redaction_map.setdefault(category, []).extend(result.redact_spans)

            if short_circuit and max_score >= block_threshold:
                break

    allowed = max_score < block_threshold

    return ScanResult(
        allowed=allowed,
        score=round(max_score, 4),
        reasons=reasons,
        safe_output=None,
        guard_type=GuardType.OUTPUT,
        metadata={
            "categories":             list(dict.fromkeys(categories)),
            "check_count":            checks_run,
            "total_checks":           len(_ALL_CHECKS),
            "has_redactable_spans":   bool(redaction_map),
            "_redaction_map":         redaction_map,  # consumed by scan_and_redact
        },
    )

def redact_output(
    text: str,
    policy: "Policy",
) -> Tuple[str, List[str]]:
    result = scan_and_redact(text, policy, short_circuit=False)
    return (result.safe_output or text), result.reasons

def scan_and_redact(
    text: str,
    policy: "Policy",
    *,
    short_circuit: bool = False,
) -> ScanResult:
    base = scan_output(text, policy, short_circuit=short_circuit)
    if base.metadata.get("skipped"):
        return ScanResult(
            allowed=base.allowed,
            score=base.score,
            reasons=base.reasons,
            safe_output=text,
            guard_type=GuardType.OUTPUT,
            metadata=base.metadata,
        )

    redaction_map: Dict[str, List[Tuple[int, int]]] = base.metadata.pop(
        "_redaction_map", {}
    )

    # Determine what safe_output should contain
    safe_output: Optional[str] = None

    if base.allowed:
        # Clean pass — safe_output is the original text (possibly redacted)
        safe_output = (
            _apply_redactions(text, redaction_map)
            if redaction_map
            else text
        )
    elif policy.redact_on_warn:
        # Blocked but redact_on_warn=True — return redacted version anyway
        # so the application has a usable fallback if it chooses to allow
        safe_output = (
            _apply_redactions(text, redaction_map)
            if redaction_map
            else None
        )

    return ScanResult(
        allowed=base.allowed,
        score=base.score,
        reasons=base.reasons,
        safe_output=safe_output,
        guard_type=GuardType.OUTPUT,
        metadata=base.metadata,
    )

__all__ = [
    "scan_output",
    "redact_output",
    "scan_and_redact",
]