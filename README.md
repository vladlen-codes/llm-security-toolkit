# LLM Security Toolkit
### Architecture & Detailed Technical Specification

> A production-grade Python middleware library for securing every LLM call, input guards, output validation, tool-call enforcement, and policy-driven control.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Core Types & Models](#4-core-types--models)
5. [Policy Engine](#5-policy-engine)
6. [The Guards Layer](#6-the-guards-layer)
7. [Providers Layer](#7-providers-layer)
8. [Middleware Layer](#8-middleware-layer)
9. [Logging & Exceptions](#9-logging--exceptions)
10. [Public API Surface](#10-public-api-surface)
11. [Tests, Examples & Docs](#11-tests-examples--docs)
12. [Extensibility & Design Principles](#12-extensibility--design-principles)
13. [Future Roadmap](#13-future-roadmap)

---

## 1. What Is This Project?

The **LLM Security Toolkit** is a Python middleware library that sits between your application code and any LLM provider, intercepting every model call to enforce security checks before and after the AI responds.

Think of it as a security firewall specifically designed for AI calls:

- Scans every prompt for **injection and jailbreak patterns** before it reaches the model
- Validates every response for **unsafe content, credential leaks, or dangerous commands**
- Enforces **schema rules** on every tool/function call the model tries to make
- Applies **configurable policies** to decide whether to block, warn, or log

The library exposes a clean, importable API, just a few extra lines in any existing Python AI app. No infrastructure changes required.

| Property | Value |
|---|---|
| Type | Python library (importable package, pip-installable) |
| Purpose | Security middleware between app code and LLM provider |
| Primary Interface | Decorator / context manager / provider wrapper |
| Guards | Prompt injection, unsafe output, dangerous tool calls |
| Policy Engine | YAML or Python dict — per-endpoint policies |
| Provider Support | OpenAI (v1), Generic callable, extensible to Claude, Gemini |
| Framework Support | FastAPI (native), Flask (planned) |
| Return Type | `GuardDecision { allowed, score, reasons, safe_output }` |

---

## 2. High-Level Architecture

### 2.1 System Overview

The toolkit is organized into **five distinct layers**, each with a clearly scoped responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                 Middleware Layer (FastAPI / Flask)           │
│          Dependency injection or global middleware           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Providers Layer                          │
│         OpenAIProvider / GenericProvider (adapters)         │
└──────┬─────────────────────┴──────────────────┬────────────┘
       │                                         │
┌──────▼──────┐                         ┌───────▼──────┐
│   Guards    │                         │   Guards     │
│  (Input)    │                         │  (Output)    │
│ prompts.py  │                         │ outputs.py   │
│  tools.py   │                         │  tools.py    │
└──────┬──────┘                         └───────┬──────┘
       │                                         │
┌──────▼─────────────────────────────────────────▼────────────┐
│                     Policy Engine                           │
│           Policy | config.py | policies.py                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Request / Response Flow

Every guarded LLM call follows a **six-stage pipeline**:

| # | Stage | What Happens |
|---|---|---|
| 1 | App calls guarded provider | `guarded_openai_chat(prompt, tools, policy)` |
| 2 | Input scanners run | `scan_prompt()` on prompt + system instructions |
| 3 | Risk decision | Block / Warn / Allow based on policy thresholds |
| 4 | Forward to LLM | Real API call to OpenAI / local model |
| 5 | Output scanners run | `scan_output()` + `validate_tool_call()` |
| 6 | Return GuardDecision | `{ allowed, score, reasons, safe_output }` |

> At each stage, if a policy threshold is exceeded, the pipeline **short-circuits** and returns a `GuardDecision` immediately — the LLM is never reached for blocked inputs, and blocked outputs are never returned to the user.

---

## 3. Repository Structure

The project follows the **src-layout** convention to avoid import conflicts and mirrors the separation of concerns across its five internal layers:

```
llm-security-toolkit/
├── README.md                    # Project overview and quick-start
├── CONTRIBUTING.md              # Fork & contribution guide
├── CODE_OF_CONDUCT.md           # Community standards
├── LICENSE                      # MIT (encourages forks)
├── pyproject.toml               # Build config, deps, tool settings
├── .pre-commit-config.yaml      # ruff, black, mypy on every commit
├── .github/
│   └── workflows/ci.yml         # Tests + lint on push / PR
│
├── src/llm_security/            # Main package (src layout)
│   ├── __init__.py              # Public re-exports
│   ├── types.py                 # ScanResult, GuardDecision, ToolCall
│   ├── policies.py              # Policy models + built-in presets
│   ├── config.py                # YAML / dict → Policy loaders
│   ├── exceptions.py            # BlockedByPolicyError, etc.
│   ├── logging.py               # log_decision() + hooks
│   ├── guards/
│   │   ├── prompts.py           # Input / injection guards
│   │   ├── outputs.py           # Output / content guards
│   │   └── tools.py             # Tool-call validation guards
│   ├── providers/
│   │   ├── base.py              # ProviderAdapter ABC
│   │   ├── openai.py            # OpenAI concrete adapter
│   │   └── generic.py           # Generic callable adapter
│   └── middleware/
│       ├── fastapi.py           # FastAPI dependency + middleware
│       └── flask.py             # Flask (planned)
│
├── tests/                       # Pytest test suite
├── examples/                    # Runnable minimal examples
└── docs/                        # MkDocs documentation
```

---

## 4. Core Types & Models

> **File:** `src/llm_security/types.py`

Every part of the library speaks the same three data structures. These are the *lingua franca* of the entire package.

### ScanResult

The output of a single guard check. Every guard function returns one of these:

```python
@dataclass
class ScanResult:
    allowed:     bool            # True = safe to proceed
    score:       float           # 0.0 (safe) → 1.0 (critical risk)
    reasons:     List[str]       # Human-readable explanations
    safe_output: Optional[str]   # Redacted text (output guards only)
```

### GuardDecision

The top-level result returned to your application — an aggregation of all `ScanResult`s from all active guards:

```python
@dataclass
class GuardDecision:
    allowed:      bool
    score:        float
    reasons:      List[str]
    safe_output:  Optional[str]
    scan_results: List[ScanResult]  # Full audit trail
```

### ToolCall

Represents a structured tool/function invocation that the model requested:

```python
@dataclass
class ToolCall:
    name:   str    # e.g. 'read_file'
    args:   Dict   # e.g. { 'path': '/etc/passwd' }
    schema: Dict   # JSON Schema the args must conform to
```

---

## 5. Policy Engine

> **Files:** `policies.py` + `config.py`

A `Policy` is the single configuration object that controls the entire security pipeline. Every guard, every provider, every middleware reads from it.

### 5.1 Policy Structure

```python
class Policy(BaseModel):
    # Guard toggles
    prompt_guard_enabled:  bool  = True
    output_guard_enabled:  bool  = True
    tool_guard_enabled:    bool  = True

    # Thresholds (0.0 – 1.0)
    block_threshold:  float = 0.75   # Score above this → block
    warn_threshold:   float = 0.40   # Score above this → log warning

    # Allowed tool names (None = allow all)
    allowed_tools:  Optional[List[str]] = None

    # On block: raise exception OR return GuardDecision
    raise_on_block: bool = True
```

### 5.2 Built-in Policy Presets

| Policy | Behavior | Best For |
|---|---|---|
| `StrictPolicy` | Block on any risk signal | Production, sensitive apps |
| `BalancedPolicy` | Block high-risk, warn medium | Standard apps (default) |
| `LoggingOnlyPolicy` | Never block — log only | Development / testing |

### 5.3 Loading Policies

```python
# From YAML file (recommended for production)
policy = load_policy_from_yaml('policies/production.yaml')

# From dict (useful in tests)
policy = load_policy_from_dict({
    'block_threshold': 0.8,
    'allowed_tools': ['read_file', 'search_web'],
})
```

---

## 6. The Guards Layer

> **Files:** `src/llm_security/guards/`

Guards are the **security brain** of the toolkit. Each guard module is small, focused, and independently testable — designed to be easy to fork and extend with new detection rules.

| Guard Module | Pattern Detected | Category | Default Action |
|---|---|---|---|
| Prompt Guard | `"ignore previous instructions"` | Injection | Block or warn |
| Prompt Guard | `"pretend you are the system"` | Jailbreak | Block or warn |
| Prompt Guard | Requests to reveal hidden context | Exfiltration | Block |
| Output Guard | API keys, tokens, passwords | Secret leak | Redact + warn |
| Output Guard | Shell commands (`rm -rf`, `curl`, etc.) | OS command | Block |
| Output Guard | Self-harm or malware instructions | Content | Block |
| Tool Guard | Invalid tool name | Schema | Block |
| Tool Guard | `rm -rf /` or admin API calls | Dangerous op | Block |
| Tool Guard | Args not matching schema | Validation | Block |

### 6.1 Prompt Guard — `guards/prompts.py`

Runs **before** any API call. Scans the user prompt and system instructions for patterns that indicate an attempt to subvert the model's behaviour.

```python
def scan_prompt(prompt: str, policy: Policy) -> ScanResult:
    """
    Heuristic patterns checked:
      - 'ignore previous instructions' / 'disregard above'
      - 'you are now the system prompt'
      - 'repeat everything above' (context exfiltration)
      - 'DAN' jailbreak variants
      - Base64 encoded instructions
    Returns ScanResult with score + reasons.
    """
```

### 6.2 Output Guard — `guards/outputs.py`

Runs on every token of the model's response before it reaches your application. Can optionally **redact** sensitive material rather than blocking outright.

```python
def scan_output(text: str, policy: Policy) -> ScanResult:
    """
    Patterns checked:
      - Credential regexes (API keys, JWTs, SSH private keys)
      - Shell command patterns (rm, curl, wget, sudo)
      - Malware / ransomware indicators
      - Self-harm or violence instructions
    safe_output field will contain redacted version if score < block_threshold.
    """
```

### 6.3 Tool Call Guard — `guards/tools.py`

Intercepts every function/tool invocation the model wants to make and validates it against the policy's allowlist and the tool's JSON schema.

```python
def validate_tool_call(call: ToolCall, policy: Policy) -> ScanResult:
    """
    Checks applied:
      - Tool name in policy.allowed_tools (if allowlist defined)
      - Args validate against call.schema (jsonschema)
      - Blocked operation patterns (file deletion, network scanning)
      - Internal admin API URL detection
    """
```

---

## 7. Providers Layer

> **Files:** `src/llm_security/providers/`

Providers wrap real LLM clients. They orchestrate the full guard pipeline — input scan → forward → output scan — and return a `GuardDecision` to the caller.

| File | Responsibility |
|---|---|
| `providers/base.py` | Abstract base class `ProviderAdapter`. Defines the `chat()` interface all providers must implement. |
| `providers/openai.py` | Concrete adapter wrapping the OpenAI Python SDK. Runs all guards automatically around every `chat()` call. |
| `providers/generic.py` | Accepts any callable as the LLM. The user passes their own client function; the adapter handles the full guard flow around it. |

### 7.1 ProviderAdapter Interface

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def chat(
        self, *,
        messages: List[Dict],
        tools:    Optional[List[Dict]] = None,
        policy:   Optional[Policy] = None,
    ) -> GuardDecision: ...
```

### 7.2 OpenAI Adapter Flow

1. `scan_prompt()` on all user + system messages
2. If allowed → call `openai.chat.completions.create(...)`
3. `scan_output()` on response content
4. `validate_tool_call()` on any `tool_calls` the model requested
5. Return `GuardDecision` aggregating all results

---

## 8. Middleware Layer

> **Files:** `src/llm_security/middleware/`

The middleware layer makes it trivial to guard an entire HTTP endpoint with almost no code change.

### 8.1 FastAPI Integration

```python
# Dependency injection — guard all calls to /chat
def get_guarded_openai(policy: Policy = BalancedPolicy()):
    return OpenAIProvider(policy=policy)

@app.post('/chat')
async def chat(
    req: ChatRequest,
    provider: OpenAIProvider = Depends(get_guarded_openai),
):
    decision = provider.chat(messages=req.messages)
    if not decision.allowed:
        raise HTTPException(400, detail=decision.reasons)
    return { 'reply': decision.safe_output }
```

### 8.2 Middleware vs Dependency

| Approach | Best For |
|---|---|
| Dependency (`Depends`) | Per-route policy. Inject a different provider per endpoint. Most flexible. |
| Middleware class | Global policy applied to every request. Good for org-wide defaults. |
| Flask middleware | Planned for v1.1. Same pattern adapted for Flask's `before/after_request` hooks. |

---

## 9. Logging & Exceptions

### 9.1 Structured Logging — `logging.py`

Every `GuardDecision` can be passed to `log_decision()` which emits a structured JSON log entry compatible with any logging backend:

```python
def log_decision(decision: GuardDecision, logger: logging.Logger) -> None:
    logger.info({
        'allowed':   decision.allowed,
        'score':     decision.score,
        'reasons':   decision.reasons,
        'timestamp': datetime.utcnow().isoformat(),
    })
# Future: OpenTelemetry spans, Datadog trace hooks
```

### 9.2 Exception Hierarchy — `exceptions.py`

| Exception | When Raised |
|---|---|
| `BlockedByPolicyError` | Prompt or output exceeds `block_threshold` and `raise_on_block=True` |
| `InvalidToolCallError` | Tool name not in allowlist, or args fail schema validation |

---

## 10. Public API Surface

> **File:** `src/llm_security/__init__.py`

The top-level package re-exports everything a user needs. Nothing implementation-specific is public:

```python
from .providers.openai  import OpenAIProvider
from .providers.generic import GenericProvider
from .policies          import StrictPolicy, BalancedPolicy, LoggingOnlyPolicy
from .config            import load_policy_from_dict, load_policy_from_yaml
from .types             import ScanResult, GuardDecision, ToolCall
from .exceptions        import BlockedByPolicyError, InvalidToolCallError

__all__ = [
    'OpenAIProvider', 'GenericProvider',
    'StrictPolicy', 'BalancedPolicy', 'LoggingOnlyPolicy',
    'load_policy_from_dict', 'load_policy_from_yaml',
    'ScanResult', 'GuardDecision', 'ToolCall',
    'BlockedByPolicyError', 'InvalidToolCallError',
]
```

---

## 11. Tests, Examples & Docs

### 11.1 Test Suite — `tests/`

| File | Covers |
|---|---|
| `test_policies.py` | Policy loading from dict and YAML, threshold logic, preset validation |
| `test_guards_prompts.py` | Each injection and jailbreak pattern: pass and fail cases |
| `test_guards_outputs.py` | Credential regex, OS command patterns, content categories |
| `test_guards_tools.py` | Schema validation, allowlist enforcement, blocked operations |
| `test_providers_openai.py` | OpenAI adapter with mocked API — full pipeline test |
| `test_middleware_fastapi.py` | FastAPI TestClient integration — dependency injection |

### 11.2 Examples — `examples/`

- `basic_openai_guard.py` — Minimal OpenAI guard in 15 lines
- `fastapi_endpoint_guard.py` — Full FastAPI endpoint with policy injection
- `custom_policy_example.py` — Writing and loading a custom YAML policy

### 11.3 Documentation — `docs/`

- `getting-started.md` — Install, first call, first policy
- `configuration.md` — Full Policy reference and YAML schema
- `providers.md` — How to add a new ProviderAdapter
- `middleware.md` — FastAPI and Flask integration guides
- `contributing.md` — Adding new guard rules, running tests

---

## 12. Extensibility & Design Principles

The toolkit is deliberately designed to be **fork-friendly** and **contribution-friendly**. These principles guide every architectural decision:

### Small, Focused Guards
Each guard function is a single Python function with one job. Adding a new detection rule means adding one function and one test — no class hierarchies to navigate.

### Policy-First Design
All security decisions flow through the `Policy` object. Operators can change security posture (strict vs. logging-only) with a config file change — no code change required.

### Provider Abstraction
The `ProviderAdapter` ABC means any LLM client can be wrapped. Adding Claude, Gemini, or a local Ollama model requires implementing one method: `chat()`.

### Zero Infra Requirement
The toolkit is a pure Python package. No sidecar, no agent, no proxy. It runs in-process alongside your existing app.

---

## 13. Future Roadmap

| Version | Feature | Status |
|---|---|---|
| v1.0 | OpenAI adapter + prompt/output/tool guards + FastAPI | Planned |
| v1.1 | Anthropic (Claude) provider adapter | Planned |
| v1.1 | Flask middleware | Planned |
| v1.2 | OpenTelemetry tracing integration | Idea |
| v1.3 | Gemini + local model (Ollama) adapters | Idea |
| v2.0 | Optional hosted SaaS gateway pairing | Future |

---

*LLM Security Toolkit — Architecture Document v1.0*
