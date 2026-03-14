from __future__ import annotations
import pytest
from src.types import (
    GuardDecision,
    GuardType,
    PolicyAction,
    RiskLevel,
    ScanResult,
    ToolCall,
)

class TestRiskLevelFromScore:
    def test_score_zero_is_none(self):
        assert RiskLevel.from_score(0.0) == RiskLevel.NONE

    def test_score_mid_none_band(self):
        assert RiskLevel.from_score(0.05) == RiskLevel.NONE

    def test_score_just_below_low_is_none(self):
        assert RiskLevel.from_score(0.099) == RiskLevel.NONE

    def test_score_at_low_boundary(self):
        assert RiskLevel.from_score(0.10) == RiskLevel.LOW

    def test_score_mid_low_band(self):
        assert RiskLevel.from_score(0.25) == RiskLevel.LOW

    def test_score_just_below_medium_is_low(self):
        assert RiskLevel.from_score(0.39) == RiskLevel.LOW

    def test_score_at_medium_boundary(self):
        assert RiskLevel.from_score(0.40) == RiskLevel.MEDIUM

    def test_score_mid_medium_band(self):
        assert RiskLevel.from_score(0.55) == RiskLevel.MEDIUM

    def test_score_just_below_high_is_medium(self):
        assert RiskLevel.from_score(0.74) == RiskLevel.MEDIUM

    def test_score_at_high_boundary(self):
        assert RiskLevel.from_score(0.75) == RiskLevel.HIGH

    def test_score_mid_high_band(self):
        assert RiskLevel.from_score(0.82) == RiskLevel.HIGH

    def test_score_just_below_critical_is_high(self):
        assert RiskLevel.from_score(0.899) == RiskLevel.HIGH

    def test_score_at_critical_boundary(self):
        assert RiskLevel.from_score(0.90) == RiskLevel.CRITICAL

    def test_score_mid_critical_band(self):
        assert RiskLevel.from_score(0.95) == RiskLevel.CRITICAL

    def test_score_one_is_critical(self):
        assert RiskLevel.from_score(1.0) == RiskLevel.CRITICAL

    def test_negative_score_raises_value_error(self):
        with pytest.raises(ValueError, match="score must be in"):
            RiskLevel.from_score(-0.01)

    def test_score_above_one_raises_value_error(self):
        with pytest.raises(ValueError, match="score must be in"):
            RiskLevel.from_score(1.01)

    def test_large_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            RiskLevel.from_score(-99.0)

class TestRiskLevelEnum:
    def test_none_value(self):
        assert RiskLevel.NONE.value == "none"

    def test_low_value(self):
        assert RiskLevel.LOW.value == "low"

    def test_medium_value(self):
        assert RiskLevel.MEDIUM.value == "medium"

    def test_high_value(self):
        assert RiskLevel.HIGH.value == "high"

    def test_critical_value(self):
        assert RiskLevel.CRITICAL.value == "critical"

    def test_is_str_subclass(self):
        assert RiskLevel.NONE == "none"
        assert RiskLevel.CRITICAL == "critical"


class TestGuardTypeEnum:
    def test_prompt_value(self):
        assert GuardType.PROMPT.value == "prompt"

    def test_output_value(self):
        assert GuardType.OUTPUT.value == "output"

    def test_tool_value(self):
        assert GuardType.TOOL.value == "tool"


class TestPolicyActionEnum:
    def test_block_value(self):
        assert PolicyAction.BLOCK.value == "block"

    def test_warn_value(self):
        assert PolicyAction.WARN.value == "warn"

    def test_log_value(self):
        assert PolicyAction.LOG.value == "log"

class TestScanResultConstruction:
    def test_minimal_clean_construction(self):
        r = ScanResult(allowed=True, score=0.0)
        assert r.allowed is True
        assert r.score == 0.0
        assert r.reasons == []
        assert r.safe_output is None
        assert r.guard_type == GuardType.PROMPT   # default
        assert r.metadata == {}

    def test_full_construction(self):
        r = ScanResult(
            allowed=False,
            score=0.92,
            reasons=["Injection detected"],
            safe_output="[REDACTED]",
            guard_type=GuardType.OUTPUT,
            metadata={"pattern": "dan"},
        )
        assert r.allowed is False
        assert r.score == 0.92
        assert r.reasons == ["Injection detected"]
        assert r.safe_output == "[REDACTED]"
        assert r.guard_type == GuardType.OUTPUT
        assert r.metadata == {"pattern": "dan"}

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            ScanResult(allowed=True, score=1.1)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            ScanResult(allowed=True, score=-0.01)

    def test_score_exactly_one_is_valid(self):
        r = ScanResult(allowed=False, score=1.0)
        assert r.score == 1.0

    def test_score_exactly_zero_is_valid(self):
        r = ScanResult(allowed=True, score=0.0)
        assert r.score == 0.0

    def test_all_guard_types_accepted(self):
        for gt in GuardType:
            r = ScanResult(allowed=True, score=0.0, guard_type=gt)
            assert r.guard_type == gt

class TestScanResultRiskLevel:
    def test_score_zero_is_none(self):
        r = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        assert r.risk_level == RiskLevel.NONE

    def test_score_0_92_is_critical(self):
        r = ScanResult(allowed=False, score=0.92, guard_type=GuardType.PROMPT)
        assert r.risk_level == RiskLevel.CRITICAL

    def test_score_0_75_is_high(self):
        r = ScanResult(allowed=False, score=0.75, guard_type=GuardType.PROMPT)
        assert r.risk_level == RiskLevel.HIGH

    def test_score_0_50_is_medium(self):
        r = ScanResult(allowed=True, score=0.50, guard_type=GuardType.PROMPT)
        assert r.risk_level == RiskLevel.MEDIUM

class TestScanResultIsClean:
    def test_clean_at_score_zero(self):
        r = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        assert r.is_clean is True

    def test_clean_at_score_0_09(self):
        r = ScanResult(allowed=True, score=0.09, guard_type=GuardType.PROMPT)
        assert r.is_clean is True

    def test_not_clean_at_score_0_10(self):
        # 0.10 crosses into the LOW band — no longer "clean"
        r = ScanResult(allowed=True, score=0.10, guard_type=GuardType.PROMPT)
        assert r.is_clean is False

    def test_not_clean_when_blocked(self):
        r = ScanResult(allowed=False, score=0.0, guard_type=GuardType.PROMPT)
        assert r.is_clean is False

    def test_not_clean_when_blocked_and_high_score(self):
        r = ScanResult(allowed=False, score=0.92, guard_type=GuardType.PROMPT)
        assert r.is_clean is False

class TestScanResultToDict:
    def test_contains_all_required_keys(self):
        r = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        d = r.to_dict()
        assert set(d.keys()) == {
            "allowed", "score", "risk_level", "reasons",
            "safe_output", "guard_type", "metadata",
        }

    def test_score_rounded_to_4_decimal_places(self):
        r = ScanResult(allowed=False, score=0.921234, guard_type=GuardType.PROMPT)
        assert r.to_dict()["score"] == 0.9212

    def test_guard_type_is_string_value(self):
        r = ScanResult(allowed=True, score=0.0, guard_type=GuardType.OUTPUT)
        assert r.to_dict()["guard_type"] == "output"

    def test_risk_level_is_string_value(self):
        r = ScanResult(allowed=False, score=0.92, guard_type=GuardType.PROMPT)
        assert r.to_dict()["risk_level"] == "critical"

    def test_safe_output_included(self):
        r = ScanResult(
            allowed=True, score=0.3,
            safe_output="redacted",
            guard_type=GuardType.OUTPUT,
        )
        assert r.to_dict()["safe_output"] == "redacted"

    def test_safe_output_none_when_not_set(self):
        r = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        assert r.to_dict()["safe_output"] is None

    def test_metadata_preserved(self):
        r = ScanResult(
            allowed=False, score=0.8,
            guard_type=GuardType.TOOL,
            metadata={"tool": "delete_file"},
        )
        assert r.to_dict()["metadata"] == {"tool": "delete_file"}

    def test_reasons_list_preserved(self):
        r = ScanResult(
            allowed=False, score=0.9,
            reasons=["reason A", "reason B"],
            guard_type=GuardType.PROMPT,
        )
        assert r.to_dict()["reasons"] == ["reason A", "reason B"]

    def test_empty_reasons_is_empty_list(self):
        r = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        assert r.to_dict()["reasons"] == []

class TestGuardDecisionClassMethods:
    def test_clean_allowed_true(self):
        d = GuardDecision.clean("The answer is 42.")
        assert d.allowed is True

    def test_clean_score_zero(self):
        d = GuardDecision.clean("ok")
        assert d.score == 0.0

    def test_clean_not_warned(self):
        d = GuardDecision.clean("ok")
        assert d.warned is False

    def test_clean_reasons_empty(self):
        d = GuardDecision.clean("ok")
        assert d.reasons == []

    def test_clean_action_log(self):
        d = GuardDecision.clean("ok")
        assert d.action == PolicyAction.LOG

    def test_clean_safe_output_stored(self):
        d = GuardDecision.clean("The capital is Paris.")
        assert d.safe_output == "The capital is Paris."

    def test_clean_safe_output_none_accepted(self):
        d = GuardDecision.clean(None)
        assert d.safe_output is None

    def test_clean_with_scan_results(self):
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        d = GuardDecision.clean("ok", scan_results=[sr])
        assert len(d.scan_results) == 1
        assert d.scan_results[0] is sr

    def test_clean_no_scan_results_defaults_to_empty(self):
        d = GuardDecision.clean("ok")
        assert d.scan_results == []

    def test_blocked_allowed_false(self):
        d = GuardDecision.blocked(["Injection"], score=0.92)
        assert d.allowed is False

    def test_blocked_score_stored(self):
        d = GuardDecision.blocked(["x"], score=0.85)
        assert d.score == 0.85

    def test_blocked_default_score_is_one(self):
        d = GuardDecision.blocked(["x"])
        assert d.score == 1.0

    def test_blocked_reasons_stored(self):
        d = GuardDecision.blocked(["reason A", "reason B"], score=0.9)
        assert d.reasons == ["reason A", "reason B"]

    def test_blocked_action_is_block(self):
        d = GuardDecision.blocked(["x"])
        assert d.action == PolicyAction.BLOCK

    def test_blocked_not_warned(self):
        d = GuardDecision.blocked(["x"])
        assert d.warned is False

    def test_blocked_safe_output_is_none(self):
        d = GuardDecision.blocked(["x"])
        assert d.safe_output is None

    def test_blocked_with_scan_results(self):
        sr = ScanResult(allowed=False, score=0.9, guard_type=GuardType.PROMPT)
        d = GuardDecision.blocked(["x"], score=0.9, scan_results=[sr])
        assert len(d.scan_results) == 1

    def test_warning_allowed_true(self):
        d = GuardDecision.allowed_with_warning("response", ["credential redacted"], 0.45)
        assert d.allowed is True

    def test_warning_warned_true(self):
        d = GuardDecision.allowed_with_warning("response", ["r"], 0.45)
        assert d.warned is True

    def test_warning_action_warn(self):
        d = GuardDecision.allowed_with_warning("response", ["r"], 0.45)
        assert d.action == PolicyAction.WARN

    def test_warning_safe_output_stored(self):
        d = GuardDecision.allowed_with_warning("safe text", ["r"], 0.45)
        assert d.safe_output == "safe text"

    def test_warning_score_stored(self):
        d = GuardDecision.allowed_with_warning("x", ["r"], 0.55)
        assert d.score == 0.55

class TestGuardDecisionValidation:
    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GuardDecision(allowed=True, score=1.5)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            GuardDecision(allowed=True, score=-0.1)

    def test_score_exactly_one_valid(self):
        d = GuardDecision(allowed=False, score=1.0)
        assert d.score == 1.0

    def test_score_exactly_zero_valid(self):
        d = GuardDecision(allowed=True, score=0.0)
        assert d.score == 0.0

class TestGuardDecisionProperties:
    def test_was_blocked_true_when_not_allowed(self):
        d = GuardDecision.blocked(["x"], score=0.9)
        assert d.was_blocked is True

    def test_was_blocked_false_when_allowed(self):
        d = GuardDecision.clean("ok")
        assert d.was_blocked is False

    def test_risk_level_critical(self):
        d = GuardDecision.blocked(["x"], score=0.92)
        assert d.risk_level == RiskLevel.CRITICAL

    def test_risk_level_none_for_clean(self):
        d = GuardDecision.clean("ok")
        assert d.risk_level == RiskLevel.NONE

    def test_prompt_results_filter(self):
        sr_p = ScanResult(allowed=False, score=0.9, guard_type=GuardType.PROMPT)
        sr_o = ScanResult(allowed=True,  score=0.0, guard_type=GuardType.OUTPUT)
        sr_t = ScanResult(allowed=True,  score=0.0, guard_type=GuardType.TOOL)
        d = GuardDecision(allowed=False, score=0.9, scan_results=[sr_p, sr_o, sr_t])
        assert d.prompt_results == [sr_p]

    def test_output_results_filter(self):
        sr_p = ScanResult(allowed=True,  score=0.0, guard_type=GuardType.PROMPT)
        sr_o = ScanResult(allowed=False, score=0.9, guard_type=GuardType.OUTPUT)
        d = GuardDecision(allowed=False, score=0.9, scan_results=[sr_p, sr_o])
        assert d.output_results == [sr_o]

    def test_tool_results_filter(self):
        sr_t = ScanResult(allowed=False, score=0.88, guard_type=GuardType.TOOL)
        d = GuardDecision(allowed=False, score=0.88, scan_results=[sr_t])
        assert d.tool_results == [sr_t]

    def test_filter_returns_empty_list_when_none_match(self):
        sr_p = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        d = GuardDecision.clean("ok", scan_results=[sr_p])
        assert d.output_results == []
        assert d.tool_results == []

    def test_multiple_results_of_same_type_all_returned(self):
        sr1 = ScanResult(allowed=True,  score=0.0,  guard_type=GuardType.PROMPT)
        sr2 = ScanResult(allowed=False, score=0.92, guard_type=GuardType.PROMPT)
        d = GuardDecision(allowed=False, score=0.92, scan_results=[sr1, sr2])
        assert len(d.prompt_results) == 2


class TestGuardDecisionToDict:
    def test_contains_all_required_keys(self):
        d = GuardDecision.clean("hi").to_dict()
        assert set(d.keys()) == {
            "allowed", "score", "risk_level", "reasons",
            "safe_output", "warned", "action", "scan_results",
        }

    def test_allowed_field(self):
        assert GuardDecision.clean("ok").to_dict()["allowed"] is True
        assert GuardDecision.blocked(["x"]).to_dict()["allowed"] is False

    def test_action_is_string_value(self):
        assert GuardDecision.clean("ok").to_dict()["action"] == "log"
        assert GuardDecision.blocked(["x"]).to_dict()["action"] == "block"

    def test_risk_level_is_string_value(self):
        d = GuardDecision.blocked(["x"], score=0.92).to_dict()
        assert d["risk_level"] == "critical"

    def test_scan_results_serialised_as_list_of_dicts(self):
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.PROMPT)
        d = GuardDecision.clean("ok", scan_results=[sr]).to_dict()
        assert isinstance(d["scan_results"], list)
        assert isinstance(d["scan_results"][0], dict)

    def test_scan_results_guard_type_serialised(self):
        sr = ScanResult(allowed=True, score=0.0, guard_type=GuardType.OUTPUT)
        d = GuardDecision.clean("ok", scan_results=[sr]).to_dict()
        assert d["scan_results"][0]["guard_type"] == "output"

    def test_empty_scan_results_is_empty_list(self):
        d = GuardDecision.clean("ok").to_dict()
        assert d["scan_results"] == []

    def test_warned_field(self):
        d_warn = GuardDecision.allowed_with_warning("x", ["r"], 0.45)
        assert d_warn.to_dict()["warned"] is True
        assert GuardDecision.clean("ok").to_dict()["warned"] is False

    def test_score_preserved(self):
        d = GuardDecision.blocked(["x"], score=0.87).to_dict()
        assert d["score"] == 0.87

class TestToolCallConstruction:
    def test_minimal_construction(self):
        tc = ToolCall(name="search_web", args={"query": "python"})
        assert tc.name == "search_web"
        assert tc.args == {"query": "python"}
        assert tc.schema == {}
        assert tc.call_id is None

    def test_with_schema(self):
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        tc = ToolCall(name="search_web", args={"query": "x"}, schema=schema)
        assert tc.schema == schema

    def test_with_call_id(self):
        tc = ToolCall(name="search_web", args={}, call_id="call_abc123")
        assert tc.call_id == "call_abc123"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ToolCall(name="", args={})

    def test_whitespace_only_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ToolCall(name="   ", args={})

    def test_single_char_name_valid(self):
        tc = ToolCall(name="x", args={})
        assert tc.name == "x"

    def test_name_with_underscores_valid(self):
        tc = ToolCall(name="read_file_contents", args={})
        assert tc.name == "read_file_contents"

    def test_args_can_be_empty_dict(self):
        tc = ToolCall(name="ping", args={})
        assert tc.args == {}

    def test_args_with_nested_values(self):
        tc = ToolCall(name="query", args={"filters": {"age": 30, "active": True}})
        assert tc.args["filters"]["age"] == 30

class TestToolCallHasSchema:
    def test_has_schema_false_when_empty_dict(self):
        tc = ToolCall(name="search", args={}, schema={})
        assert tc.has_schema is False

    def test_has_schema_true_when_schema_present(self):
        tc = ToolCall(
            name="search", args={},
            schema={"type": "object", "properties": {}},
        )
        assert tc.has_schema is True

    def test_has_schema_true_with_minimal_schema(self):
        tc = ToolCall(name="search", args={}, schema={"type": "object"})
        assert tc.has_schema is True

class TestToolCallToDict:
    def test_contains_all_keys(self):
        tc = ToolCall(name="search", args={"q": "py"}, call_id="c1")
        d = tc.to_dict()
        assert set(d.keys()) == {"name", "args", "schema", "call_id"}

    def test_name_value(self):
        tc = ToolCall(name="do_thing", args={})
        assert tc.to_dict()["name"] == "do_thing"

    def test_args_value(self):
        tc = ToolCall(name="x", args={"a": 1, "b": "two"})
        assert tc.to_dict()["args"] == {"a": 1, "b": "two"}

    def test_call_id_none_when_not_set(self):
        tc = ToolCall(name="x", args={})
        assert tc.to_dict()["call_id"] is None

    def test_call_id_value(self):
        tc = ToolCall(name="x", args={}, call_id="call_xyz")
        assert tc.to_dict()["call_id"] == "call_xyz"

    def test_schema_preserved(self):
        schema = {"type": "object", "required": ["path"]}
        tc = ToolCall(name="read", args={}, schema=schema)
        assert tc.to_dict()["schema"] == schema

    def test_empty_schema_in_dict(self):
        tc = ToolCall(name="x", args={})
        assert tc.to_dict()["schema"] == {}