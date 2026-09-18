import json

from patchestry_llm.decomp import parse_printed_unit
from patchestry_llm.providers.fake import FakeProvider
from patchestry_llm.tier2 import (
    Tier2Options,
    clean_definition,
    parse_errors,
    render_unit,
    run_tier2,
)

from conftest import FUNCTION_KEY

FLAT_UNIT = """\
// patchestry:tu format=1 target=ARM:LE:32:Cortex arch=ARM
typedef unsigned char undefined1;
extern int g_state;
int FUN_20000000(int param_1);
// patchestry:function-begin ram:20000000 name=FUN_20000000 symbol=FUN_20000000
int FUN_20000000(int param_1) {
    int iVar1;
    iVar1 = param_1 + g_state;
    return iVar1;
}
// patchestry:function-end ram:20000000
"""

GOOD = "int FUN_20000000(int param_1) {\n    return param_1 + g_state;\n}\n"
DROPS_GLOBAL = "int FUN_20000000(int param_1) {\n    return param_1;\n}\n"
SYNTAX_ERROR = "int FUN_20000000(int param_1) {\n    return param_1 +;\n}\n"
PROSE = "Here is the cleaned-up function:\n\n```c\n" + GOOD + "```\n\nLet me know if you need more."


def test_render_unit_round_trips_a_printed_unit():
    unit = parse_printed_unit(FLAT_UNIT)
    assert render_unit(unit) == FLAT_UNIT
    replaced = render_unit(unit, {FUNCTION_KEY: GOOD})
    assert "return param_1 + g_state;" in replaced and "iVar1" not in replaced
    assert replaced.count("// patchestry:function-begin") == 1


def test_clean_definition_strips_fences_prose_and_markers():
    assert clean_definition(GOOD, "FUN_20000000") == GOOD
    assert clean_definition(PROSE, "FUN_20000000") == GOOD
    marked = "// patchestry:function-begin x name=y symbol=z\n" + GOOD + "// patchestry:function-end x\n"
    assert clean_definition(marked, "FUN_20000000") == GOOD
    assert clean_definition("I cannot rewrite this.", "FUN_20000000") is None
    assert clean_definition("int other(void) { return 0; }", "FUN_20000000") is None


def test_parse_errors_reads_clang_and_from_c_lines():
    stderr = (
        "[ERROR] (/x/Diagnostic.hpp:77) Diag Error: [ERROR] a.c:3:15: expected expression\n"
        "[ERROR] (/x/main.cpp:468) from-c: 1 error(s) parsing /tmp/a.c\n"
        "[INFO] (/x/y.cpp:1) noise\n"
    )
    assert parse_errors(stderr) == ["a.c:3:15: expected expression", "1 error(s) parsing /tmp/a.c"]


def test_tier2_accepts_a_faithful_rewrite(patchir_decomp, program, tmp_path):
    provider = FakeProvider({FUNCTION_KEY: GOOD})
    result = run_tier2(program, binary=patchir_decomp, provider=provider,
                       options=Tier2Options(prompt_dir=tmp_path / "prompts"), workdir=tmp_path / "work")
    assert [o.status for o in result.outcomes] == ["accepted"]
    attempt = result.outcomes[0].attempts[0]
    assert attempt.accepted and attempt.verdict == "pass" and attempt.flags == []
    assert "return param_1 + g_state;" in result.unit_text and "iVar1" not in result.unit_text
    assert result.verified
    assert result.final_validation["summary"]["failed"] == 0
    assert (tmp_path / "work" / "final.cir").exists()
    prompt = provider.calls[0][1]
    assert "Function key: ram:20000000" in prompt
    assert "extern int g_state;" in prompt and "Function to rewrite:" in prompt
    assert (tmp_path / "prompts" / "ram_20000000.attempt1.reply.txt").exists()


def test_tier2_feeds_findings_back_and_accepts_a_later_attempt(patchir_decomp, program, tmp_path):
    provider = FakeProvider({FUNCTION_KEY: [DROPS_GLOBAL, SYNTAX_ERROR, PROSE]})
    result = run_tier2(program, binary=patchir_decomp, provider=provider,
                       options=Tier2Options(attempts=3), workdir=tmp_path)
    outcome = result.outcomes[0]
    assert outcome.status == "accepted"
    first, second, third = outcome.attempts
    assert first.verdict == "fail" and {f["code"] for f in first.flags} >= {"GLOBAL_LOST"}
    assert not first.accepted
    assert second.verdict is None and second.errors and "expected expression" in second.errors[0]
    assert third.accepted and third.text == GOOD
    retry_prompt = provider.calls[1][1]
    assert "Attempt 1 was rejected" in retry_prompt and "GLOBAL_LOST" in retry_prompt
    assert "Attempt 2 was rejected" in provider.calls[2][1] and "compiler:" in provider.calls[2][1]
    assert "return param_1 + g_state;" in result.unit_text


def test_tier2_keeps_the_flat_body_when_every_attempt_fails(patchir_decomp, program, tmp_path):
    provider = FakeProvider({"*": DROPS_GLOBAL})
    result = run_tier2(program, binary=patchir_decomp, provider=provider,
                       options=Tier2Options(attempts=2, functions=[FUNCTION_KEY, "ram:nope"]), workdir=tmp_path)
    statuses = {o.key: o.status for o in result.outcomes}
    assert statuses == {"ram:nope": "missing", FUNCTION_KEY: "kept-flat"}
    kept = [o for o in result.outcomes if o.key == FUNCTION_KEY][0]
    assert len(kept.attempts) == 2 and all(a.verdict == "fail" for a in kept.attempts)
    assert "iVar1 = param_1 + g_state;" in result.unit_text
    assert result.verified and result.final_validation["summary"]["failed"] == 0
    report = result.report()
    assert report["summary"] == {"missing": 1, "kept-flat": 1}
    assert report["functions"][1]["attempts"][0]["flags"][0]["code"] == "GLOBAL_LOST"
