import json

from patchestry_llm.providers.fake import FakeProvider
from patchestry_llm.tier1 import Tier1Options, run_tier1

from conftest import FUNCTION_KEY

PROPOSAL = {
    "functions": {FUNCTION_KEY: {
        "display_name": "get_state",
        "comment": "Adds the argument to the global state and returns the sum.",
        "parameters": {"0": {"name": "count"}},
        "locals": {"l0": {"name": "sum", "type": "t_u32"}},
    }},
    "globals": {"ram:30000000": {"name": "state_counter"}},
    "types": {
        "t_ctx": {"kind": "struct", "name": "device_ctx", "size": 8,
                  "fields": [{"name": "fd", "type": "t_int", "offset": 0},
                             {"name": "flags", "type": "t_int", "offset": 4}]},
    },
}


def test_tier1_refines_and_relifts_cleanly(patchir_decomp, program, tmp_path):
    provider = FakeProvider({FUNCTION_KEY: PROPOSAL})
    result = run_tier1(program, binary=patchir_decomp, provider=provider,
                       options=Tier1Options(prompt_dir=tmp_path / "prompts"), workdir=tmp_path / "work",
                       source_name="sample.json")
    assert [o.status for o in result.outcomes] == ["applied"]
    assert result.outcomes[0].rejected == []
    refined = result.program
    assert list(refined)[:4] == ["architecture", "id", "format", "refinement"]
    assert refined["refinement"]["provider"] == "fake" and refined["refinement"]["tier"] == 1
    assert refined["refinement"]["functions"][FUNCTION_KEY]["accepted"] == 7  # name, comment, param, local name+type, global, type
    assert refined["functions"][FUNCTION_KEY]["display_name"] == "get_state"
    assert refined["globals"]["ram:30000000"]["name"] == "state_counter"
    assert "t_ctx" in refined["types"]
    assert result.verified and result.warnings == []
    printed = result.verify.printed_unit()
    body = printed.functions[FUNCTION_KEY].text
    assert body.startswith("/* Adds the argument to the global state")
    assert "int get_state(int count) {" in body
    assert "state_counter" in body and "unsigned int sum;" in body
    assert "struct device_ctx" in printed.preamble
    # prompts and replies were saved
    assert (tmp_path / "prompts" / "system.txt").exists()
    assert (tmp_path / "prompts" / "ram_20000000.prompt.txt").exists()
    assert "Function key: ram:20000000" in provider.calls[0][1]


def test_tier1_keeps_going_past_bad_replies(patchir_decomp, program, tmp_path):
    provider = FakeProvider({FUNCTION_KEY: "Sorry, I cannot help with that."})
    result = run_tier1(program, binary=patchir_decomp, provider=provider,
                       options=Tier1Options(), workdir=tmp_path)
    assert [o.status for o in result.outcomes] == ["unparsable"]
    assert result.program["functions"][FUNCTION_KEY].get("display_name") is None
    assert result.verified


def test_tier1_reports_rejections_and_missing_functions(patchir_decomp, program, tmp_path):
    provider = FakeProvider({"*": {"functions": {FUNCTION_KEY: {"display_name": "while"}}}})
    result = run_tier1(program, binary=patchir_decomp, provider=provider,
                       options=Tier1Options(functions=[FUNCTION_KEY, "ram:deadbeef"]), workdir=tmp_path)
    statuses = {o.key: o.status for o in result.outcomes}
    assert statuses == {"ram:deadbeef": "missing", FUNCTION_KEY: "rejected"}
    rejected = [o for o in result.outcomes if o.key == FUNCTION_KEY][0].rejected
    assert rejected[0]["where"] == f"functions[{FUNCTION_KEY}].display_name"
