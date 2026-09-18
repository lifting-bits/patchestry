import pytest

from patchestry_llm.model import function_inventory
from patchestry_llm.proposal import Refiner, extract_json, is_identifier

from conftest import FUNCTION_KEY


def apply(program, proposal, **kw):
    refiner = Refiner(program)
    return refiner, refiner.apply(proposal, **kw)


def test_is_identifier_rejects_keywords_and_bad_spellings():
    assert is_identifier("get_state")
    assert is_identifier("_buf2")
    assert not is_identifier("int")
    assert not is_identifier("2fast")
    assert not is_identifier("has space")
    assert not is_identifier("")
    assert not is_identifier(None)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("[1, 2]")


def test_renames_and_comment_land_in_the_copy(program):
    proposal = {
        "functions": {FUNCTION_KEY: {
            "display_name": "get_state",
            "comment": "  Adds the counter\n to the state. ",
            "parameters": {"0": {"name": "count"}},
            "locals": {"l0": {"name": "acc"}},
        }},
        "globals": {"ram:30000000": {"name": "state_counter"}},
    }
    refiner, report = apply(program, proposal)
    assert report.rejected == []
    fn = refiner.program["functions"][FUNCTION_KEY]
    assert fn["display_name"] == "get_state"
    assert fn["name"] == "FUN_20000000"  # linker symbol untouched
    assert fn["comment"] == "Adds the counter to the state."
    ops = fn["basic_blocks"]["ram:20000000:entry"]["operations"]
    assert ops["p0"]["name"] == "count" and ops["l0"]["name"] == "acc"
    assert refiner.program["globals"]["ram:30000000"]["name"] == "state_counter"
    # the input program is untouched
    assert program["functions"][FUNCTION_KEY].get("display_name") is None
    assert program["globals"]["ram:30000000"]["name"] == "g_state"
    assert len(report.accepted) == 5


def test_bad_names_are_rejected_not_applied(program):
    proposal = {
        "functions": {FUNCTION_KEY: {
            "display_name": "int",
            "parameters": {"0": {"name": "g_state"}},        # shadows a global
            "locals": {"l0": {"name": "param_1"}},            # duplicates the parameter
        }},
        "globals": {"ram:30000000": {"name": "FUN_20000000"}},  # collides with a function
    }
    refiner, report = apply(program, proposal)
    assert report.accepted == []
    reasons = {r.where: r.reason for r in report.rejected}
    assert f"functions[{FUNCTION_KEY}].display_name" in reasons
    assert f"functions[{FUNCTION_KEY}].parameters[0]" in reasons
    assert f"functions[{FUNCTION_KEY}].locals[l0]" in reasons
    assert "globals[ram:30000000]" in reasons
    assert refiner.program == program


def test_retype_must_keep_the_byte_size(program):
    proposal = {"functions": {FUNCTION_KEY: {
        "parameters": {"0": {"type": "t_u32"}},
        "locals": {"l0": {"type": "t_u8"}},
        "return_type": "t_u8",
    }}}
    refiner, report = apply(program, proposal)
    fn = refiner.program["functions"][FUNCTION_KEY]
    assert fn["basic_blocks"]["ram:20000000:entry"]["operations"]["p0"]["type"] == "t_u32"
    assert fn["type"]["parameter_types"] == ["t_u32"]
    assert fn["basic_blocks"]["ram:20000000:entry"]["operations"]["l0"]["type"] == "t_int"
    assert fn["type"]["return_type"] == "t_int"
    wheres = [r.where for r in report.rejected]
    assert f"functions[{FUNCTION_KEY}].locals[l0]" in wheres
    assert f"functions[{FUNCTION_KEY}].return_type" in wheres


def test_new_types_can_be_used_in_the_same_reply(program):
    proposal = {
        "types": {
            "t_ctx": {"kind": "struct", "name": "device_ctx", "size": 8,
                      "fields": [{"name": "flags", "type": "t_int", "offset": 4},
                                 {"name": "fd", "type": "t_int", "offset": 0}]},
            "t_ctxp": {"kind": "pointer", "size": 4, "element_type": "t_ctx"},
            "t_ctx_td": {"kind": "typedef", "name": "device_ctx_t", "size": 8, "base_type": "t_ctx"},
            "t_buf": {"kind": "array", "size": 8, "element_type": "t_u8", "num_elements": 8},
        },
        "functions": {FUNCTION_KEY: {"locals": {"l0": {"name": "ctx", "type": "t_ctxp"}}}},
    }
    refiner, report = apply(program, proposal)
    assert report.rejected == []
    types = refiner.program["types"]
    assert [f["name"] for f in types["t_ctx"]["fields"]] == ["fd", "flags"]  # sorted by offset
    assert types["t_ctxp"] == {"kind": "pointer", "size": 4, "element_type": "t_ctx"}
    assert types["t_ctx_td"]["base_type"] == "t_ctx"
    assert types["t_buf"]["num_elements"] == 8
    local = refiner.program["functions"][FUNCTION_KEY]["basic_blocks"]["ram:20000000:entry"]["operations"]["l0"]
    assert local == {"mnemonic": "DECLARE_LOCAL", "name": "ctx", "type": "t_ctxp", "kind": "local"}


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ({"kind": "struct", "name": "s", "size": 8,
          "fields": [{"name": "a", "type": "t_int", "offset": 0}, {"name": "b", "type": "t_int", "offset": 2}]},
         "overlaps"),
        ({"kind": "struct", "name": "s", "size": 4, "fields": [{"name": "a", "type": "t_int", "offset": 4}]},
         "past size"),
        ({"kind": "struct", "name": "s", "size": 4, "fields": [{"name": "a", "type": "t_nope", "offset": 0}]},
         "existing type key"),
        ({"kind": "struct", "name": "int", "size": 4, "fields": [{"name": "a", "type": "t_int", "offset": 0}]},
         "C identifier"),
        ({"kind": "enum", "name": "e", "size": 4}, "kind must be"),
        ({"kind": "pointer", "size": "4", "element_type": "t_int"}, "positive integer"),
        ({"kind": "array", "size": 5, "element_type": "t_u8", "num_elements": 4}, "is not"),
        ({"kind": "typedef", "name": "g_state", "size": 4, "base_type": "t_int"}, "already taken"),
    ],
)
def test_malformed_new_types_are_rejected(program, spec, fragment):
    refiner, report = apply(program, {"types": {"t_new": spec}})
    assert "t_new" not in refiner.program["types"]
    assert any(fragment in r.reason for r in report.rejected), report.rejected


def test_existing_type_key_is_not_overwritten(program):
    refiner, report = apply(program, {"types": {"t_int": {"kind": "pointer", "size": 4, "element_type": "t_u8"}}})
    assert refiner.program["types"]["t_int"] == program["types"]["t_int"]
    assert report.rejected and "already exists" in report.rejected[0].reason


def test_only_function_restriction(program):
    proposal = {"functions": {"ram:99999999": {"display_name": "other"}, FUNCTION_KEY: {"display_name": "mine"}}}
    refiner, report = apply(program, proposal, only_function=FUNCTION_KEY)
    assert refiner.program["functions"][FUNCTION_KEY]["display_name"] == "mine"
    assert [r.where for r in report.rejected] == ["functions[ram:99999999]"]


def test_unknown_keys_and_shapes_are_reported(program):
    proposal = {
        "functions": {FUNCTION_KEY: {"parameters": {"7": {"name": "x"}}, "locals": {"nope": {"name": "y"}},
                                     "comment": 12}},
        "globals": {"ram:1": {"name": "z"}},
        "types": "not-an-object",
    }
    refiner, report = apply(program, proposal)
    assert refiner.program == program
    wheres = {r.where for r in report.rejected}
    assert {f"functions[{FUNCTION_KEY}].parameters[7]", f"functions[{FUNCTION_KEY}].locals[nope]",
            f"functions[{FUNCTION_KEY}].comment", "globals[ram:1]", "types"} <= wheres


def test_inventory_reads_declarations_callees_and_globals(program):
    inventory = function_inventory(program, FUNCTION_KEY)
    assert [d.name for d in inventory.parameters] == ["param_1"]
    assert [d.op_key for d in inventory.locals] == ["l0"]
    assert inventory.globals_used == ["ram:30000000"]
    assert inventory.return_type == "t_int" and inventory.has_body
