from patchestry_llm.decomp import parse_printed_unit, refine_warnings, run_patchir_decomp

PRINTED = """\
// patchestry:tu format=1 target=ARM:LE:32:Cortex arch=ARM
typedef unsigned char undefined1;
extern int g_state;
int FUN_20000000(int param_1);
// patchestry:function-begin ram:20000000 name=FUN_20000000 symbol=FUN_20000000
int FUN_20000000(int param_1) {
    return param_1 + g_state;
}
// patchestry:function-end ram:20000000
"""


def test_parse_printed_unit_splits_header_preamble_and_functions():
    unit = parse_printed_unit(PRINTED)
    assert unit.header == {"format": "1", "target": "ARM:LE:32:Cortex", "arch": "ARM"}
    assert "extern int g_state;" in unit.preamble
    assert list(unit.functions) == ["ram:20000000"]
    function = unit.functions["ram:20000000"]
    assert function.name == "FUN_20000000" and function.symbol == "FUN_20000000"
    assert function.text.startswith("int FUN_20000000(int param_1) {")
    assert "patchestry:" not in function.text


def test_refine_warnings_are_extracted_from_glog_lines():
    stderr = (
        "[INFO] (x.cpp:1) something\n"
        "[WARNING] (/a/FunctionBuilder.cpp:292) refine: get_state: 2 DECLARE_PARAMETER op(s) but the prototype has 1; using prototype types and default names\n"
        "[WARNING] (/a/TypeBuilder.cpp:155) refine: device_ctx is 8 bytes in C but 16 in the P-Code model\n"
    )
    assert refine_warnings(stderr) == [
        "get_state: 2 DECLARE_PARAMETER op(s) but the prototype has 1; using prototype types and default names",
        "device_ctx is 8 bytes in C but 16 in the P-Code model",
    ]


def test_run_patchir_decomp_prints_marked_c(patchir_decomp, program_file, tmp_path):
    result = run_patchir_decomp(patchir_decomp, program_file, tmp_path / "out", print_tu=True, emit_cir=True)
    assert result.ok, result.stderr
    assert result.output(".cir").exists()
    unit = result.printed_unit()
    assert unit.header["target"] == "ARM:LE:32:Cortex"
    assert "ram:20000000" in unit.functions
    assert "param_1" in unit.functions["ram:20000000"].text
    assert result.warnings == []
