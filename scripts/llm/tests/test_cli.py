import json

from patchestry_llm.cli import main

from conftest import FUNCTION_KEY


def test_cli_tier1_with_fake_provider(patchir_decomp, program_file, tmp_path, capsys):
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps({FUNCTION_KEY: {"functions": {FUNCTION_KEY: {"display_name": "get_state"}}}}))
    output = tmp_path / "out" / "refined.json"
    code = main(["tier1", "--input", str(program_file), "--output", str(output), "--provider", "fake",
                 "--fake-responses", str(replies), "--patchir-decomp", str(patchir_decomp), "--pretty",
                 "--workdir", str(tmp_path / "work")])
    assert code == 0
    refined = json.loads(output.read_text())
    assert refined["functions"][FUNCTION_KEY]["display_name"] == "get_state"
    report = json.loads((tmp_path / "out" / "refined.json.report.json").read_text())
    assert report["summary"] == {"applied": 1}
    assert report["verify"]["returncode"] == 0
    err = capsys.readouterr().err
    assert "applied (1 accepted, 0 rejected)" in err


def test_cli_needs_fake_responses(tmp_path, program_file):
    code = main(["tier1", "--input", str(program_file), "--output", str(tmp_path / "o.json"),
                 "--provider", "fake", "--patchir-decomp", "/nonexistent/patchir-decomp"])
    assert code == 1


def test_cli_tier2_with_fake_provider(patchir_decomp, program_file, tmp_path, capsys):
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps({FUNCTION_KEY: "int FUN_20000000(int param_1) {\n    return param_1 + g_state;\n}\n"}))
    prefix = tmp_path / "out" / "structured"
    code = main(["tier2", "--input", str(program_file), "--output", str(prefix), "--provider", "fake",
                 "--fake-responses", str(replies), "--patchir-decomp", str(patchir_decomp), "--emit-cir",
                 "--workdir", str(tmp_path / "work")])
    assert code == 0
    unit = (tmp_path / "out" / "structured.c").read_text()
    assert "return param_1 + g_state;" in unit and unit.startswith("// patchestry:tu format=1")
    assert (tmp_path / "out" / "structured.cir").exists()
    validation = json.loads((tmp_path / "out" / "structured.validation.json").read_text())
    assert validation["functions"][FUNCTION_KEY]["verdict"] == "pass"
    report = json.loads((tmp_path / "out" / "structured.report.json").read_text())
    assert report["summary"] == {"accepted": 1}
    assert "accepted after 1 attempt(s)" in capsys.readouterr().err


def test_cli_tier2_strict_fails_on_fallback(patchir_decomp, program_file, tmp_path):
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps({"*": "int FUN_20000000(int param_1) {\n    return param_1;\n}\n"}))
    prefix = tmp_path / "structured"
    code = main(["tier2", "--input", str(program_file), "--output", str(prefix), "--provider", "fake",
                 "--fake-responses", str(replies), "--patchir-decomp", str(patchir_decomp), "--attempts", "1",
                 "--strict"])
    assert code == 2
    assert "iVar1" in (tmp_path / "structured.c").read_text()  # flat body kept, still written
