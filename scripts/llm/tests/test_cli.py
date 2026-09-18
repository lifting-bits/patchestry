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
