# patchestry-llm

Out-of-process LLM refinement for `patchir-decomp`. The decompiler stays
hermetic: this package runs it, reads what it prints, asks a model for
improvements, and hands the result back for the tool to lift and check.

Tier 1 (`patchestry-refine tier1`) proposes names, comments and types and
applies them as plain edits to a copy of the Ghidra export JSON:
`display_name` on functions, `name` on `DECLARE_PARAMETER` / `DECLARE_LOCAL`
ops and on globals, `comment` on functions, new `types` entries, and
same-size retypes of parameters, locals and return values. Every edit is
validated before it lands (C identifiers, unique names, byte sizes, struct
layouts); rejected edits go to the report. The refined JSON is lifted once
more so the lifter's `refine:` warnings and any hard failure surface in the
report and exit code.

Tier 2 (`patchestry-refine tier2`) replaces the structuring engine: the
decompiler prints the flat goto C (`-emit-flat-baseline -print-tu`), the
model rewrites one function at a time into structured C, and the reply is
spliced into the unit and checked with `patchir-decomp -from-c -input
<json> -validate-pcode`. A reply is kept when its function parses, lowers
and validates (`pass`, or `warn` unless `--no-accept-warnings`); otherwise
the validator findings and compiler errors go back to the model for another
attempt (`--attempts`, default 3), and after the last one the flat body
stays. Correctness is decided by the tool, never by this package.

## Install

```sh
cd scripts/llm
uv sync            # creates .venv with the anthropic and openai SDKs and pytest
```

Credentials come from the SDKs' usual environment: `ANTHROPIC_API_KEY` (or an
`ant auth login` profile) and `OPENAI_API_KEY`. The `fake` provider needs
neither.

## Use

```sh
# Claude (default provider), one prompt per function, re-lift and report
uv run --project scripts/llm patchestry-refine tier1 \
    --input func.json --output func.refined.json

# OpenAI, only two functions, keep prompts and replies for inspection
uv run --project scripts/llm patchestry-refine tier1 \
    --input func.json --output func.refined.json --provider openai \
    --function ram:00022cd4 --function ram:000230a0 --prompt-dir /tmp/prompts

# Replay canned proposals without a network
uv run --project scripts/llm patchestry-refine tier1 \
    --input func.json --output func.refined.json \
    --provider fake --fake-responses proposals.json

# Tier 2: structured C from the flat lift, validated against the P-Code;
# writes structured.c, structured.validation.json, structured.report.json, structured.cir
uv run --project scripts/llm patchestry-refine tier2 \
    --input func.refined.json --output structured --emit-cir --with-instructions
```

Tier 2 exit code 2 means the final unit did not lift, or `--strict` saw a
function that kept its flat body. The `.c` output is a complete
translation unit with the `patchestry:` markers, so it re-enters the tool
with `-from-c` as is.

The decompiler is found through `--patchir-decomp`, `$PATCHIR_DECOMP`, `PATH`,
or the repository build tree. `--lean` prompts with the flat goto C
(`-emit-flat-baseline`); `--with-instructions` adds the disassembly from a
`--emit-instructions` export to each prompt. Exit code 2 means the refined
JSON did not lift, or `--strict` saw `refine:` warnings; the output and the
report (`<output>.report.json`) are still written.

## Proposal format

A reply is one JSON object; keys are the ones the prompt lists.

```json
{
  "functions": {
    "ram:20000000": {
      "display_name": "get_state",
      "comment": "Adds the counter to the global state.",
      "parameters": {"0": {"name": "count"}},
      "locals": {"l0": {"name": "acc", "type": "t_u32"}},
      "return_type": "t_int"
    }
  },
  "globals": {"ram:30000000": {"name": "g_state"}},
  "types": {
    "t_ctx": {"kind": "struct", "name": "device_ctx", "size": 8,
              "fields": [{"name": "fd", "type": "t_int", "offset": 0},
                         {"name": "flags", "type": "t_int", "offset": 4}]},
    "t_ctxp": {"kind": "pointer", "size": 4, "element_type": "t_ctx"}
  }
}
```

The refined JSON carries a top-level `refinement` object (tool, provider,
model, timestamp, per-function counts) that the lifter ignores.

## Test

```sh
uv run --project scripts/llm pytest
```

Tests that need the decompiler skip when it is not built; set
`PATCHIR_DECOMP` to point at one outside the build tree.
