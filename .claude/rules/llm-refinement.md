---
description: LLM refinement stage (scripts/llm, patchestry-refine)
paths:
  - "scripts/llm/**"
---

# LLM Refinement Stage

`scripts/llm` is a `uv` project (`patchestry_llm`, CLI `patchestry-refine`).
It never links against the decompiler: it runs `patchir-decomp`, reads the
printed C and its `// patchestry:` markers, and greps `refine:` warnings.

## Key Files

| Task | Files |
|------|-------|
| Run the decompiler, parse markers and `refine:` warnings | `patchestry_llm/decomp.py` |
| Read the Ghidra JSON (params, locals, globals, types) | `patchestry_llm/model.py` |
| Validate and apply a proposal (the only place that edits JSON) | `patchestry_llm/proposal.py` |
| Tier 1 prompt text | `patchestry_llm/prompt.py` |
| Providers (`anthropic`, `openai`, `fake`) | `patchestry_llm/providers/` |
| Tier 1 orchestration and provenance | `patchestry_llm/tier1.py` |
| CLI | `patchestry_llm/cli.py` |

## Rules

- Correctness checks live in C++ (`PcodeValidator`, `refine:` warnings). Python
  validates shape and namespaces before applying an edit; it does not reason
  about semantics.
- Never write SDK calls from memory: the `claude-api` skill for `anthropic`,
  context7 for `openai`. Keep each provider in its own module.
- New edit kinds go through `Refiner` with a rejection reason and a test.
- The `fake` provider is the test path; no test may need a network or a key.

## Test

```sh
uv run --project scripts/llm pytest
```

Decompiler-backed tests use `PATCHIR_DECOMP`, `PATH`, or the build tree and
skip otherwise. Build Debug first (`cmake --build --preset debug -j`).
