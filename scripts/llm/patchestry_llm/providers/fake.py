# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Canned replies keyed by the `Function key:` line of the prompt.

For tests and for replaying a saved proposal without a network.  A JSON
file passed as `--fake-responses` maps function keys to a proposal
object (or a reply string); the key "*" is the fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import Completion, ProviderError

FUNCTION_KEY = re.compile(r"^Function key: (\S+)$", re.MULTILINE)


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        responses_file: str | Path | None = None,
        model: str | None = None,
        **_ignored,
    ):
        self.model = model or "none"
        self.responses: dict[str, str] = {}
        merged: dict[str, Any] = {}
        if responses_file:
            merged.update(json.loads(Path(responses_file).read_text()))
        if responses:
            merged.update(responses)
        for key, value in merged.items():
            self.responses[key] = value if isinstance(value, str) else json.dumps(value)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        match = FUNCTION_KEY.search(user)
        key = match.group(1) if match else ""
        text = self.responses.get(key, self.responses.get("*"))
        if text is None:
            raise ProviderError(f"fake: no canned reply for function {key!r}")
        return Completion(text=text, provider=self.name, model=self.model, input_tokens=0, output_tokens=0)
