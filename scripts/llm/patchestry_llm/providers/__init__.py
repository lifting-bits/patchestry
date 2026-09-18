# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Model providers: one `complete(system, user)` call each.

The SDKs are imported lazily so the fake provider and the tests never need
them installed or an API key set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

PROVIDER_NAMES = ("anthropic", "openai", "fake")


class ProviderError(RuntimeError):
    """The provider could not return a usable completion."""


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> Completion: ...


def make_provider(name: str, model: str | None = None, **options) -> Provider:
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, **options)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, **options)
    if name == "fake":
        from .fake import FakeProvider

        return FakeProvider(**options)
    raise ProviderError(f"unknown provider {name!r}; expected one of {', '.join(PROVIDER_NAMES)}")
