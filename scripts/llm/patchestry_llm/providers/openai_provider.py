# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""OpenAI models through the official `openai` SDK (Responses API)."""

from __future__ import annotations

from . import Completion, ProviderError

DEFAULT_MODEL = "gpt-5.5"


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None, *, client=None, **_ignored):
        import openai

        self._sdk = openai
        self.model = model or DEFAULT_MODEL
        # Credentials come from OPENAI_API_KEY; the SDK retries on its own.
        self.client = client or openai.OpenAI()

    def complete(self, system: str, user: str) -> Completion:
        sdk = self._sdk
        try:
            response = self.client.responses.create(
                model=self.model, instructions=system, input=user
            )
        except sdk.RateLimitError as error:
            raise ProviderError(f"openai: rate limited: {error}") from error
        except sdk.APIStatusError as error:
            raise ProviderError(f"openai: HTTP {error.status_code}: {error}") from error
        except sdk.APIConnectionError as error:
            raise ProviderError(f"openai: connection failed: {error}") from error
        text = getattr(response, "output_text", "") or ""
        if not text:
            raise ProviderError("openai: the reply carried no text")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            provider=self.name,
            model=getattr(response, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
