# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Claude through the official `anthropic` SDK (Messages API)."""

from __future__ import annotations

from . import Completion, ProviderError

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, *, max_tokens: int = DEFAULT_MAX_TOKENS, client=None):
        import anthropic

        self._sdk = anthropic
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        # Credentials come from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
        # `ant auth login` profile; the SDK retries 429/5xx on its own.
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str) -> Completion:
        sdk = self._sdk
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except sdk.RateLimitError as error:
            raise ProviderError(f"anthropic: rate limited: {error}") from error
        except sdk.APIStatusError as error:
            raise ProviderError(f"anthropic: HTTP {error.status_code}: {error.message}") from error
        except sdk.APIConnectionError as error:
            raise ProviderError(f"anthropic: connection failed: {error}") from error
        if response.stop_reason == "refusal":
            raise ProviderError("anthropic: the model declined the request")
        if response.stop_reason == "max_tokens":
            raise ProviderError(
                f"anthropic: reply truncated at {self.max_tokens} tokens; raise --max-tokens"
            )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            provider=self.name,
            model=getattr(response, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
