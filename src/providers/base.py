"""Provider abstractions for prompting baseline runs.

A provider wraps a concrete model backend (local vLLM today; API providers
later). Its single responsibility is to turn a list of `Example`s plus a
common prompt suffix into a list of `GenerationResult`s, driving whatever
chat template + sampling parameters are appropriate for the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.prompting.dataset import Example


@dataclass
class GenerationResult:
    id: str
    prompt: str
    raw_output: str
    finish_reason: str
    tokens_out: int


class Provider(Protocol):
    def generate(
        self, examples: list[Example], prompt_suffix: str
    ) -> list[GenerationResult]: ...

    def close(self) -> None: ...


def build_provider(cfg: dict[str, Any]) -> Provider:
    """Dispatch on `cfg['provider']` to a concrete backend.

    Only `local_vllm` is registered so far; API providers will slot in here
    once network access and keys are available.
    """
    provider_name = cfg.get('provider')
    if provider_name == 'local_vllm':
        from src.providers.local_vllm import LocalVLLMProvider

        return LocalVLLMProvider(cfg)
    raise ValueError(f'Unknown provider: {provider_name!r}')
