"""Local vLLM provider for prompting baseline and LoRA adapter runs.

Uses one local vLLM engine for base-model and LoRA Adapter inference.
Baseline runs load the base model only; LoRA Adapter runs additionally attach
an adapter via vLLM's `LoRARequest`, matching how the evaluator combines
base model and adapter at inference time.

Key behaviours:
- The chat template is applied via the tokenizer, with `enable_thinking`
  passed through when the model config asks for it. If the template does
  not accept that kwarg, we retry without it so templates like GLM-4.6
  (which uses a different flag or none at all) still work.
- If `cfg['adapter_path']` is set, vLLM is started with `enable_lora=True`
  and every generate call carries a `LoRARequest`. Otherwise the base
  model is used verbatim — baseline behaviour is unchanged.
- One vLLM instance per process. `close()` releases GPU memory so shell
  loops that iterate over configs stay clean.
"""

from __future__ import annotations

from typing import Any

from src.prompting.dataset import Example
from src.providers.base import GenerationResult


class LocalVLLMProvider:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._llm = None  # lazy init to keep import cost off the CLI
        self._lora_request = None
        # Optional LoRA adapter directory (peft-format `adapter_config.json`
        # + `adapter_model.safetensors`). None -> base model only.
        self._adapter_path = cfg.get('adapter_path')

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        # vLLM imports are heavy (torch/CUDA); defer until we actually need them.
        from vllm import LLM

        vllm_kwargs = dict(self.cfg.get('vllm', {}))
        if self._adapter_path:
            # enable_lora + max_lora_rank must be set at engine start; the
            # per-request LoRARequest cannot override them later. 32 matches
            # the competition ceiling (max_lora_rank=32) and every adapter
            # we ship stays at or below that.
            vllm_kwargs.setdefault('enable_lora', True)
            vllm_kwargs.setdefault('max_lora_rank', 32)
        self._llm = LLM(model=self.cfg['model_path'], **vllm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

        if self._adapter_path:
            from vllm.lora.request import LoRARequest

            # lora_name is a human label; lora_int_id must be a positive
            # int unique within this engine — we only ever load one.
            self._lora_request = LoRARequest(
                lora_name='adapter',
                lora_int_id=1,
                lora_path=self._adapter_path,
            )

    def generate(
        self, examples: list[Example], prompt_suffix: str
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        self._lazy_init()
        assert self._llm is not None

        chat_cfg = self.cfg.get('chat', {}) or {}
        enable_thinking = chat_cfg.get('enable_thinking')
        system_prompt = chat_cfg.get('system')

        prompts: list[str] = []
        for ex in examples:
            user_content = ex.prompt
            if prompt_suffix:
                # Match the notebook: suffix is appended to the user message
                # with a leading newline.
                user_content = user_content + '\n' + prompt_suffix

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': user_content})

            template_kwargs = {
                'tokenize': False,
                'add_generation_prompt': True,
            }
            if enable_thinking is not None:
                template_kwargs['enable_thinking'] = enable_thinking

            try:
                prompt_text = self._tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError:
                # Template does not accept `enable_thinking` (or another
                # optional kwarg); drop it and retry.
                template_kwargs.pop('enable_thinking', None)
                prompt_text = self._tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
            prompts.append(prompt_text)

        sampling_params = SamplingParams(**self.cfg.get('sampling', {}))
        generate_kwargs: dict[str, Any] = {'sampling_params': sampling_params}
        if self._lora_request is not None:
            generate_kwargs['lora_request'] = self._lora_request
        outputs = self._llm.generate(prompts, **generate_kwargs)

        results: list[GenerationResult] = []
        for ex, prompt_text, output in zip(examples, prompts, outputs):
            first = output.outputs[0]
            results.append(
                GenerationResult(
                    id=ex.id,
                    prompt=prompt_text,
                    raw_output=first.text,
                    finish_reason=(first.finish_reason or ''),
                    tokens_out=len(first.token_ids),
                )
            )
        return results

    def close(self) -> None:
        if self._llm is None:
            return
        try:
            del self._llm
        finally:
            self._llm = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
