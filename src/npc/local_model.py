"""Local Hugging Face inference for base and tuned models.

Used by both the Gradio/CLI demo and the eval harness so base-vs-tuned comparisons
run through identical generation settings. Designed to work on a small (4GB) GPU:
loads in bf16/fp16 by default (a 0.6B/1.7B fits), with optional 4-bit and CPU
fallback. Optionally applies a PEFT LoRA adapter on top of a base model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    # Qwen3 supports a "thinking" mode; NPC replies should be non-thinking.
    enable_thinking: bool = False


class NpcModel:
    def __init__(
        self,
        model_id: str,
        adapter_id: Optional[str] = None,
        load_in_4bit: bool = False,
        dtype: str = "auto",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.adapter_id = adapter_id

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype, "auto")

        quant_kwargs = {}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                # bitsandbytes often unavailable on Windows; fall back to fp16.
                load_in_4bit = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **quant_kwargs,
        )

        if adapter_id:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_id)

        self.model.eval()

    def _build_inputs(self, user: str, system: Optional[str], enable_thinking: bool):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # Qwen3 chat template supports enable_thinking; guard for other models.
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def generate(self, user: str, system: Optional[str] = None, cfg: Optional[GenConfig] = None) -> str:
        import torch

        cfg = cfg or GenConfig()
        inputs = self._build_inputs(user, system, cfg.enable_thinking)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        # Strip any leaked Qwen3 <think> ... </think> block just in case.
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text.strip()
