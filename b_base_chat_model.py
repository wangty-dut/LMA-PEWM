from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _env_int(name: str, default: int) -> int:
    try:
        value = os.environ.get(name, "").strip()
        return int(value) if value else int(default)
    except Exception:
        return int(default)

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

class BaseChatModel:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.1,
        do_sample: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_sleep: Optional[float] = None,
        force_json_response: Optional[bool] = None,
    ) -> None:
        self.model_name = str(model_path)
        self.model_path = str(Path(model_path).expanduser())
        self.device = self._resolve_device(device)

        if max_new_tokens is None:
            max_new_tokens = _env_int("LLM_MAX_NEW_TOKENS", 32768)

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = temperature
        self.do_sample = bool(do_sample)
        self.enable_thinking = (
            _env_bool("LLM_ENABLE_THINKING", False)
            if enable_thinking is None
            else bool(enable_thinking)
        )
        self.force_json_response = (
            _env_bool("LLM_FORCE_JSON_RESPONSE", False)
            if force_json_response is None
            else bool(force_json_response)
        )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }

        if self.device.startswith("cuda"):
            model_kwargs["torch_dtype"] = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        elif self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = str(device or "auto").strip().lower()

        if requested in {"auto", "api"}:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")

        if requested == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")

        return requested

    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prepared = [dict(message) for message in messages]

        if not self.force_json_response:
            return prepared

        instruction = "Return only one valid JSON object with no Markdown fences or extra text."

        if prepared and prepared[0].get("role") == "system":
            content = str(prepared[0].get("content", "")).rstrip()
            prepared[0]["content"] = f"{content}\n{instruction}".strip()
        else:
            prepared.insert(0, {"role": "system", "content": instruction})

        return prepared

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        prepared = self._prepare_messages(messages)

        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        kwargs["enable_thinking"] = bool(self.enable_thinking)

        try:
            return self.tokenizer.apply_chat_template(prepared, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(prepared, **kwargs)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._build_prompt(messages)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample and self.temperature is not None:
            generation_kwargs["temperature"] = float(self.temperature)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                **generation_kwargs,
            )

        input_length = encoded["input_ids"].shape[-1]
        generated_ids = output_ids[0, input_length:]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        return text

    def release(self) -> None:
        if getattr(self, "model", None) is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
