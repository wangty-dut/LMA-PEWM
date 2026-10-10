from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

LOGGER = logging.getLogger("production_full_ft")

@dataclass
class ExperimentConfig:

    train_file: str = r"D:\wyj_project\wyj\Router-R1\data_process\sft_datasets\steel-production\production_router_messages_train.jsonl"
    test_file: Optional[str] = r"D:\wyj_project\wyj\Router-R1\data_process\sft_datasets\steel-production\production_router_messages_test.jsonl"

    model_name_or_path: str = r"C:\Users\Administrator\.cache\modelscope\hub\models\Qwen\Qwen3.5-4B"

    output_dir: str = r"D:\wyj_project\wyj\Router-R1\outputs\qwen35_4b_steel_full_sft"

    max_length: int = 4096
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 3e-6
    weight_decay: float = 0.05
    num_train_epochs: float = 30.0
    warmup_ratio: float = 0.03

    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2

    seed: int = 42

    packing: bool = False
    gradient_checkpointing: bool = True
    use_flash_attn2: bool = True

    optim: str = "adamw_torch"

    train_on_assistant_only: bool = True
    mask_system_and_user_loss: bool = True
    strict_data_check: bool = False
    drop_invalid_samples: bool = True
    preprocessing_num_proc: int = 1
    dataloader_num_workers: int = 0

    min_assistant_tokens: int = 64

    min_prompt_tokens: int = 128

    inject_strict_router_constraints: bool = True

    report_to: str = "none"
    ignore_hardware_check: bool = False

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def detect_total_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total = 0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.get_device_properties(i).total_memory
    return total / (1024 ** 3)

def detect_bf16_support() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def infer_model_size_hint(model_name_or_path: str) -> str:
    name = str(model_name_or_path).lower()
    for s in ["72b", "32b", "27b", "14b", "9b", "8b", "7b", "4b"]:
        if s in name:
            return s
    return "unknown"

def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2)
    if content is None:
        return ""
    return str(content)

def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages must be a non-empty list")

    normalized = []
    for j, turn in enumerate(messages):
        if not isinstance(turn, dict):
            raise ValueError(f"Message {j} is not a dict")

        role = str(turn.get("role", "")).strip()
        if role == "":
            raise ValueError(f"Message {j} is missing role")

        content = normalize_content(turn.get("content", ""))
        normalized.append({"role": role, "content": content})

    return normalized

def validate_messages_record(example: Dict[str, Any], idx: int) -> Tuple[bool, str]:
    try:

        if "messages" not in example:
            return False, f"Sample {idx} is missing the messages field"

        messages = normalize_messages(example["messages"])

        roles = [m.get("role") for m in messages]

        if "assistant" not in roles:
            return False, f"Sample {idx} does not contain an assistant message"

        if roles[-1] != "assistant":
            return False, f"The last message in sample {idx} is not from assistant"

        if "user" not in roles:
            return False, f"Sample {idx} does not contain a user message"

        last_content = messages[-1].get("content", "")

        if not isinstance(last_content, str) or not last_content.strip():
            return False, f"The final assistant content in sample {idx} is empty or not text"

        return True, "ok"

    except Exception as e:
        return False, f"Sample {idx} processing error: {str(e)}"

def validate_dataset(ds: DatasetDict, strict: bool = False) -> Dict[str, List[Tuple[int, str]]]:
    invalid_records: Dict[str, List[Tuple[int, str]]] = {}
    for split_name, split_ds in ds.items():
        LOGGER.info("Checking dataset split=%s, size=%s", split_name, len(split_ds))
        invalid_records[split_name] = []

        for i in range(len(split_ds)):
            ok, reason = validate_messages_record(split_ds[i], i)
            if not ok:
                invalid_records[split_name].append((i, reason))
                if strict:
                    raise ValueError(f"[{split_name}] sample {i} is invalid: {reason}")

        if invalid_records[split_name]:
            LOGGER.warning(
                "split=%s detected %s invalid samples; first 5: %s",
                split_name,
                len(invalid_records[split_name]),
                invalid_records[split_name][:5],
            )
        else:
            LOGGER.info("split=%s passed data validation", split_name)

    return invalid_records

def filter_invalid_dataset(
    ds: DatasetDict,
    invalid_records: Dict[str, List[Tuple[int, str]]],
) -> DatasetDict:
    new_mapping = {}
    for split_name, split_ds in ds.items():
        invalid_idx = {idx for idx, _ in invalid_records.get(split_name, [])}
        if not invalid_idx:
            new_mapping[split_name] = split_ds
            continue

        keep_indices = [i for i in range(len(split_ds)) if i not in invalid_idx]
        LOGGER.warning(
            "split=%s original samples=%s, kept samples=%s, removed invalid samples=%s",
            split_name,
            len(split_ds),
            len(keep_indices),
            len(invalid_idx),
        )
        new_mapping[split_name] = split_ds.select(keep_indices)
    return DatasetDict(new_mapping)

def build_dataset(cfg: ExperimentConfig) -> DatasetDict:
    data_files: Dict[str, str] = {
        "train": cfg.train_file,
    }
    if cfg.test_file and Path(cfg.test_file).exists():
        data_files["test"] = cfg.test_file
    elif cfg.test_file:
        LOGGER.warning("test_file does not exist; test split loading was skipped: %s", cfg.test_file)

    ds = load_dataset("json", data_files=data_files)

    invalid_records = validate_dataset(ds, strict=cfg.strict_data_check)
    if cfg.drop_invalid_samples:
        ds = filter_invalid_dataset(ds, invalid_records)

    return ds

def preview_one_sample(ds: DatasetDict) -> None:
    sample = ds["train"][0]
    messages = normalize_messages(sample["messages"])

    preview = {
        "keys": list(sample.keys()),
        "sample_id": sample.get("sample_id", None),
        "task": sample.get("task", None),
        "num_messages": len(messages),
        "roles": [m["role"] for m in messages],
        "assistant_preview": messages[-1]["content"][:200],
    }
    LOGGER.info("Training sample preview: %s", json.dumps(preview, ensure_ascii=False))

def run_hardware_guard(cfg: ExperimentConfig) -> None:
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    total_mem_gb = detect_total_gpu_memory_gb()
    model_hint = infer_model_size_hint(cfg.model_name_or_path)

    LOGGER.info("Detected GPU count: %s", n_gpu)
    LOGGER.info("Detected total GPU memory (GB): %.2f", total_mem_gb)
    LOGGER.info("Model size hint: %s", model_hint)

    if cfg.ignore_hardware_check:
        LOGGER.warning("ignore_hardware_check=True; skipping hardware blocking checks.")
        return

    if n_gpu == 0:
        raise RuntimeError("No CUDA GPU was detected; training cannot proceed.")

    if model_hint in {"27b", "32b", "72b"} and n_gpu == 1 and total_mem_gb <= 90:
        raise RuntimeError(
            "The current setup appears to use a single GPU with <=90 GB memory, while the model appears to be in the 27B/32B/72B range; full fine-tuning is unlikely to run reliably."
            "Use a smaller model, LoRA/QLoRA, or multiple GPUs."
        )

def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_model(model_name_or_path: str, use_flash_attn2: bool = False):
    dtype = torch.bfloat16 if detect_bf16_support() else torch.float16

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    return model

def build_router_strict_constraint_text() -> str:
    return (
        "You are strictly solving the task production_candidate_action_generation.\n"
        "You must output only the candidate-action result JSON.\n"
        "Do NOT generate a complete production schedule.\n"
        "Do NOT generate rolling_units, start_time tables, duration_min, heats_list, or full sequencing plans.\n"
        "Do NOT change the task name.\n"
        "The output task field must remain exactly production_candidate_action_generation.\n"
        "The output must follow only this JSON structure: task, mode, objective, candidate_actions, candidate_summary.\n"
        "Do NOT add extra fields such as status, summary, key_metrics, constraints_satisfied, rolling_units, or any chain-of-thought content.\n"
        "Use only the provided input evidence.\n"
        "If evidence is insufficient, stay conservative within the required schema instead of inventing details."
    )

def enhance_messages_for_router_task(
    messages: List[Dict[str, str]],
    task_name: str,
    cfg: ExperimentConfig,
) -> List[Dict[str, str]]:
    if not cfg.inject_strict_router_constraints:
        return messages

    if task_name != "production_candidate_action_generation":
        return messages

    enhanced = []
    strict_text = build_router_strict_constraint_text()

    for idx, m in enumerate(messages):
        role = m["role"]
        content = m["content"]

        if idx == 0 and role == "system":
            content = content.rstrip() + "\n\n" + strict_text
        enhanced.append({"role": role, "content": content})

    return enhanced

def apply_chat_template_safe(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        parts = []
        for m in messages:
            role = m["role"].strip().lower()
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

def build_train_text_and_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
) -> Tuple[str, str]:
    full_text = apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
    )

    prompt_messages = messages[:-1]
    prompt_text = apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )
    return full_text, prompt_text

def tokenize_preserve_assistant(
    tokenizer,
    full_text: str,
    prompt_text: str,
    cfg: ExperimentConfig,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    full_ids = tokenizer(
        full_text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    prompt_ids = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    if len(prompt_ids) > len(full_ids):
        raise ValueError("Prompt token length exceeds full token length; chat template processing is invalid")

    assistant_ids = full_ids[len(prompt_ids):]

    if len(assistant_ids) == 0:
        raise ValueError("Assistant token length is 0")

    original_prompt_len = len(prompt_ids)
    original_assistant_len = len(assistant_ids)

    if len(full_ids) <= cfg.max_length:
        input_ids = full_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + assistant_ids
        stats = {
            "original_total_len": len(full_ids),
            "original_prompt_len": original_prompt_len,
            "original_assistant_len": original_assistant_len,
            "kept_prompt_len": len(prompt_ids),
            "kept_assistant_len": len(assistant_ids),
            "was_truncated": 0,
        }
        return input_ids, attention_mask, labels, stats

    if len(assistant_ids) >= cfg.max_length - cfg.min_prompt_tokens:
        kept_assistant_len = cfg.max_length - cfg.min_prompt_tokens
        if kept_assistant_len < cfg.min_assistant_tokens:
            raise ValueError("max_length is too small to preserve the minimum number of supervised assistant tokens")
        assistant_ids = assistant_ids[:kept_assistant_len]
        prompt_budget = cfg.max_length - len(assistant_ids)
    else:
        prompt_budget = cfg.max_length - len(assistant_ids)
        prompt_budget = max(prompt_budget, cfg.min_prompt_tokens)

    if prompt_budget <= 0:
        raise ValueError("prompt_budget <= 0; the sample cannot be constructed")

    kept_prompt_ids = prompt_ids[-prompt_budget:] if len(prompt_ids) > prompt_budget else prompt_ids

    input_ids = kept_prompt_ids + assistant_ids
    if len(input_ids) > cfg.max_length:
        overflow = len(input_ids) - cfg.max_length
        if overflow >= len(kept_prompt_ids):
            raise ValueError("Overflow is too large and would remove the entire prompt")
        kept_prompt_ids = kept_prompt_ids[overflow:]
        input_ids = kept_prompt_ids + assistant_ids

    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(kept_prompt_ids) + assistant_ids

    valid_label_tokens = sum(1 for x in labels if x != -100)
    if valid_label_tokens < cfg.min_assistant_tokens:
        raise ValueError("There are still too few valid supervised assistant tokens after truncation")

    stats = {
        "original_total_len": len(full_ids),
        "original_prompt_len": original_prompt_len,
        "original_assistant_len": original_assistant_len,
        "kept_prompt_len": len(kept_prompt_ids),
        "kept_assistant_len": len(assistant_ids),
        "was_truncated": 1,
    }
    return input_ids, attention_mask, labels, stats

def tokenize_one_example(example: Dict[str, Any], tokenizer, cfg: ExperimentConfig) -> Dict[str, Any]:
    messages = normalize_messages(example["messages"])
    task_name = str(example.get("task", "")).strip()

    messages = enhance_messages_for_router_task(messages, task_name, cfg)

    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message is not from assistant; standard SFT cannot be performed")

    full_text, prompt_text = build_train_text_and_prompt(tokenizer, messages)

    input_ids, attention_mask, labels, stats = tokenize_preserve_assistant(
        tokenizer=tokenizer,
        full_text=full_text,
        prompt_text=prompt_text,
        cfg=cfg,
    )

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": full_text[:4000],
        "prompt_text": prompt_text[-2000:],
        "sample_id": str(example.get("sample_id", "")),
        "task": task_name,
        "original_total_len": stats["original_total_len"],
        "original_prompt_len": stats["original_prompt_len"],
        "original_assistant_len": stats["original_assistant_len"],
        "kept_prompt_len": stats["kept_prompt_len"],
        "kept_assistant_len": stats["kept_assistant_len"],
        "was_truncated": stats["was_truncated"],
    }
    return out

def preprocess_split(split_ds: Dataset, split_name: str, tokenizer, cfg: ExperimentConfig) -> Dataset:
    processed_records = []
    bad_samples: List[Tuple[int, str]] = []

    for i in range(len(split_ds)):
        example = split_ds[i]
        try:
            item = tokenize_one_example(example, tokenizer, cfg)
            processed_records.append(item)
        except Exception as e:
            bad_samples.append((i, str(e)))

    if bad_samples:
        LOGGER.warning(
            "split=%s preprocessing failures=%s; first 5=%s",
            split_name,
            len(bad_samples),
            bad_samples[:5],
        )

    if len(processed_records) == 0:
        raise RuntimeError(f"split={split_name} has no usable samples after preprocessing; check the data or relax the filtering criteria.")

    processed_ds = Dataset.from_list(processed_records)

    truncated_count = int(sum(processed_ds["was_truncated"])) if "was_truncated" in processed_ds.column_names else 0
    LOGGER.info(
        "split=%s preprocessing complete: original=%s, successful=%s, failed=%s, truncated=%s",
        split_name,
        len(split_ds),
        len(processed_ds),
        len(bad_samples),
        truncated_count,
    )
    return processed_ds

def preprocess_dataset(ds: DatasetDict, tokenizer, cfg: ExperimentConfig) -> DatasetDict:
    mapping = {}
    for split_name, split_ds in ds.items():
        mapping[split_name] = preprocess_split(split_ds, split_name, tokenizer, cfg)
    return DatasetDict(mapping)

def preview_tokenized_sample(tokenized_ds: DatasetDict) -> None:
    sample = tokenized_ds["train"][0]
    preview = {
        "sample_id": sample.get("sample_id", ""),
        "task": sample.get("task", ""),
        "input_len": len(sample["input_ids"]),
        "label_len": len(sample["labels"]),
        "num_valid_label_tokens": int(sum(1 for x in sample["labels"] if x != -100)),
        "original_total_len": sample.get("original_total_len", -1),
        "original_prompt_len": sample.get("original_prompt_len", -1),
        "original_assistant_len": sample.get("original_assistant_len", -1),
        "kept_prompt_len": sample.get("kept_prompt_len", -1),
        "kept_assistant_len": sample.get("kept_assistant_len", -1),
        "was_truncated": sample.get("was_truncated", -1),
        "text_preview": sample["text"][:300],
    }
    LOGGER.info("Tokenized sample preview: %s", json.dumps(preview, ensure_ascii=False))

class SaveConfigCallback(TrainerCallback):
    def __init__(self, cfg_dict: Dict[str, Any]) -> None:
        self.cfg_dict = cfg_dict
        self.done = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.done:
            return
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path = out_dir / "run_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg_dict, f, ensure_ascii=False, indent=2)
        self.done = True

def main() -> None:
    setup_logging()

    cfg = ExperimentConfig()

    set_seed(cfg.seed)
    run_hardware_guard(cfg)

    LOGGER.info("Current experiment configuration:")
    LOGGER.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    ds = build_dataset(cfg)
    preview_one_sample(ds)

    tokenizer = load_tokenizer(cfg.model_name_or_path)
    tokenized_ds = preprocess_dataset(ds, tokenizer, cfg)
    preview_tokenized_sample(tokenized_ds)

    model = load_model(cfg.model_name_or_path, use_flash_attn2=cfg.use_flash_attn2)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    os.makedirs(cfg.output_dir, exist_ok=True)

    fp16 = (not detect_bf16_support()) and torch.cuda.is_available()
    bf16 = detect_bf16_support()

    train_cfg = SFTConfig(
        output_dir=cfg.output_dir,
        do_train=True,
        do_eval=False,
        max_length=cfg.max_length,
        packing=False,
        assistant_only_loss=False,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=[],
        optim=cfg.optim,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_first_step=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        processing_class=tokenizer,
        train_dataset=tokenized_ds["train"],
        eval_dataset=None,
        callbacks=[SaveConfigCallback(asdict(cfg))],
    )

    LOGGER.info("Starting training ...")
    train_result = trainer.train()
    LOGGER.info("Training completed.")

    LOGGER.info("Saving final model ...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = dict(train_result.metrics)
    metrics["train_samples_raw"] = len(ds["train"])
    metrics["train_samples_tokenized"] = len(tokenized_ds["train"])
    if "test" in ds:
        metrics["test_samples_raw"] = len(ds["test"])
        metrics["test_samples_tokenized"] = len(tokenized_ds["test"])

    metrics_path = Path(cfg.output_dir) / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    LOGGER.info("Training metrics saved to: %s", metrics_path)

    if "test" in tokenized_ds:
        LOGGER.info("Starting test-set evaluation ...")
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized_ds["test"],
            metric_key_prefix="test",
        )
        test_metrics_path = Path(cfg.output_dir) / "test_metrics.json"
        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        LOGGER.info("Test metrics saved to: %s", test_metrics_path)

    summary = {
        "model_name_or_path": cfg.model_name_or_path,
        "output_dir": cfg.output_dir,
        "train_file": cfg.train_file,
        "test_file": cfg.test_file if "test" in tokenized_ds else None,
        "max_length": cfg.max_length,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "optim": cfg.optim,
        "packing": False,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "bf16": bf16,
        "fp16": fp16,
        "train_on_assistant_only": cfg.train_on_assistant_only,
        "min_assistant_tokens": cfg.min_assistant_tokens,
        "min_prompt_tokens": cfg.min_prompt_tokens,
        "inject_strict_router_constraints": cfg.inject_strict_router_constraints,
        "train_samples_raw": len(ds["train"]),
        "train_samples_tokenized": len(tokenized_ds["train"]),
    }

    if "test" in tokenized_ds:
        summary["test_samples_raw"] = len(ds["test"])
        summary["test_samples_tokenized"] = len(tokenized_ds["test"])

    summary_path = Path(cfg.output_dir) / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("=" * 80)
    LOGGER.info("Production Full FT Summary")
    LOGGER.info("=" * 80)
    LOGGER.info(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
