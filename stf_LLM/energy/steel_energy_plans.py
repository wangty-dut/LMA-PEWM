# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import random
import inspect
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from multiprocessing import freeze_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)

MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\Qwen\Qwen3.5-2B"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(PROJECT_DIR, "energy_llm_train_dataset.jsonl")

OUTPUT_ROOT = os.path.join(PROJECT_DIR, "energy_fullft_outputs")
SPLIT_SAVE_DIR = os.path.join(OUTPUT_ROOT, "splits")
FULL_MODEL_SAVE_PATH = os.path.join(OUTPUT_ROOT, "full_finetuned_model")

RESULTS_SAVE_PATH = os.path.join(OUTPUT_ROOT, "test_results.jsonl")
TRAIN_LOG_JSON = os.path.join(OUTPUT_ROOT, "training_log_history.json")
TRAIN_CURVE_PNG = os.path.join(OUTPUT_ROOT, "training_curve.png")
RUN_CONFIG_JSON = os.path.join(OUTPUT_ROOT, "run_config.json")
TRAIN_METRICS_JSON = os.path.join(OUTPUT_ROOT, "train_metrics.json")
VAL_METRICS_JSON = os.path.join(OUTPUT_ROOT, "val_metrics.json")
TEST_METRICS_JSON = os.path.join(OUTPUT_ROOT, "test_metrics.json")

# =========================
# Dataset split configuration
# =========================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# =========================
# Training configuration
# =========================
SEED = 42
PRINT_FULL_OUTPUT = False

# Use single-GPU/standard training first; keep FSDP disabled for now
USE_FSDP_IF_MULTI_GPU = False

LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

# Key settings: total length and prompt/answer allocation
MAX_SEQ_LEN = 65536
MAX_PROMPT_TOKENS = 49152
MAX_ANSWER_TOKENS = 16384
MAX_NEW_TOKENS = 8192

DEBUG_PRINT_FIRST_N_SAMPLES = 3


# =========================
# Handle TrainingArguments parameter-name differences across transformers versions
# =========================
def make_training_args_kwargs(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    safe_kwargs = {}
    for k, v in kwargs.items():
        if k in params:
            safe_kwargs[k] = v
    return safe_kwargs


# =========================
# Training monitoring callback
# =========================
class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self.epoch_metrics = []
        self.current_epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_logs = {"epoch": float(self.current_epoch), "train_loss": None, "eval_loss": None}

        train_losses = [log["loss"] for log in state.log_history if "loss" in log]
        if train_losses:
            epoch_logs["train_loss"] = float(np.mean(train_losses))

        eval_logs = [log for log in state.log_history if "eval_loss" in log]
        if eval_logs:
            epoch_logs["eval_loss"] = float(eval_logs[-1]["eval_loss"])

        self.epoch_metrics.append(epoch_logs)
        self.current_epoch += 1

    def save_to_csv(self, prefix="energy_fullft_metrics"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_ROOT, f"{prefix}_{timestamp}.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "eval_loss"])
            writer.writeheader()
            for m in self.epoch_metrics:
                writer.writerow({
                    "epoch": m["epoch"],
                    "train_loss": "" if m["train_loss"] is None else m["train_loss"],
                    "eval_loss": "" if m["eval_loss"] is None else m["eval_loss"],
                })
        print(f"[Monitor] metrics saved to: {filename}")


# =========================
# JSONL loading / saving
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"[Data] Failed to parse JSON on line {line_idx}: {e}")
    return data


def save_jsonl(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# =========================
# Dataset splitting
# =========================
def split_dataset(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        raise ValueError("[Data] The dataset is empty and cannot be split")

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
        raise ValueError("[Data] TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1")

    records = records.copy()
    random.Random(SPLIT_SEED).shuffle(records)

    n = len(records)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    if len(val_records) == 0 and len(test_records) > 1:
        val_records.append(test_records.pop())
    if len(test_records) == 0 and len(val_records) > 1:
        test_records.append(val_records.pop())

    return train_records, val_records, test_records


def build_split_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    print(f"[Data] Loading dataset file: {DATASET_FILE}")
    all_records = load_jsonl(DATASET_FILE)

    train_records, val_records, test_records = split_dataset(all_records)

    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)
    save_jsonl(train_records, os.path.join(SPLIT_SAVE_DIR, "train.jsonl"))
    save_jsonl(val_records, os.path.join(SPLIT_SAVE_DIR, "val.jsonl"))
    save_jsonl(test_records, os.path.join(SPLIT_SAVE_DIR, "test.jsonl"))

    print(f"[Data] raw total count={len(all_records)}")
    print(f"[Data] split train/val/test = {len(train_records)}/{len(val_records)}/{len(test_records)}")
    print(f"[Data] split files saved to: {SPLIT_SAVE_DIR}")

    train_raw = Dataset.from_list(train_records)
    val_raw = Dataset.from_list(val_records)
    test_raw = Dataset.from_list(test_records)

    return train_raw, val_raw, test_raw


# =========================
# Message normalization
# =========================
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

    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"Message {i} is not a dict")
        role = str(m.get("role", "")).strip().lower()
        if not role:
            raise ValueError(f"Message {i} is missing role")
        content = normalize_content(m.get("content", ""))
        out.append({"role": role, "content": content})
    return out


# =========================
# chat template
# =========================
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


def build_prompt_and_answer(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message is not from assistant")

    prompt_messages = messages[:-1]
    answer_text = messages[-1]["content"]

    prompt_text = apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )
    return prompt_text, answer_text


def format_sample_for_sft(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    messages = normalize_messages(sample.get("messages", []))
    prompt_text, answer_text = build_prompt_and_answer(tokenizer, messages)
    return {
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "messages": messages,
    }


# =========================
# Tokenization + label masking (train on assistant tokens only)
# =========================
def tokenize_and_mask(examples, tokenizer: AutoTokenizer):
    prompts = examples["prompt_text"]
    answers = examples["answer_text"]
    messages_col = examples["messages"]

    input_ids_list, attention_mask_list, labels_list, messages_list = [], [], [], []

    eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    for idx, (prompt_text, answer_text, messages) in enumerate(zip(prompts, answers, messages_col)):
        enc_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            padding=False,
            add_special_tokens=False,
        )
        prompt_ids = enc_prompt["input_ids"]

        answer_with_eos = answer_text + eos_text
        enc_answer = tokenizer(
            answer_with_eos,
            truncation=True,
            max_length=MAX_ANSWER_TOKENS,
            padding=False,
            add_special_tokens=False,
        )
        answer_ids = enc_answer["input_ids"]

        if len(answer_ids) == 0:
            raise ValueError(f"[Data] answer_ids is empty; sample messages={messages}")

        total_len = len(prompt_ids) + len(answer_ids)
        if total_len > MAX_SEQ_LEN:
            overflow = total_len - MAX_SEQ_LEN
            if overflow >= len(prompt_ids):
                prompt_ids = []
                answer_ids = answer_ids[:MAX_SEQ_LEN]
            else:
                prompt_ids = prompt_ids[overflow:]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + answer_ids[:]

        valid_label_count = sum(1 for x in labels if x != -100)
        if valid_label_count == 0:
            raise ValueError(f"[Data] The number of valid labels is 0; sample messages={messages}")

        raw_prompt_ids = tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        raw_answer_ids = tokenizer(
            answer_text + eos_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        if idx < DEBUG_PRINT_FIRST_N_SAMPLES:
            print(
                f"[DEBUG] sample={idx}, "
                f"raw_prompt_len={len(raw_prompt_ids)}, "
                f"raw_answer_len={len(raw_answer_ids)}, "
                f"prompt_len={len(prompt_ids)}, "
                f"answer_len={len(answer_ids)}, "
                f"full_len={len(input_ids)}, "
                f"valid_labels={valid_label_count}"
            )

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        messages_list.append(messages)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "messages": messages_list,
    }


# =========================
# Custom collator
# =========================
class DataCollatorForCausalLMWithLabels:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            attn = f["attention_mask"]
            lab = f["labels"]

            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(attn + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =========================
# Environment utilities
# =========================
def pick_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def has_flash_attn2() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def pick_attention_impl() -> str:
    return "flash_attention_2" if has_flash_attn2() else "sdpa"


def build_fsdp_config() -> Dict[str, Any]:
    return {
        "limit_all_gathers": True,
        "activation_checkpointing": True,
        "use_orig_params": True,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": True,
    }


# =========================
# Base model + tokenizer
# =========================
def load_base_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print("[Init] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = pick_compute_dtype()
    attn_impl = pick_attention_impl()

    print(f"[Init] compute_dtype={compute_dtype}")
    print(f"[Init] attn_implementation={attn_impl}")
    print("[Init] Loading full model for full fine-tuning...")

    load_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=compute_dtype,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=compute_dtype,
            **load_kwargs,
        )

    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("[Init] gradient checkpointing enabled.")
        except Exception as e:
            print(f"[Init] gradient checkpointing enable failed: {e}")

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Init] Trainable params: {trainable_params:,}")
    print(f"[Init] Total params    : {total_params:,}")
    print(f"[Init] Trainable %     : {100 * trainable_params / total_params:.4f}%")

    return model, tokenizer


# =========================
# Dataset loading
# =========================
def load_coupled_datasets(tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    train_raw, val_raw, test_raw = build_split_datasets()

    def fmt_fn(sample):
        return format_sample_for_sft(sample, tokenizer)

    train_fmt = train_raw.map(fmt_fn, remove_columns=train_raw.column_names, num_proc=1)
    val_fmt = val_raw.map(fmt_fn, remove_columns=val_raw.column_names, num_proc=1)
    test_fmt = test_raw.map(fmt_fn, remove_columns=test_raw.column_names, num_proc=1)

    def tok_fn(examples):
        return tokenize_and_mask(examples, tokenizer)

    train_tok = train_fmt.map(
        tok_fn,
        batched=True,
        batch_size=2,
        num_proc=1,
        remove_columns=["prompt_text", "answer_text"]
    )
    val_tok = val_fmt.map(
        tok_fn,
        batched=True,
        batch_size=2,
        num_proc=1,
        remove_columns=["prompt_text", "answer_text"]
    )
    test_tok = test_fmt.map(
        tok_fn,
        batched=True,
        batch_size=2,
        num_proc=1,
        remove_columns=["prompt_text", "answer_text"]
    )

    return train_tok, val_tok, test_tok


# =========================
# Save Trainer logs and plot curves
# =========================
def save_trainer_log_and_plot(trainer: Trainer):
    log_history = trainer.state.log_history
    with open(TRAIN_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)
    print(f"[Log] Saved trainer log_history to: {TRAIN_LOG_JSON}")

    steps_train, loss_train = [], []
    steps_eval, loss_eval = [], []
    for item in log_history:
        if "loss" in item and "step" in item:
            steps_train.append(item["step"])
            loss_train.append(item["loss"])
        if "eval_loss" in item and "step" in item:
            steps_eval.append(item["step"])
            loss_eval.append(item["eval_loss"])

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        if steps_train:
            plt.plot(steps_train, loss_train, label="train_loss")
        if steps_eval:
            plt.plot(steps_eval, loss_eval, label="eval_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training / Eval Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(TRAIN_CURVE_PNG, dpi=200)
        plt.close()
        print(f"[Plot] Saved loss curve to: {TRAIN_CURVE_PNG}")
    except Exception as e:
        print(f"[Plot] Failed to plot curve: {e}")


# =========================
# Inference testing
# =========================
def extract_structured_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end + 1]
        cand2 = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            return json.loads(cand2)
        except Exception:
            return {}
    return {}


def extract_messages(sample: Dict[str, Any]) -> Tuple[str, str, str]:
    msgs = sample.get("messages", [])
    system_text, user_text, assistant_text = "", "", ""

    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        content = normalize_content(m.get("content", ""))
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    return system_text.strip(), user_text.strip(), assistant_text.strip()


def build_infer_prompt_from_sample(sample: Dict[str, Any], tokenizer) -> str:
    messages = normalize_messages(sample.get("messages", []))
    prompt_messages = [m for m in messages if m["role"] in ["system", "user"]]
    return apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_once(model, tokenizer, prompt: str) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
        padding=False,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def evaluate_and_save_results(model, tokenizer, test_samples: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH) or ".", exist_ok=True)

    json_ok = 0
    with open(RESULTS_SAVE_PATH, "w", encoding="utf-8") as f:
        for i, s in enumerate(test_samples):
            prompt = build_infer_prompt_from_sample(s, tokenizer)
            gen_txt = generate_once(model, tokenizer, prompt)

            if PRINT_FULL_OUTPUT:
                print("\n" + "=" * 100)
                print(f"[RAW OUTPUT] Test Sample {i}")
                print("=" * 100)
                print(gen_txt)
                print("=" * 100)

            pred_obj = extract_structured_json(gen_txt)
            if pred_obj:
                json_ok += 1

            _, _, gt = extract_messages(s)
            gt_obj = extract_structured_json(gt) if isinstance(gt, str) else {}

            rec = {
                "idx": i,
                "pred_json_parse_ok": bool(pred_obj),
                "pred_json": pred_obj,
                "gt_json": gt_obj,
                "raw_generation": gen_txt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n[Test Summary]")
    print(f"  JSON parse ok rate: {json_ok}/{len(test_samples)} = {json_ok / max(1, len(test_samples)):.2%}")
    print(f"  results saved to: {RESULTS_SAVE_PATH}")


# =========================
# Training + saving + reloading + testing
# =========================
def train_then_test():
    set_seed(SEED)

    print(f"[Env] torch={torch.__version__}")
    try:
        import transformers
        print(f"[Env] transformers={transformers.__version__}")
    except Exception:
        pass

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"[Env] cuda available={torch.cuda.is_available()}")
    print(f"[Env] WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    if torch.cuda.is_available():
        print(f"[Env] GPU count visible={torch.cuda.device_count()}")
        idx = local_rank if local_rank < torch.cuda.device_count() else 0
        print(f"[Env] Current GPU={torch.cuda.get_device_name(idx)}")
        print(f"[Env] VRAM={torch.cuda.get_device_properties(idx).total_memory / 1e9:.2f} GB")
        print(f"[Env] BF16={torch.cuda.is_bf16_supported()}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(FULL_MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)

    run_config = {
        "MODEL_PATH": MODEL_PATH,
        "DATASET_FILE": DATASET_FILE,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "SPLIT_SAVE_DIR": SPLIT_SAVE_DIR,
        "FULL_MODEL_SAVE_PATH": FULL_MODEL_SAVE_PATH,
        "RESULTS_SAVE_PATH": RESULTS_SAVE_PATH,
        "TRAIN_LOG_JSON": TRAIN_LOG_JSON,
        "TRAIN_CURVE_PNG": TRAIN_CURVE_PNG,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "SPLIT_SEED": SPLIT_SEED,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS,
        "MAX_ANSWER_TOKENS": MAX_ANSWER_TOKENS,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "SEED": SEED,
        "USE_FSDP_IF_MULTI_GPU": USE_FSDP_IF_MULTI_GPU,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS,
        "PER_DEVICE_TRAIN_BATCH_SIZE": PER_DEVICE_TRAIN_BATCH_SIZE,
        "PER_DEVICE_EVAL_BATCH_SIZE": PER_DEVICE_EVAL_BATCH_SIZE,
        "GRADIENT_ACCUMULATION_STEPS": GRADIENT_ACCUMULATION_STEPS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
    }
    with open(RUN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    model, tokenizer = load_base_model_and_tokenizer()

    train_dataset, val_dataset, test_dataset = load_coupled_datasets(tokenizer)
    print(f"[Data] train size={len(train_dataset)}, val size={len(val_dataset)}, test size={len(test_dataset)}")

    for inspect_idx in range(min(3, len(train_dataset))):
        labels = train_dataset[inspect_idx]["labels"]
        valid_count = sum(1 for x in labels if x != -100)
        print(f"[CHECK] train_dataset[{inspect_idx}] valid_labels={valid_count}, total_len={len(labels)}")

    monitor = TrainingMonitor()
    compute_dtype = pick_compute_dtype()
    has_bf16 = compute_dtype == torch.bfloat16
    has_fp16 = compute_dtype == torch.float16

    use_fsdp = USE_FSDP_IF_MULTI_GPU and world_size > 1
    print(f"[Init] use_fsdp={use_fsdp}")

    training_args_raw = dict(
        output_dir=FULL_MODEL_SAVE_PATH,

        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,

        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,

        optim="adamw_torch",
        logging_steps=10,
        report_to="none",

        fp16=has_fp16,
        bf16=has_bf16,

        gradient_checkpointing=True,
        max_grad_norm=1.0,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=0,
        remove_unused_columns=False,
        save_safetensors=True,
        ddp_find_unused_parameters=False,
    )

    if use_fsdp:
        training_args_raw["fsdp"] = "full_shard"
        training_args_raw["fsdp_config"] = build_fsdp_config()

    training_args = TrainingArguments(**make_training_args_kwargs(**training_args_raw))
    print(f"[DEBUG] final fsdp = {getattr(training_args, 'fsdp', None)}")
    print(f"[DEBUG] final fsdp_config = {getattr(training_args, 'fsdp_config', None)}")

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[monitor],
    )

    if torch.cuda.is_available():
        print(f"[Train] VRAM before train: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("[Train] Start full fine-tuning...")
    train_result = trainer.train()

    print("[Train] Saving full fine-tuned model/tokenizer...")
    trainer.save_model(FULL_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(FULL_MODEL_SAVE_PATH)
    monitor.save_to_csv(prefix="energy_fullft_metrics")
    print(f"[Train] Saved to: {FULL_MODEL_SAVE_PATH}")

    save_trainer_log_and_plot(trainer)

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    train_metrics["val_samples"] = len(val_dataset)
    train_metrics["test_samples"] = len(test_dataset)
    with open(TRAIN_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    print("[Eval] Evaluating on validation/test loss...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    with open(VAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, ensure_ascii=False, indent=2)

    with open(TEST_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Reload] Loading full fine-tuned model for inference...")
    infer_load_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=pick_attention_impl(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        infer_model = AutoModelForCausalLM.from_pretrained(
            FULL_MODEL_SAVE_PATH,
            dtype=pick_compute_dtype(),
            **infer_load_kwargs,
        )
    except TypeError:
        infer_model = AutoModelForCausalLM.from_pretrained(
            FULL_MODEL_SAVE_PATH,
            torch_dtype=pick_compute_dtype(),
            **infer_load_kwargs,
        )

    infer_model.eval()

    print("[Test] Loading split test samples (jsonl)...")
    test_samples = load_jsonl(os.path.join(SPLIT_SAVE_DIR, "test.jsonl"))

    evaluate_and_save_results(infer_model, tokenizer, test_samples)
    print("[Done] Full fine-tuning + Test finished.")


def main():
    train_then_test()


if __name__ == "__main__":
    freeze_support()
    main()