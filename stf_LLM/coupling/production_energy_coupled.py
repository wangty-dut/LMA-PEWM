# -*- coding: utf-8 -*-
import os
import re
import csv
import json
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

# =========================
# Path configuration
# =========================
MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\Qwen\Qwen3.5-4B"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA_PATH = os.path.join(PROJECT_DIR, "production_energy_coupled_messages_train.jsonl")
TEST_DATA_PATH  = os.path.join(PROJECT_DIR, "production_energy_coupled_messages_test.jsonl")

OUTPUT_ROOT = os.path.join(PROJECT_DIR, "qwen35_4b_fullft_outputs")
FULL_MODEL_SAVE_PATH = os.path.join(OUTPUT_ROOT, "full_finetuned_model")
RESULTS_SAVE_PATH = os.path.join(OUTPUT_ROOT, "test_results.jsonl")
TRAIN_LOG_JSON = os.path.join(OUTPUT_ROOT, "training_log_history.json")
TRAIN_CURVE_PNG = os.path.join(OUTPUT_ROOT, "training_curve.png")

MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 5120
SEED = 42
PRINT_FULL_OUTPUT = False

# =========================
# Recommended strategy for full fine-tuning of the 4B model
# =========================
# Single GPU: use Trainer directly for full fine-tuning
# Multiple GPUs: use FSDP (more memory-efficient than standard DDP)
USE_FSDP_IF_MULTI_GPU = True

LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03


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

    def save_to_csv(self, prefix="production_energy_coupled_fullft_metrics"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(MODEL_PATH, f"{prefix}_{timestamp}.csv")
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
# JSONL loading
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


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
        role = str(m.get("role", "")).strip()
        if not role:
            raise ValueError(f"Message {i} is missing role")
        content = normalize_content(m.get("content", ""))
        out.append({"role": role, "content": content})
    return out


# =========================
# Chat template handling
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


def build_prompt_and_fulltext(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message is not from assistant")

    prompt_messages = messages[:-1]
    full_messages = messages

    prompt = apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )
    full_text = apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=full_messages,
        add_generation_prompt=False,
    )
    return prompt, full_text


def format_sample_for_sft(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    messages = normalize_messages(sample.get("messages", []))
    prompt, full_text = build_prompt_and_fulltext(tokenizer, messages)
    return {"text": full_text, "prompt": prompt}


# =========================
# Tokenization + label masking (train on assistant tokens only)
# =========================
def tokenize_and_mask(examples, tokenizer: AutoTokenizer):
    texts = examples["text"]
    prompts = examples["prompt"]

    input_ids_list, attention_mask_list, labels_list = [], [], []

    for text, prompt in zip(texts, prompts):
        enc_full = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]

        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
            add_special_tokens=False,
        )
        prompt_len = len(enc_prompt["input_ids"])
        prompt_len = min(prompt_len, len(input_ids))

        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attn)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
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
# Base model + tokenizer (full fine-tuning version)
# =========================
def load_base_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print("[Init] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

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
# Dataset loading (JSONL -> Dataset)
# =========================
def load_coupled_datasets(tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    print(f"[Data] Loading train jsonl: {TRAIN_DATA_PATH}")
    train_list = load_jsonl(TRAIN_DATA_PATH)

    print(f"[Data] Loading test  jsonl: {TEST_DATA_PATH}")
    test_list = load_jsonl(TEST_DATA_PATH)

    print(f"[Data] raw train count={len(train_list)}, raw test count={len(test_list)}")

    train_raw = Dataset.from_list(train_list)
    test_raw = Dataset.from_list(test_list)

    def fmt_fn(sample):
        return format_sample_for_sft(sample, tokenizer)

    train_fmt = train_raw.map(fmt_fn, remove_columns=train_raw.column_names, num_proc=1)
    test_fmt = test_raw.map(fmt_fn, remove_columns=test_raw.column_names, num_proc=1)

    def tok_fn(examples):
        return tokenize_and_mask(examples, tokenizer)

    train_tok = train_fmt.map(tok_fn, batched=True, batch_size=2, num_proc=1, remove_columns=["text", "prompt"])
    test_tok = test_fmt.map(tok_fn, batched=True, batch_size=2, num_proc=1, remove_columns=["text", "prompt"])

    return train_tok, test_tok


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
    prompt = apply_chat_template_safe(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )
    return prompt


@torch.no_grad()
def generate_once(model, tokenizer, prompt: str) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
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
        print(f"[Env] BF16={torch.cuda.is_bf16_supported()}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(FULL_MODEL_SAVE_PATH, exist_ok=True)

    model, tokenizer = load_base_model_and_tokenizer()
    train_dataset, eval_dataset = load_coupled_datasets(tokenizer)
    print(f"[Data] train size={len(train_dataset)}, eval(test) size={len(eval_dataset)}")

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
        training_args_raw["fsdp"] = "full_shard auto_wrap"
        training_args_raw["fsdp_config"] = build_fsdp_config()

    training_args = TrainingArguments(**make_training_args_kwargs(**training_args_raw))

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[monitor],
    )

    if torch.cuda.is_available():
        try:
            print(f"[Train] VRAM before train: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        except Exception:
            pass

    print("[Train] Start full fine-tuning...")
    trainer.train()

    print("[Train] Saving full fine-tuned model/tokenizer...")
    trainer.save_model(FULL_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(FULL_MODEL_SAVE_PATH)
    monitor.save_to_csv(prefix="production_energy_coupled_fullft_metrics")
    print(f"[Train] Saved to: {FULL_MODEL_SAVE_PATH}")

    save_trainer_log_and_plot(trainer)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Reload] Loading full fine-tuned model for inference...")
    infer_model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL_SAVE_PATH,
        torch_dtype=pick_compute_dtype(),
        trust_remote_code=True,
        attn_implementation=pick_attention_impl(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    infer_model.eval()

    print("[Test] Loading test samples (jsonl)...")
    test_samples = load_jsonl(TEST_DATA_PATH)

    evaluate_and_save_results(infer_model, tokenizer, test_samples)
    print("[Done] Full fine-tuning + Test finished.")


def main():
    train_then_test()


if __name__ == "__main__":
    freeze_support()
    main()