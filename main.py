from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from e1_couple_state_assembler import JointStateAssembler
from g_lma_pewm import LMAPEWorldModel
from h_world_model_training import (
    FixedLLMSemanticExtractor,
    SemanticCacheConfig,
    WorldModelTrainer,
    WorldModelTrainingConfig,
    build_model_config,
    evaluate_checkpoint,
    load_semantic_records,
    prepare_semantic_records,
)


THIS_DIR = Path(__file__).resolve().parent
DATA_PROCESS_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROD_MODEL_PATH = str(
    DATA_PROCESS_ROOT / "sft_datasets" / "steel-production" / "outputs" / "qwen35_4b_steel_full_sft"
)
DEFAULT_ENERGY_MODEL_PATH = str(
    DATA_PROCESS_ROOT / "sft_datasets" / "energy" / "energy_fullt_outputs" / "full_finetuned_model"
)
DEFAULT_COUPLE_MODEL_PATH = str(
    DATA_PROCESS_ROOT / "sft_datasets" / "energy-production" / "qwen35_4b_fullt_outputs" / "full_finetuned_model"
)
DEFAULT_ARTIFACT_DIR = THIS_DIR / "world_model_artifacts"


def parse_days(text: str) -> List[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first-stage LMA-PEWM world model with fixed task-specific LLMs.")
    parser.add_argument("--mode", choices=["pipeline", "prepare", "train", "evaluate"], default="pipeline")
    parser.add_argument("--days", type=str, default="", help="Comma-separated day ids. Empty uses all common production-energy days.")
    parser.add_argument("--history_window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_samples_per_day", type=int, default=0, help="0 means all valid samples.")

    parser.add_argument("--prod_model", type=str, default=DEFAULT_PROD_MODEL_PATH)
    parser.add_argument("--energy_model", type=str, default=DEFAULT_ENERGY_MODEL_PATH)
    parser.add_argument("--couple_model", type=str, default=DEFAULT_COUPLE_MODEL_PATH)
    parser.add_argument("--llm_device", type=str, default="auto")
    parser.add_argument("--prod_tokens", type=int, default=8192)
    parser.add_argument("--energy_tokens", type=int, default=8192)
    parser.add_argument("--couple_tokens", type=int, default=8192)
    parser.add_argument("--no_llm_semantics", action="store_true", help="Debug mode: use structured adapter inputs instead of fixed LLM semantic outputs.")

    parser.add_argument("--artifact_dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--semantic_cache", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--semantic_dim", type=int, default=64)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_assembler() -> JointStateAssembler:
    assembler = JointStateAssembler(
        production_daily_objects_path=str(THIS_DIR / "production_daily_objects.json"),
        production_optimization_results_path=None,
        energy_data_path=str(THIS_DIR / "data_of_IES.xlsx"),
        heat_json_path=str(THIS_DIR / "heat_segments.json"),
        heat_excel_path=str(THIS_DIR / "heat_segments.xlsx"),
        patch_metadata_json_path=str(THIS_DIR / "patch_metadata.json"),
        patch_metadata_excel_path=str(THIS_DIR / "patch_metadata.xlsx"),
        patches_dir=str(THIS_DIR / "patches"),
        include_legends_in_meta=False,
    )
    assembler.load(sample_size=None)
    return assembler


def resolve_days(assembler: JointStateAssembler, requested: str) -> List[str]:
    days = parse_days(requested)
    if days:
        return days
    if hasattr(assembler, "get_common_day_ids"):
        days = assembler.get_common_day_ids()
    else:
        prod_days = list(getattr(assembler.production_adapter, "day_to_obj", {}).keys())
        energy_days = set(getattr(assembler.energy_adapter, "day_to_heats", {}).keys())
        days = [str(x) for x in prod_days if x in energy_days]
    if not days:
        raise RuntimeError("No common production-energy day ids were found.")
    return days


def enforce_stage1_world_model_policy() -> None:
    os.environ["PROD_FORECAST_NO_TARGET"] = "1"
    os.environ["PROD_USE_LABEL_GUIDED_INFERENCE"] = "0"
    os.environ["PROD_USE_SCHEDULE_GUIDED_INFERENCE"] = "0"
    os.environ["LLM_ENABLE_THINKING"] = "0"
    os.environ["LLM_FORCE_JSON_RESPONSE"] = "1"


def prepare_cache(args: argparse.Namespace, assembler: JointStateAssembler, days: List[str], cache_path: Path) -> List[dict]:
    extractor: Optional[FixedLLMSemanticExtractor] = None
    use_llm_semantics = not bool(args.no_llm_semantics)
    if use_llm_semantics:
        for model_path in [args.prod_model, args.energy_model, args.couple_model]:
            if not Path(model_path).expanduser().exists():
                raise FileNotFoundError(f"Local fine-tuned model path does not exist: {model_path}")
        extractor = FixedLLMSemanticExtractor(
            prod_model_path=args.prod_model,
            energy_model_path=args.energy_model,
            couple_model_path=args.couple_model,
            device=args.llm_device,
            prod_tokens=args.prod_tokens,
            energy_tokens=args.energy_tokens,
            couple_tokens=args.couple_tokens,
            auto_release=True,
        )

    cfg = SemanticCacheConfig(
        history_window=args.history_window,
        horizon=args.horizon,
        stride=args.stride,
        max_samples_per_day=(None if args.max_samples_per_day <= 0 else args.max_samples_per_day),
        use_llm_semantics=use_llm_semantics,
        stop_on_error=False,
    )
    records = prepare_semantic_records(
        assembler=assembler,
        day_ids=days,
        cache_path=str(cache_path),
        semantic_extractor=extractor,
        config=cfg,
    )
    return records


def train_world_model(args: argparse.Namespace, records: List[dict], artifact_dir: Path) -> dict:
    train_cfg = WorldModelTrainingConfig(
        history_window=args.history_window,
        horizon=args.horizon,
        semantic_dim=args.semantic_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_attention_heads=args.attention_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        device=args.train_device,
        num_workers=args.num_workers,
        patience=args.patience,
        checkpoint_dir=str(artifact_dir),
    )
    model = LMAPEWorldModel(build_model_config(train_cfg))
    trainer = WorldModelTrainer(model=model, config=train_cfg)
    summary = trainer.fit(records)
    config_path = artifact_dir / "stage1_training_config.json"
    config_path.write_text(
        json.dumps(
            {
                "world_model_training": asdict(train_cfg),
                "llm_paths": {
                    "production": args.prod_model,
                    "energy": args.energy_model,
                    "coupling": args.couple_model,
                },
                "history_window": args.history_window,
                "horizon": args.horizon,
                "stride": args.stride,
                "uses_fixed_llm_semantics": not bool(args.no_llm_semantics),
                "target_leakage_policy": "future targets are used only as supervised world-model labels and are not included in LLM semantic inputs",
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    enforce_stage1_world_model_policy()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.semantic_cache).expanduser().resolve() if args.semantic_cache else artifact_dir / "semantic_world_model_records.jsonl"
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else artifact_dir / "lma_pewm_best.pt"

    if args.mode == "evaluate":
        records = load_semantic_records(str(cache_path))
        metrics = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            records=records,
            device=args.train_device,
            batch_size=args.batch_size,
        )
        out_path = artifact_dir / "world_model_evaluation.json"
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    assembler = build_assembler()
    days = resolve_days(assembler, args.days)
    print(f"Stage-1 world-model days: {days}")
    print(f"Semantic cache: {cache_path}")
    print(f"Artifact directory: {artifact_dir}")

    records: List[dict]
    if args.mode in {"prepare", "pipeline"}:
        records = prepare_cache(args, assembler, days, cache_path)
        print(f"Prepared semantic records: {len(records)}")
        if args.mode == "prepare":
            return
    else:
        records = load_semantic_records(str(cache_path))

    summary = train_world_model(args, records, artifact_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
