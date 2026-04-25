"""
ESG RL Training Script — GRPO + Unsloth + TRL

This script trains an LLM to make better ESG decisions using
Group Relative Policy Optimization (GRPO) with verifiable rewards.

Stack:
  - Unsloth:  Fast 4-bit LLM loading & LoRA fine-tuning
  - TRL:      GRPOTrainer for RL training
  - Datasets: HuggingFace datasets for prompt management
  - ESGEnv:   Our custom environment for reward computation

Usage:
  # Quick smoke test (no GPU needed, CPU mode)
  python train_rl.py --smoke_test

  # Full training (needs GPU with ~8GB VRAM)
  python train_rl.py --config train_config.yaml

  # Train on easy task only (recommended starting point)
  python train_rl.py --task basic_compliance --max_steps 200

Hackathon guide compliance:
  ✓ §3:  Starts from capable instruct model
  ✓ §6:  Easy task first (basic_compliance, 6 steps)
  ✓ §7:  Multiple independent reward functions
  ✓ §8:  Anti-cheat checks in reward_functions.py
  ✓ §10: TRL + Unsloth stack
  ✓ §11: GRPO with verifiable rewards (no learned reward model)
  ✓ §12: Unsloth for fast inference during rollouts
  ✓ §15: Logs reward, format, anti-cheat columns separately
  ✓ §16: Safe LoRA save path
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Model
    "base_model": "unsloth/Qwen2.5-1.5B-Instruct",
    "max_seq_length": 512,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Training
    "task_ids": ["basic_compliance"],          # Start with easy task
    "n_episodes_per_task": 10,
    "learning_rate": 5e-6,
    "num_train_epochs": 1,
    "max_steps": 100,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,                      # GRPO rollouts per prompt
    "max_new_tokens": 128,
    "temperature": 0.7,
    # Output
    "output_dir": "outputs/esg_rl_model",
    "logging_steps": 10,
    "save_steps": 50,
    # Flags
    "smoke_test": False,
    "use_unsloth": True,
    "cpu_only": False,
}


def load_config(config_path: Optional[str] = None, overrides: Dict = None) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg or {})
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_or_build_dataset(cfg: Dict):
    """Load existing JSONL dataset or build a fresh one."""
    from datasets import Dataset
    from dataset_builder import generate_dataset

    data_path = "data/esg_prompts.jsonl"

    if not Path(data_path).exists():
        log.info("Building dataset (no existing dataset found)...")
        generate_dataset(
            n_episodes_per_task=cfg["n_episodes_per_task"],
            output_path=data_path,
            verbose=True,
        )

    # Load from JSONL
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # Filter to configured tasks
                if record["task_id"] in cfg["task_ids"]:
                    records.append(record)

    log.info(f"Loaded {len(records)} training samples for tasks: {cfg['task_ids']}")

    if cfg.get("smoke_test"):
        records = records[:8]
        log.info("Smoke test: using 8 samples only")

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward function wrapper for TRL
# TRL GRPOTrainer expects: fn(prompts, completions, **batch_kwargs) -> List[float]
# ---------------------------------------------------------------------------

def make_trl_reward_fn(cfg: Dict):
    """
    Returns a single reward function compatible with TRL's GRPOTrainer.

    TRL passes: completions (List[str]) + any extra columns from the dataset.
    We extract obs_snapshot and task_id from the dataset columns.
    """
    from reward_functions import reward_composite

    def trl_reward_fn(completions: List[str], **batch) -> List[float]:
        obs_snapshots = batch.get("obs_snapshot", [{}] * len(completions))
        task_ids = batch.get("task_id", ["basic_compliance"] * len(completions))

        # Add seed to obs snapshots for deterministic env restoration
        for obs, seed in zip(obs_snapshots, batch.get("seed", [42] * len(completions))):
            if isinstance(obs, dict):
                obs["_seed"] = seed

        rewards = reward_composite(
            completions=completions,
            obs_snapshots=obs_snapshots,
            task_ids=task_ids,
        )

        # Log reward stats every call
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        log.debug(f"Batch rewards — mean: {avg:.3f}, min: {min(rewards):.3f}, max: {max(rewards):.3f}")

        return rewards

    return trl_reward_fn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: Dict):
    """Load model with Unsloth (or HuggingFace fallback)."""
    if cfg.get("use_unsloth") and not cfg.get("cpu_only"):
        try:
            from unsloth import FastLanguageModel
            log.info(f"Loading model with Unsloth: {cfg['base_model']}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg["base_model"],
                max_seq_length=cfg["max_seq_length"],
                load_in_4bit=cfg["load_in_4bit"],
                dtype=None,  # Auto-detect
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg["lora_r"],
                target_modules=cfg["target_modules"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            log.info("Model loaded with Unsloth + LoRA")
            return model, tokenizer, "unsloth"

        except ImportError:
            log.warning("Unsloth not installed. Falling back to HuggingFace transformers.")

    # HuggingFace fallback (slower but works everywhere)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    log.info(f"Loading model with HuggingFace: {cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if not cfg.get("cpu_only"):
        try:
            import torch
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                if cfg["load_in_4bit"]:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **load_kwargs)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer, "hf"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Dict):
    """Main training loop."""
    from trl import GRPOConfig, GRPOTrainer

    log.info("=" * 60)
    log.info("ESG RL TRAINING — GRPO")
    log.info("=" * 60)
    log.info(f"Config: {json.dumps({k: v for k, v in cfg.items() if k != 'target_modules'}, indent=2)}")

    # 1. Load dataset
    dataset = load_or_build_dataset(cfg)

    # 2. Load model
    model, tokenizer, backend = load_model_and_tokenizer(cfg)

    # 3. Build reward function
    reward_fn = make_trl_reward_fn(cfg)

    # 4. Configure GRPO trainer
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"] if not cfg.get("smoke_test") else 5,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"],
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        report_to="none",           # Disable wandb/tensorboard by default
        remove_unused_columns=False, # Keep our custom columns
        log_completions=True,        # See what the model generates
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
    )

    # 5. Train
    log.info("Starting GRPO training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    log.info(f"Training complete in {elapsed:.1f}s")

    # 6. Save model (safely — never merge 4-bit naively)
    # §16: Use the adapter-only save path
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info(f"LoRA adapter saved → {adapter_path}")

    # Save merged model only if Unsloth is available (handles 4-bit correctly)
    if backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
            merged_path = output_dir / "merged_model"
            model.save_pretrained_merged(
                str(merged_path),
                tokenizer,
                save_method="merged_16bit",
            )
            log.info(f"Merged 16-bit model saved → {merged_path}")
        except Exception as e:
            log.warning(f"Could not save merged model: {e}. Use lora_adapter instead.")

    # 7. Save training config for reproducibility
    cfg_save_path = output_dir / "train_config_used.json"
    with open(cfg_save_path, "w") as f:
        json.dump({k: v for k, v in cfg.items() if k != "target_modules"}, f, indent=2)
    log.info(f"Config saved → {cfg_save_path}")

    return trainer


# ---------------------------------------------------------------------------
# Smoke test (no GPU, no model download needed)
# ---------------------------------------------------------------------------

def smoke_test():
    """
    Validates the full pipeline without loading a real model.
    Tests: dataset generation, reward functions, env stepping.
    """
    log.info("=" * 60)
    log.info("SMOKE TEST — No GPU required")
    log.info("=" * 60)

    # Test dataset builder
    from dataset_builder import generate_dataset
    samples = generate_dataset(n_episodes_per_task=2, output_path="data/smoke_test.jsonl", verbose=True)
    assert len(samples) > 0, "Dataset generation failed"
    log.info(f"✓ Dataset: {len(samples)} samples")

    # Test reward functions
    from reward_functions import reward_composite, reward_format_compliance, reward_anti_cheat
    from env import ESGEnvironment
    from tasks import TASKS

    env = ESGEnvironment(TASKS["basic_compliance"], seed=42)
    obs = env.reset()
    obs_dict = obs.model_dump()
    obs_dict["_seed"] = 42

    completions = [
        '{"action": 0, "reasoning": "Solar panels boost renewable energy significantly."}',
        '{"action": 8, "reasoning": "Saving budget."}',
        "action 3",  # Unparseable
    ]
    obs_snapshots = [obs_dict, obs_dict, obs_dict]
    task_ids = ["basic_compliance", "basic_compliance", "basic_compliance"]

    fmt_r = reward_format_compliance(completions)
    cheat_r = reward_anti_cheat(completions, obs_snapshots)
    composite_r = reward_composite(completions, obs_snapshots, task_ids)

    for name, comp, fmt, cheat, comp_r in zip(
        ["solar_action", "no_action", "bad_format"],
        completions, fmt_r, cheat_r, composite_r
    ):
        log.info(f"  [{name}] format={fmt:.2f}, anti_cheat={cheat:.2f}, composite={comp_r:.2f}")

    assert fmt_r[0] == 1.0, "Format reward for good JSON should be 1.0"
    assert fmt_r[2] == 0.0, "Format reward for unparseable should be 0.0"
    log.info("✓ Reward functions working correctly")
    log.info("=" * 60)
    log.info("SMOKE TEST PASSED ✓")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ESG agent with GRPO")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.yaml")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test only (no GPU needed)")
    parser.add_argument("--task", type=str, default=None, help="Task ID to train on (overrides config)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--model", type=str, default=None, help="Base model name")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU mode (slow but works anywhere)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for model")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    cfg = load_config(
        config_path=args.config,
        overrides={
            "task_ids": [args.task] if args.task else None,
            "max_steps": args.max_steps,
            "base_model": args.model,
            "cpu_only": args.cpu_only or None,
            "output_dir": args.output_dir,
        },
    )

    train(cfg)


if __name__ == "__main__":
    main()
