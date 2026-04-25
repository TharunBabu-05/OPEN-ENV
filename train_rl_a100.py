"""
train_rl_a100.py — A100-Optimized GRPO Training with Curriculum Learning

Designed for NVIDIA A100 80GB. Features:
- BF16 precision
- Curriculum learning (easy -> medium -> hard)
- Expanded dataset (350 samples)
- Enhanced reward signals
- GPU utilization monitoring
- Checkpoint separation (preserves old models)
"""

import json
import logging
import os
import sys
import time
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builder (expanded: 350 samples with hybrid strategy)
# ---------------------------------------------------------------------------

def build_expanded_dataset(cfg: Dict) -> list:
    """Build 350+ training samples using heuristic, random, and adversarial strategies."""
    sys.path.insert(0, str(Path(__file__).parent))
    from env import ESGEnvironment
    from tasks import TASKS

    curriculum = cfg.get("curriculum", [])
    strategy = cfg.get("dataset_strategy", {"heuristic_pct": 60, "random_pct": 25, "adversarial_pct": 15})
    target_size = cfg.get("dataset_size", 350)

    all_task_ids = []
    episodes_per_task = {}
    for stage in curriculum:
        for tid in stage["task_ids"]:
            all_task_ids.append(tid)
            episodes_per_task[tid] = stage.get("n_episodes_per_task", 40)

    if not all_task_ids:
        all_task_ids = ["basic_compliance"]
        episodes_per_task = {"basic_compliance": 40}

    samples = []
    seen_states = set()

    for task_id in all_task_ids:
        task_config = TASKS[task_id]
        n_episodes = episodes_per_task.get(task_id, 40)

        # Heuristic rollouts (60%)
        n_heuristic = int(n_episodes * strategy.get("heuristic_pct", 60) / 100)
        for ep in range(n_heuristic):
            seed = 100 + ep
            env = ESGEnvironment(task_config=task_config, seed=seed)
            obs = env.reset()
            for step in range(task_config.max_steps):
                # Smart heuristic: priority-based action selection
                action = _heuristic_action(obs, task_config)
                obs_dict = obs.model_dump()
                obs_dict["_seed"] = seed

                prompt = _build_prompt(obs, task_id)
                state_key = f"{task_id}_{seed}_{step}"
                if state_key not in seen_states:
                    seen_states.add(state_key)
                    samples.append({
                        "prompt": prompt,
                        "obs_snapshot": obs_dict,
                        "task_id": task_id,
                        "seed": seed,
                        "step": step,
                        "strategy": "heuristic",
                    })

                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break

        # Random exploration (25%)
        n_random = int(n_episodes * strategy.get("random_pct", 25) / 100)
        for ep in range(n_random):
            seed = 1000 + ep
            env = ESGEnvironment(task_config=task_config, seed=seed)
            obs = env.reset()
            for step in range(task_config.max_steps):
                action = random.randint(0, 8)
                obs_dict = obs.model_dump()
                obs_dict["_seed"] = seed

                prompt = _build_prompt(obs, task_id)
                state_key = f"{task_id}_{seed}_{step}"
                if state_key not in seen_states:
                    seen_states.add(state_key)
                    samples.append({
                        "prompt": prompt,
                        "obs_snapshot": obs_dict,
                        "task_id": task_id,
                        "seed": seed,
                        "step": step,
                        "strategy": "random",
                    })

                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break

        # Adversarial edge cases (15%)
        n_adversarial = int(n_episodes * strategy.get("adversarial_pct", 15) / 100)
        for ep in range(n_adversarial):
            seed = 5000 + ep
            env = ESGEnvironment(task_config=task_config, seed=seed)
            obs = env.reset()
            # Force edge-case sequences: all same action, budget drain, NO_ACTION spam
            edge_patterns = [
                [0, 0, 0, 0, 0, 0],       # Solar spam
                [8, 8, 8, 8, 8, 8],       # NO_ACTION spam
                [0, 1, 0, 1, 0, 1],       # Expensive oscillation
                [7, 7, 7, 2, 2, 2],       # Cheap action spam
                [0, 5, 3, 1, 4, 6],       # Diverse but expensive
                [2, 6, 7, 8, 8, 8],       # Start cheap, then idle
            ]
            pattern = edge_patterns[ep % len(edge_patterns)]

            for step, action in enumerate(pattern[:task_config.max_steps]):
                obs_dict = obs.model_dump()
                obs_dict["_seed"] = seed

                prompt = _build_prompt(obs, task_id)
                state_key = f"{task_id}_{seed}_{step}"
                if state_key not in seen_states:
                    seen_states.add(state_key)
                    samples.append({
                        "prompt": prompt,
                        "obs_snapshot": obs_dict,
                        "task_id": task_id,
                        "seed": seed,
                        "step": step,
                        "strategy": "adversarial",
                    })

                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break

    random.shuffle(samples)
    log.info(f"Dataset built: {len(samples)} samples across {len(all_task_ids)} tasks")
    log.info(f"  Strategy breakdown: heuristic={sum(1 for s in samples if s['strategy']=='heuristic')}, "
             f"random={sum(1 for s in samples if s['strategy']=='random')}, "
             f"adversarial={sum(1 for s in samples if s['strategy']=='adversarial')}")
    return samples


def _heuristic_action(obs, task_config) -> int:
    """Priority-based heuristic: spend budget on most-needed improvement."""
    from models import Action
    budget = obs.available_budget

    priorities = []
    if obs.renewable_energy_pct < 60 and budget >= 150_000:
        priorities.append((0, 3))  # Solar: high priority
    if obs.carbon_emissions_tons > 500 and budget >= 40_000:
        priorities.append((4, 2))  # Carbon offset
    if obs.waste_recycled_pct < 60 and budget >= 25_000:
        priorities.append((2, 2))  # Recycling
    if obs.diversity_score < 70 and budget >= 50_000:
        priorities.append((5, 1))  # Diversity
    if budget >= 80_000:
        priorities.append((1, 1))  # HVAC
    if budget >= 60_000:
        priorities.append((3, 1))  # Water
    if budget >= 30_000:
        priorities.append((6, 1))  # Wellness
    if budget >= 15_000:
        priorities.append((7, 1))  # Audit

    if priorities:
        priorities.sort(key=lambda x: -x[1])
        return priorities[0][0]
    return 8  # No action


def _build_prompt(obs, task_id: str) -> str:
    """Build the LLM prompt from observation."""
    return (
        f"You are an ESG sustainability strategist. Task: {task_id}\n\n"
        f"Current State:\n"
        f"- Carbon Emissions: {obs.carbon_emissions_tons:.0f} tons\n"
        f"- Renewable Energy: {obs.renewable_energy_pct:.1f}%\n"
        f"- Waste Recycled: {obs.waste_recycled_pct:.1f}%\n"
        f"- Water Usage: {obs.water_usage_cubic_m:.0f} m3\n"
        f"- Diversity Score: {obs.diversity_score:.1f}\n"
        f"- Employee Satisfaction: {obs.employee_satisfaction:.1f}\n"
        f"- Budget: ${obs.available_budget:,.0f}\n"
        f"- Month: {obs.current_month}\n"
        f"- Compliance Violations: {obs.compliance_violations}\n\n"
        f"Actions: 0=Solar, 1=HVAC, 2=Recycle, 3=Water, 4=CarbonOffset, "
        f"5=Diversity, 6=Wellness, 7=Audit, 8=NoAction\n\n"
        f'Respond with JSON: {{"action": <int>, "reasoning": "<text>"}}'
    )


# ---------------------------------------------------------------------------
# Enhanced reward function with waste boost and synergy
# ---------------------------------------------------------------------------

def make_a100_reward_fn(cfg: Dict):
    """Enhanced reward function with waste boost, synergy, and normalization."""
    from reward_functions import (
        reward_env_outcome,
        reward_format_compliance,
        reward_anti_cheat,
        reward_task_progress,
    )

    weights = cfg.get("reward_weights", {})
    w_env = weights.get("env_outcome", 0.55)
    w_fmt = weights.get("format_compliance", 0.15)
    w_cheat = weights.get("anti_cheat", 0.10)
    w_prog = weights.get("task_progress", 0.10)
    w_waste = weights.get("waste_boost", 0.05)
    w_synergy = weights.get("synergy_bonus", 0.05)

    def _normalize_completion(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict):
                    parts.append(item.get("content", str(item)))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(c)

    def _waste_reward(completions, obs_snapshots):
        """Extra reward for waste recycling improvements."""
        rewards = []
        for obs_dict in obs_snapshots:
            pct = obs_dict.get("waste_recycled_pct", 0)
            if pct >= 70:
                rewards.append(1.0)
            elif pct >= 50:
                rewards.append(0.5)
            elif pct >= 30:
                rewards.append(0.2)
            else:
                rewards.append(-0.2)
        return rewards

    def _synergy_reward(completions, obs_snapshots):
        """Bonus when multiple ESG dimensions improve simultaneously."""
        rewards = []
        for obs_dict in obs_snapshots:
            score = 0.0
            if obs_dict.get("renewable_energy_pct", 0) > 30:
                score += 0.3
            if obs_dict.get("carbon_emissions_tons", 9999) < 800:
                score += 0.3
            if obs_dict.get("waste_recycled_pct", 0) > 40:
                score += 0.3
            if obs_dict.get("diversity_score", 0) > 60:
                score += 0.1
            # Synergy bonus only if 3+ dimensions are good
            rewards.append(score if score >= 0.6 else 0.0)
        return rewards

    def enhanced_reward_fn(completions, **batch):
        completions = [_normalize_completion(c) for c in completions]
        obs_snapshots = batch.get("obs_snapshot", [{}] * len(completions))
        task_ids = batch.get("task_id", ["basic_compliance"] * len(completions))

        for obs, seed in zip(obs_snapshots, batch.get("seed", [42] * len(completions))):
            if isinstance(obs, dict):
                obs["_seed"] = seed

        r_env = reward_env_outcome(completions, obs_snapshots, task_ids)
        r_fmt = reward_format_compliance(completions)
        r_cheat = reward_anti_cheat(completions, obs_snapshots)
        r_prog = reward_task_progress(completions, obs_snapshots, task_ids)
        r_waste = _waste_reward(completions, obs_snapshots)
        r_synergy = _synergy_reward(completions, obs_snapshots)

        combined = []
        for i in range(len(completions)):
            total = (w_env * r_env[i] + w_fmt * r_fmt[i] + w_cheat * r_cheat[i] +
                     w_prog * r_prog[i] + w_waste * r_waste[i] + w_synergy * r_synergy[i])
            # Normalize to [-1, 1]
            total = max(-1.0, min(1.0, total))
            combined.append(float(total))

        avg = sum(combined) / len(combined) if combined else 0.0
        log.debug(f"Rewards -- mean: {avg:.3f}")
        return combined

    return enhanced_reward_fn


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

def log_gpu_stats(step: int):
    """Log GPU utilization and memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
            util_pct = (mem_used / mem_total) * 100
            log.info(f"[Step {step}] GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({util_pct:.0f}%)")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main training function with curriculum
# ---------------------------------------------------------------------------

def train_a100(cfg: Dict):
    """A100-optimized GRPO training with curriculum learning."""
    import torch
    from datasets import Dataset

    log.info("=" * 60)
    log.info("ESG RL TRAINING -- A100 GRPO WITH CURRICULUM")
    log.info("=" * 60)
    log.info(f"Config:\n{json.dumps(cfg, indent=2, default=str)}")

    # 1. Build expanded dataset
    log.info("Building expanded dataset...")
    samples = build_expanded_dataset(cfg)
    log.info(f"Total samples: {len(samples)}")

    # 2. Load model
    log.info(f"Loading model: {cfg['base_model']}")
    if cfg.get("use_unsloth", True) and not cfg.get("cpu_only", False):
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model"],
            max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"],
            dtype=torch.bfloat16 if cfg.get("precision") == "bf16" else torch.float16,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        log.info("Model loaded with Unsloth + LoRA (A100 mode)")
    else:
        raise RuntimeError("A100 training requires GPU + Unsloth")

    # 3. Curriculum training loop
    from trl import GRPOConfig, GRPOTrainer
    import inspect

    curriculum = cfg.get("curriculum", [{"stage": 1, "task_ids": ["basic_compliance"], "steps": cfg["max_steps"]}])
    global_step = 0
    reward_history = []

    for stage_cfg in curriculum:
        stage_num = stage_cfg["stage"]
        stage_tasks = stage_cfg["task_ids"]
        stage_steps = stage_cfg["steps"]

        log.info(f"\n{'='*60}")
        log.info(f"CURRICULUM STAGE {stage_num}: {stage_tasks} ({stage_steps} steps)")
        log.info(f"{'='*60}")

        # Filter dataset for this stage's tasks
        stage_samples = [s for s in samples if s["task_id"] in stage_tasks]
        if not stage_samples:
            log.warning(f"No samples for stage {stage_num}, skipping")
            continue

        dataset = Dataset.from_list([{
            "prompt": s["prompt"],
            "obs_snapshot": s["obs_snapshot"],
            "task_id": s["task_id"],
            "seed": s["seed"],
        } for s in stage_samples])

        # Build reward function
        reward_fn = make_a100_reward_fn(cfg)

        # Configure GRPO for this stage
        grpo_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
        use_bf16 = cfg.get("precision") == "bf16"

        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        grpo_kwargs = dict(
            output_dir=str(output_dir / f"stage_{stage_num}"),
            learning_rate=cfg["learning_rate"],
            num_train_epochs=1,
            max_steps=stage_steps,
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_generations=cfg["num_generations"],
            temperature=cfg["temperature"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            report_to="none",
            remove_unused_columns=False,
            seed=42,
            bf16=use_bf16,
            fp16=not use_bf16,
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
        )

        if "max_completion_length" in grpo_params:
            grpo_kwargs["max_completion_length"] = cfg["max_new_tokens"]
        elif "max_new_tokens" in grpo_params:
            grpo_kwargs["max_new_tokens"] = cfg["max_new_tokens"]

        if "log_completions" in grpo_params:
            grpo_kwargs["log_completions"] = True

        training_args = GRPOConfig(**grpo_kwargs)

        # Build trainer
        trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
        trainer_kwargs = dict(
            model=model,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=dataset,
        )
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        trainer = GRPOTrainer(**trainer_kwargs)

        # Train this stage
        log.info(f"Starting Stage {stage_num} training ({stage_steps} steps)...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time
        log.info(f"Stage {stage_num} complete in {elapsed:.0f}s")

        # Log GPU stats
        log_gpu_stats(global_step + stage_steps)
        global_step += stage_steps

    # 4. Save final model
    log.info("Saving final model...")
    output_dir = Path(cfg["output_dir"])
    lora_path = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    log.info(f"LoRA adapter saved -> {lora_path}")

    # Save merged model
    try:
        merged_path = output_dir / "merged_model"
        model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
        log.info(f"Merged 16-bit model saved -> {merged_path}")
    except Exception as e:
        log.warning(f"Merged save failed (LoRA saved OK): {e}")

    # Save config
    with open(output_dir / "train_config_used.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    log.info(f"Config saved -> {output_dir / 'train_config_used.json'}")

    log.info("=" * 60)
    log.info("A100 TRAINING COMPLETE")
    log.info(f"Total steps: {global_step}")
    log.info(f"Output: {output_dir}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="A100 GRPO Training")
    parser.add_argument("--config", type=str, default="train_config_a100.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    train_a100(cfg)


if __name__ == "__main__":
    main()
