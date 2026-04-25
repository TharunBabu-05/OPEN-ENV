"""
Dataset builder for ESG RL training.

Generates a prompt dataset by rolling out the ESG environment with a
random/heuristic policy. Each sample contains:
  - prompt: system + observation text the LLM receives
  - task_id: which task was being run
  - obs_dict: serialized observation for reward computation
  - step: step index in the episode

This dataset is used as the "prompt pool" for GRPO training — the model
is asked to generate actions, which are then evaluated by the environment.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from env import ESGEnvironment
from models import Action, Observation
from tasks import TASKS


# ---------------------------------------------------------------------------
# Prompt construction (mirrors inference.py but standalone)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ESG (Environmental, Social, Governance) sustainability strategist.

Your goal is to optimize a company's ESG metrics by taking strategic actions each month.

Available Actions:
0. INSTALL_SOLAR_PANELS      - $150K upfront, boosts renewable energy over 6 months
1. UPGRADE_HVAC_EFFICIENCY   - $80K, reduces energy consumption over 12 months
2. IMPLEMENT_RECYCLING_PROGRAM - $25K, immediately increases waste recycling
3. INSTALL_WATER_RECYCLING   - $60K, reduces water usage over 12 months
4. CARBON_OFFSET_PURCHASE    - $40K, immediate carbon reduction (1 month)
5. DIVERSITY_HIRING_INITIATIVE - $50K, improves diversity score over 6 months
6. EMPLOYEE_WELLNESS_PROGRAM - $30K, improves employee satisfaction over 6 months
7. ENERGY_AUDIT              - $15K, small immediate energy reduction + ongoing benefit
8. NO_ACTION                 - $0, conserve budget for later

Key Principles:
- Solar panels compound over time — invest early for maximum benefit.
- Balance immediate impact (carbon offsets) vs. long-term ROI (solar, HVAC).
- Monitor budget: going bankrupt ends the episode with a penalty.
- Diversity and wellness actions matter for social scoring.
- Avoid repeating the same cheap action — reward hacking is penalized.

Respond with ONLY a valid JSON object:
{"action": <0-8>, "reasoning": "<brief explanation>"}"""


def build_obs_prompt(task_config: Dict[str, Any], obs: Observation, step: int) -> str:
    """Build the user-turn prompt from current observation."""
    carbon_pct = (
        (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
        / obs.baseline_carbon_emissions_tons * 100
        if obs.baseline_carbon_emissions_tons > 0 else 0.0
    )
    water_pct = (
        (obs.baseline_water_usage_cubic_m - obs.water_usage_cubic_m)
        / obs.baseline_water_usage_cubic_m * 100
        if obs.baseline_water_usage_cubic_m > 0 else 0.0
    )

    lines = [
        f"TASK: {task_config['task_id']} (difficulty={task_config['difficulty']})",
        f"Month {step + 1} of {task_config['max_steps']}",
        "",
        "TARGETS:",
        f"  Carbon reduction ≥ {task_config['target_carbon_reduction_pct']}%  (current: {carbon_pct:.1f}%)",
        f"  Renewable energy ≥ {task_config['target_renewable_pct']}%  (current: {obs.renewable_energy_pct:.1f}%)",
    ]
    if task_config.get("target_diversity_score", 0) > 0:
        lines.append(f"  Diversity score ≥ {task_config['target_diversity_score']}  (current: {obs.diversity_score:.1f})")
    if task_config.get("target_waste_recycling_pct", 0) > 0:
        lines.append(f"  Waste recycling ≥ {task_config['target_waste_recycling_pct']}%  (current: {obs.waste_recycled_pct:.1f}%)")
    if task_config.get("target_water_reduction_pct", 0) > 0:
        lines.append(f"  Water reduction ≥ {task_config['target_water_reduction_pct']}%  (current: {water_pct:.1f}%)")
    if task_config.get("target_employee_satisfaction", 0) > 0:
        lines.append(f"  Employee satisfaction ≥ {task_config['target_employee_satisfaction']}  (current: {obs.employee_satisfaction:.1f})")

    lines += [
        f"  Max compliance violations: {task_config['max_compliance_violations']}  (current: {obs.compliance_violations})",
        "",
        "CURRENT STATE:",
        f"  Energy:       {obs.energy_consumption_kwh:.0f} kWh",
        f"  Renewable:    {obs.renewable_energy_pct:.1f}%",
        f"  Carbon:       {obs.carbon_emissions_tons:.0f} tons  ({carbon_pct:.1f}% reduction from baseline)",
        f"  Waste recycled: {obs.waste_recycled_pct:.1f}%",
        f"  Water usage:  {obs.water_usage_cubic_m:.0f} m³  ({water_pct:.1f}% reduction)",
        f"  Diversity:    {obs.diversity_score:.1f}",
        f"  Satisfaction: {obs.employee_satisfaction:.1f}",
        f"  Budget left:  ${obs.available_budget:,.0f}",
        f"  Violations:   {obs.compliance_violations}",
        f"  Actions taken so far: {obs.actions_taken}",
        "",
        "Choose the best action for this month. Respond with JSON only.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollout generator
# ---------------------------------------------------------------------------

def _heuristic_action(obs: Observation, task_config: Dict[str, Any], rng: random.Random) -> int:
    """
    Simple heuristic policy for generating diverse training prompts.
    NOT the agent — just used to advance the env to varied states.
    """
    budget = obs.available_budget
    carbon_pct = (
        (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
        / obs.baseline_carbon_emissions_tons * 100
        if obs.baseline_carbon_emissions_tons > 0 else 0.0
    )

    # Prioritize with some randomness (epsilon-greedy style)
    if rng.random() < 0.2:
        return rng.randint(0, 8)  # Random exploration

    if budget < 30_000:
        return int(Action.NO_ACTION)

    # Need carbon reduction and have budget
    if carbon_pct < task_config.get("target_carbon_reduction_pct", 15) * 0.5:
        if budget >= 150_000 and obs.renewable_energy_pct < 40:
            return int(Action.INSTALL_SOLAR_PANELS)
        if budget >= 40_000:
            return int(Action.CARBON_OFFSET_PURCHASE)

    if obs.renewable_energy_pct < task_config.get("target_renewable_pct", 30) * 0.7:
        if budget >= 150_000:
            return int(Action.INSTALL_SOLAR_PANELS)
        if budget >= 80_000:
            return int(Action.UPGRADE_HVAC_EFFICIENCY)

    if obs.diversity_score < task_config.get("target_diversity_score", 60) and budget >= 50_000:
        return int(Action.DIVERSITY_HIRING_INITIATIVE)

    if obs.waste_recycled_pct < task_config.get("target_waste_recycling_pct", 0) and budget >= 25_000:
        return int(Action.IMPLEMENT_RECYCLING_PROGRAM)

    # Default: small useful action
    candidates = [int(Action.ENERGY_AUDIT), int(Action.CARBON_OFFSET_PURCHASE)]
    affordable = [a for a in candidates if budget >= 15_000]
    return rng.choice(affordable) if affordable else int(Action.NO_ACTION)


def generate_dataset(
    n_episodes_per_task: int = 20,
    seeds: List[int] = None,
    output_path: str = "data/esg_prompts.jsonl",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate a JSONL prompt dataset by rolling out the environment.

    Each line contains a prompt (system + user) and metadata for reward
    computation during GRPO training.

    Args:
        n_episodes_per_task: How many rollout episodes per task.
        seeds: RNG seeds to use (defaults to 0..n_episodes_per_task-1).
        output_path: Where to save the JSONL file.
        verbose: Print progress.

    Returns:
        List of sample dicts.
    """
    if seeds is None:
        seeds = list(range(n_episodes_per_task))

    samples: List[Dict[str, Any]] = []
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    for task_id, task_config in TASKS.items():
        task_cfg_dict = task_config.model_dump()
        if verbose:
            print(f"[dataset] Generating for task: {task_id} ({n_episodes_per_task} episodes)")

        for seed in seeds:
            rng = random.Random(seed)
            env = ESGEnvironment(task_config=task_config, seed=seed)
            obs = env.reset()

            for step in range(task_config.max_steps):
                # Record this (state, step) as a training prompt
                user_prompt = build_obs_prompt(task_cfg_dict, obs, step)
                sample = {
                    "task_id": task_id,
                    "seed": seed,
                    "step": step,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "obs_snapshot": obs.model_dump(),
                    # Full prompt in chat format for TRL
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                samples.append(sample)

                # Advance env with heuristic action (not what we're training)
                action = _heuristic_action(obs, task_cfg_dict, rng)
                try:
                    obs, _, terminated, truncated, _ = env.step(action)
                except Exception:
                    break
                if terminated or truncated:
                    break

    # Save JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[dataset] Saved {len(samples)} samples -> {out_file}")

    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ESG training dataset")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument("--output", type=str, default="data/esg_prompts.jsonl")
    args = parser.parse_args()

    samples = generate_dataset(
        n_episodes_per_task=args.episodes,
        output_path=args.output,
        verbose=True,
    )
    print(f"Total samples: {len(samples)}")
