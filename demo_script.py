"""
Demo script — shows before/after agent behavior in terminal.

This script demonstrates:
  1. A baseline (heuristic) agent run on the easy task
  2. Reward component breakdown at each step
  3. Final ESG score and state

Run this during a demo to show judges the environment in action.
Usage:
  python demo_script.py
  python demo_script.py --task aggressive_sustainability
  python demo_script.py --model outputs/esg_rl_v1/lora_adapter  (shows trained agent)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from env import ESGEnvironment
from models import Action, Observation
from tasks import TASKS, grade_task
from dataset_builder import _heuristic_action

ACTION_NAMES = {
    0: "INSTALL_SOLAR_PANELS",
    1: "UPGRADE_HVAC_EFFICIENCY",
    2: "IMPLEMENT_RECYCLING_PROGRAM",
    3: "INSTALL_WATER_RECYCLING",
    4: "CARBON_OFFSET_PURCHASE",
    5: "DIVERSITY_HIRING_INITIATIVE",
    6: "EMPLOYEE_WELLNESS_PROGRAM",
    7: "ENERGY_AUDIT",
    8: "NO_ACTION",
}

SEP = "=" * 70


def print_obs(obs: Observation, step: int, max_steps: int):
    carbon_pct = max(0, (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
                     / obs.baseline_carbon_emissions_tons * 100
                     if obs.baseline_carbon_emissions_tons > 0 else 0)
    water_pct = max(0, (obs.baseline_water_usage_cubic_m - obs.water_usage_cubic_m)
                    / obs.baseline_water_usage_cubic_m * 100
                    if obs.baseline_water_usage_cubic_m > 0 else 0)

    print(f"\n  Month {step}/{max_steps} | Budget: ${obs.available_budget:>10,.0f}")
    print(f"  Carbon Reduction: {carbon_pct:5.1f}%  (target: {obs.target_carbon_reduction_pct:.0f}%)")
    print(f"  Renewable Energy: {obs.renewable_energy_pct:5.1f}%  (target: {obs.target_renewable_pct:.0f}%)")
    print(f"  Diversity Score:  {obs.diversity_score:5.1f}")
    print(f"  Waste Recycled:   {obs.waste_recycled_pct:5.1f}%")
    print(f"  Water Reduction:  {water_pct:5.1f}%")
    print(f"  Audit Score:      {obs.audit_score:5.1f}")
    print(f"  Violations:       {obs.compliance_violations}")


def print_reward(info: dict, reward: float):
    rc = info.get("reward_components", {})
    pos = {k: v for k, v in rc.items() if v > 0 and k != "total_reward"}
    neg = {k: v for k, v in rc.items() if v < 0}

    if pos:
        print(f"  [+] " + "  ".join(f"{k.replace('_reward','').replace('_progress','')}: +{v:.3f}" for k, v in pos.items()))
    if neg:
        print(f"  [-] " + "  ".join(f"{k.replace('_penalty','')}: {v:.3f}" for k, v in neg.items()))
    print(f"  => Step reward: {reward:+.3f}")


def run_demo(task_id: str, agent_mode: str = "heuristic", model_path: str = None, delay: float = 0.3):
    task_config = TASKS[task_id]
    task_cfg_dict = task_config.model_dump()

    print(f"\n{SEP}")
    print(f"  ESG COMPLIANCE ENVIRONMENT DEMO")
    print(f"  Task:  {task_id} ({task_config.difficulty})")
    print(f"  Agent: {agent_mode}")
    print(f"  Steps: {task_config.max_steps} months | Budget: ${task_config.initial_budget:,.0f}")
    print(SEP)

    # Select agent
    if agent_mode == "random":
        rng = random.Random(42)
        agent_fn = lambda obs, cfg, r: r.randint(0, 8)
    else:
        rng = random.Random(42)
        agent_fn = _heuristic_action

    if model_path:
        from benchmark import llm_agent_factory
        print(f"  Loading trained model: {model_path}...")
        agent_fn = llm_agent_factory(model_path)
        print(f"  Model loaded.\n")

    # Run episode
    env = ESGEnvironment(task_config=task_config, seed=42)
    obs = env.reset()

    print(f"\n  INITIAL STATE:")
    print_obs(obs, 0, task_config.max_steps)

    total_reward = 0.0

    for step in range(task_config.max_steps):
        action = agent_fn(obs, task_cfg_dict, rng)
        action_name = ACTION_NAMES[action]

        print(f"\n{'-'*70}")
        print(f"  ACTION {step+1}: [{action}] {action_name}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print_obs(obs, step + 1, task_config.max_steps)
        print_reward(info, reward)

        if delay > 0:
            time.sleep(delay)

        if terminated or truncated:
            reason = "COMPLETED" if terminated else "TRUNCATED (max steps)"
            print(f"\n  Episode ended: {reason}")
            break

    # Final score
    final_score = grade_task(task_id, obs)
    carbon_pct = max(0, (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
                     / obs.baseline_carbon_emissions_tons * 100
                     if obs.baseline_carbon_emissions_tons > 0 else 0)

    print(f"\n{SEP}")
    print(f"  FINAL RESULTS")
    print(SEP)
    print(f"  Score:            {final_score:.3f} / 1.0")
    print(f"  Total Reward:     {total_reward:.2f}")
    print(f"  Carbon Reduction: {carbon_pct:.1f}% (target: {task_config.target_carbon_reduction_pct:.0f}%)")
    print(f"  Renewable Energy: {obs.renewable_energy_pct:.1f}% (target: {task_config.target_renewable_pct:.0f}%)")
    print(f"  Budget Remaining: ${obs.available_budget:,.0f}")
    print(f"  Actions taken:    {obs.actions_taken}")
    print(SEP)

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Environment Demo")
    parser.add_argument("--task", default="basic_compliance",
                        choices=list(TASKS.keys()))
    parser.add_argument("--agent", default="heuristic", choices=["random", "heuristic", "llm"])
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--fast", action="store_true", help="No delay between steps")
    args = parser.parse_args()

    score = run_demo(
        task_id=args.task,
        agent_mode=args.agent,
        model_path=args.model,
        delay=0.0 if args.fast else 0.3,
    )

    sys.exit(0 if score > 0.5 else 1)
