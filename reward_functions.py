"""
Standalone verifiable reward functions for GRPO training.

These functions are passed to TRL's GRPOTrainer as reward_funcs.
Each function receives a batch of model completions and corresponding
metadata, steps the environment, and returns a list of reward scalars.

Design principles (from hackathon guide §7, §8):
- Multiple independent reward functions (not a single monolithic one).
- Objective, verifiable signals — no LLM-as-judge.
- Anti-cheat checks built into every function.
- Each function is stateless with respect to training; it creates its own
  ephemeral environment instance to evaluate the completion.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from env import ESGEnvironment
from models import Action, Observation
from tasks import TASKS, grade_task


# ---------------------------------------------------------------------------
# Utility: parse action from LLM completion
# ---------------------------------------------------------------------------

def _parse_action(completion: str) -> Optional[int]:
    """
    Extract action integer from LLM output.

    Handles:
    - Pure JSON: {"action": 3, "reasoning": "..."}
    - Markdown-wrapped JSON: ```json\n{...}\n```
    - Bare integer fallback
    """
    text = completion.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try JSON parse
    try:
        data = json.loads(text)
        action = int(data["action"])
        if 0 <= action <= 8:
            return action
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Fallback: find first standalone integer 0-8
    match = re.search(r'\b([0-8])\b', text)
    if match:
        return int(match.group(1))

    return None  # Unparseable


# ---------------------------------------------------------------------------
# Reward function 1: Environment outcome reward
# Runs the env step and returns the shaped reward from RewardComponents.
# ---------------------------------------------------------------------------

def reward_env_outcome(
    completions: List[str],
    obs_snapshots: List[Dict[str, Any]],
    task_ids: List[str],
    **kwargs,
) -> List[float]:
    """
    Primary reward: step the environment and return shaped reward.

    - Parseable, valid action → environment reward (can be positive or negative)
    - Unparseable output → -1.0 (format failure penalty)
    - Exception during step → -0.5
    """
    rewards = []

    for completion, obs_dict, task_id in zip(completions, obs_snapshots, task_ids):
        action = _parse_action(completion)

        if action is None:
            rewards.append(-1.0)
            continue

        try:
            task_config = TASKS[task_id]
            env = ESGEnvironment(task_config=task_config, seed=obs_dict.get("_seed", 42))
            # Restore environment to the snapshot state
            obs = Observation(**{k: v for k, v in obs_dict.items() if not k.startswith("_")})
            env.reset()
            # Manually override internal state with snapshot
            env.state_internal.observation = obs
            env.state_internal.previous_carbon_emissions = obs.carbon_emissions_tons
            env.state_internal.previous_renewable_pct = obs.renewable_energy_pct
            env.state_internal.previous_diversity_score = obs.diversity_score
            env.state_internal.previous_waste_recycled_pct = obs.waste_recycled_pct
            env.state_internal.previous_water_usage = obs.water_usage_cubic_m
            env.state_internal.step_count = obs.current_month - 1

            _, reward, _, _, _ = env.step(action)
            rewards.append(float(reward))

        except Exception as e:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# Reward function 2: Format compliance
# Checks whether the output is valid parseable JSON with required fields.
# ---------------------------------------------------------------------------

def reward_format_compliance(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Format reward: returns 1.0 for well-formed JSON with action + reasoning,
    0.5 for parseable action-only, 0.0 for unparseable.
    """
    rewards = []

    for completion in completions:
        text = completion.strip()

        # Strip markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            action = int(data.get("action", -1))
            has_reasoning = bool(data.get("reasoning", "").strip())

            if 0 <= action <= 8 and has_reasoning:
                rewards.append(1.0)   # Perfect format
            elif 0 <= action <= 8:
                rewards.append(0.5)   # Valid action, no reasoning
            else:
                rewards.append(0.0)   # Invalid action value
        except (json.JSONDecodeError, TypeError, ValueError):
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Reward function 3: Anti-cheat
# Penalizes choosing NO_ACTION (action=8) or the same trivial action
# repeatedly when there's meaningful budget available.
# ---------------------------------------------------------------------------

def reward_anti_cheat(
    completions: List[str],
    obs_snapshots: List[Dict[str, Any]],
    **kwargs,
) -> List[float]:
    """
    Anti-cheat reward: penalizes lazy/exploitative strategies.

    - Choosing NO_ACTION when budget > $100K → -0.5
    - Choosing the same cheap action (audit/wellness/recycling) → -0.3
    - Otherwise → 0.0
    """
    rewards = []

    for completion, obs_dict in zip(completions, obs_snapshots):
        action = _parse_action(completion)
        budget = obs_dict.get("available_budget", 0.0)
        actions_taken = obs_dict.get("actions_taken", [])

        if action is None:
            rewards.append(0.0)  # Format reward handles this
            continue

        penalty = 0.0

        # Penalize NO_ACTION when budget is healthy
        if action == int(Action.NO_ACTION) and budget > 100_000:
            penalty -= 0.5

        # Penalize repeating trivial actions
        if actions_taken:
            last_3 = actions_taken[-3:]
            if len(last_3) >= 2 and all(a == action for a in last_3) and action in (2, 6, 7, 8):
                penalty -= 0.3

        rewards.append(penalty)

    return rewards


# ---------------------------------------------------------------------------
# Reward function 4: Task progress reward
# Computes how much overall task progress was achieved at the final step.
# This is the terminal/outcome signal.
# ---------------------------------------------------------------------------

def reward_task_progress(
    completions: List[str],
    obs_snapshots: List[Dict[str, Any]],
    task_ids: List[str],
    is_terminal: List[bool] = None,
    **kwargs,
) -> List[float]:
    """
    Terminal reward based on the grader score (0.0 to 1.0).
    Only applied at episode end; returns 0.0 for non-terminal steps.

    Scaled to [-1.0, +2.0] range to create a strong signal.
    """
    rewards = []
    is_terminal = is_terminal or [False] * len(completions)

    for completion, obs_dict, task_id, terminal in zip(
        completions, obs_snapshots, task_ids, is_terminal
    ):
        if not terminal:
            rewards.append(0.0)
            continue

        action = _parse_action(completion)
        if action is None:
            rewards.append(-1.0)
            continue

        try:
            obs = Observation(**{k: v for k, v in obs_dict.items() if not k.startswith("_")})
            score = grade_task(task_id, obs)
            # Scale: 0 score → -1.0, perfect score → +2.0
            scaled = (score - 0.5) * 4.0
            rewards.append(float(max(-1.0, min(2.0, scaled))))
        except Exception:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# Composite reward (used when a single reward_fn is needed)
# ---------------------------------------------------------------------------

def reward_composite(
    completions: List[str],
    obs_snapshots: List[Dict[str, Any]],
    task_ids: List[str],
    is_terminal: List[bool] = None,
    **kwargs,
) -> List[float]:
    """
    Weighted combination of all reward signals.
    
    Weights:
      60% environment outcome (shaped reward)
      20% format compliance
      10% anti-cheat
      10% task progress (terminal only)
    """
    env_rewards = reward_env_outcome(completions, obs_snapshots, task_ids)
    fmt_rewards = reward_format_compliance(completions)
    cheat_rewards = reward_anti_cheat(completions, obs_snapshots)
    progress_rewards = reward_task_progress(completions, obs_snapshots, task_ids, is_terminal)

    combined = []
    for r_env, r_fmt, r_cheat, r_prog in zip(env_rewards, fmt_rewards, cheat_rewards, progress_rewards):
        total = 0.6 * r_env + 0.2 * r_fmt + 0.1 * r_cheat + 0.1 * r_prog
        combined.append(float(total))

    return combined


# ---------------------------------------------------------------------------
# Exported list for TRL GRPOTrainer
# ---------------------------------------------------------------------------

ALL_REWARD_FUNCTIONS = [
    reward_env_outcome,
    reward_format_compliance,
    reward_anti_cheat,
    reward_task_progress,
]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tasks import TASKS

    # Build a sample obs snapshot
    from env import ESGEnvironment
    env = ESGEnvironment(TASKS["basic_compliance"], seed=42)
    obs = env.reset()
    obs_dict = obs.model_dump()
    obs_dict["_seed"] = 42

    # Test completions
    good = '{"action": 0, "reasoning": "Solar panels will increase renewable energy significantly."}'
    bad_fmt = "I think action 3 is good because water recycling helps."
    lazy = '{"action": 8, "reasoning": "Saving budget."}'

    for name, comp in [("good", good), ("bad_fmt", bad_fmt), ("lazy", lazy)]:
        env_r = reward_env_outcome([comp], [obs_dict], ["basic_compliance"])
        fmt_r = reward_format_compliance([comp])
        cheat_r = reward_anti_cheat([comp], [obs_dict])
        composite = reward_composite([comp], [obs_dict], ["basic_compliance"])
        print(f"\n[{name}]")
        print(f"  env_outcome:  {env_r[0]:.3f}")
        print(f"  format:       {fmt_r[0]:.3f}")
        print(f"  anti_cheat:   {cheat_r[0]:.3f}")
        print(f"  composite:    {composite[0]:.3f}")
