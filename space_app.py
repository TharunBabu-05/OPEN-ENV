"""
Gradio Space frontend for ESG Compliance Environment.

Provides an interactive web UI for judges to:
  1. Play with the environment manually (pick actions, see results)
  2. Watch the heuristic agent auto-play
  3. See reward component breakdown in real-time
  4. View final ESG metrics and score

Deploy as a HuggingFace Space with SDK=gradio.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

# Make project importable
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Install gradio: pip install gradio")

from env import ESGEnvironment
from models import Action, Observation
from tasks import TASKS, grade_task
from dataset_builder import _heuristic_action

# ---------------------------------------------------------------------------
# State management (Gradio uses session state)
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: "🌞 Install Solar Panels ($150K)",
    1: "❄️ Upgrade HVAC Efficiency ($80K)",
    2: "♻️ Implement Recycling Program ($25K)",
    3: "💧 Install Water Recycling ($60K)",
    4: "🌿 Carbon Offset Purchase ($40K)",
    5: "👥 Diversity Hiring Initiative ($50K)",
    6: "❤️ Employee Wellness Program ($30K)",
    7: "🔍 Energy Audit ($15K)",
    8: "⏸️ No Action ($0)",
}

TASK_OPTIONS = {
    "Easy — Basic Compliance (6 months, $500K)": "basic_compliance",
    "Medium — Aggressive Sustainability (9 months, $750K)": "aggressive_sustainability",
    "Hard — Carbon Neutral Excellence (12 months, $1M)": "carbon_neutral_excellence",
}


def make_empty_state():
    return {"env": None, "obs": None, "step": 0, "history": [], "task_id": "basic_compliance"}


def format_observation(obs: Observation, step: int, max_steps: int) -> str:
    """Format observation as a clean markdown string."""
    if obs is None:
        return "**No active session. Click 'Start New Episode' to begin.**"

    carbon_pct = max(0, (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
                     / obs.baseline_carbon_emissions_tons * 100
                     if obs.baseline_carbon_emissions_tons > 0 else 0)
    water_pct = max(0, (obs.baseline_water_usage_cubic_m - obs.water_usage_cubic_m)
                    / obs.baseline_water_usage_cubic_m * 100
                    if obs.baseline_water_usage_cubic_m > 0 else 0)

    budget_color = "🟢" if obs.available_budget > 100_000 else ("🟡" if obs.available_budget > 0 else "🔴")
    violation_color = "🟢" if obs.compliance_violations == 0 else ("🟡" if obs.compliance_violations <= 2 else "🔴")

    return f"""## 📊 Month {step}/{max_steps} — ESG Dashboard

### 🌿 Environmental
| Metric | Current | Target |
|--------|---------|--------|
| Carbon Reduction | **{carbon_pct:.1f}%** | {obs.target_carbon_reduction_pct:.0f}% |
| Renewable Energy | **{obs.renewable_energy_pct:.1f}%** | {obs.target_renewable_pct:.0f}% |
| Waste Recycled | **{obs.waste_recycled_pct:.1f}%** | — |
| Water Reduction | **{water_pct:.1f}%** | — |

### 👥 Social
| Metric | Value |
|--------|-------|
| Diversity Score | **{obs.diversity_score:.1f}** / 100 |
| Employee Satisfaction | **{obs.employee_satisfaction:.1f}** / 100 |

### 🏛️ Governance & Finance
| Metric | Value |
|--------|-------|
| {violation_color} Compliance Violations | **{obs.compliance_violations}** |
| Audit Score | **{obs.audit_score:.1f}** / 100 |
| {budget_color} Budget Remaining | **${obs.available_budget:,.0f}** |
| Total Investment | ${obs.total_investment:,.0f} |

### 📈 Actions Taken
`{obs.actions_taken}`
"""


def format_reward_breakdown(info: Dict) -> str:
    """Format reward components as markdown table."""
    if not info:
        return ""
    rc = info.get("reward_components", {})
    if not rc:
        return ""
    total = rc.get("total_reward", 0)
    color = "🟢" if total > 0 else "🔴"

    rows = []
    component_names = {
        "carbon_progress_reward": "Carbon Progress",
        "renewable_progress_reward": "Renewable Progress",
        "diversity_progress_reward": "Diversity Progress",
        "waste_recycling_reward": "Waste Recycling",
        "water_reduction_reward": "Water Reduction",
        "budget_penalty": "Budget Penalty",
        "compliance_penalty": "Compliance Penalty",
        "quarterly_bonus": "Quarterly Bonus",
        "synergy_bonus": "Synergy Bonus",
        "anti_cheat_penalty": "Anti-Cheat",
        "format_compliance_reward": "Format Compliance",
        "task_completion_reward": "Task Completion",
    }
    for key, label in component_names.items():
        val = rc.get(key, 0)
        if val != 0:
            sign = "+" if val > 0 else ""
            rows.append(f"| {label} | {sign}{val:.3f} |")

    if not rows:
        return f"**Step Reward: {color} {total:.3f}**"

    table = "\n".join(rows)
    return f"""**Step Reward: {color} {total:.3f}**

| Component | Value |
|-----------|-------|
{table}
"""


# ---------------------------------------------------------------------------
# Gradio action handlers
# ---------------------------------------------------------------------------

def start_episode(task_label: str, state: dict) -> Tuple:
    task_id = TASK_OPTIONS.get(task_label, "basic_compliance")
    task_config = TASKS[task_id]
    env = ESGEnvironment(task_config=task_config, seed=42)
    obs = env.reset()

    state.update({
        "env": env,
        "obs": obs,
        "step": 0,
        "history": [],
        "task_id": task_id,
        "max_steps": task_config.max_steps,
    })

    obs_text = format_observation(obs, 0, task_config.max_steps)
    action_choices = [ACTION_NAMES[i] for i in range(9)]

    return (
        state,
        obs_text,
        "",  # reward breakdown cleared
        "Episode started! Choose an action below.",
        gr.update(choices=action_choices, value=ACTION_NAMES[0]),
    )


def take_action(action_label: str, state: dict) -> Tuple:
    if state.get("env") is None:
        return state, "**Start an episode first.**", "", "No active episode."

    env = state["env"]
    obs = state["obs"]

    # Reverse lookup action ID
    action_id = next(k for k, v in ACTION_NAMES.items() if v == action_label)

    try:
        new_obs, reward, terminated, truncated, info = env.step(action_id)
    except Exception as e:
        return state, format_observation(obs, state["step"], state["max_steps"]), "", f"Error: {e}"

    state["obs"] = new_obs
    state["step"] += 1
    state["history"].append({"action": action_id, "reward": reward})

    obs_text = format_observation(new_obs, state["step"], state["max_steps"])
    reward_text = format_reward_breakdown(info)

    done = terminated or truncated
    if done:
        score = grade_task(state["task_id"], new_obs)
        status = f"Episode Complete! Final Score: **{score:.3f}** / 1.0"
    else:
        status = f"Month {state['step']} complete. Reward this step: {reward:+.3f}"

    return state, obs_text, reward_text, status


def auto_play(state: dict) -> Tuple:
    """Let the heuristic agent play one step."""
    if state.get("env") is None:
        return state, "**Start an episode first.**", "", "No active episode."

    env = state["env"]
    obs = state["obs"]
    rng = random.Random(state["step"])
    task_cfg = TASKS[state["task_id"]].model_dump()

    action_id = _heuristic_action(obs, task_cfg, rng)
    action_label = ACTION_NAMES[action_id]

    return take_action(action_label, state)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="ESG Compliance RL Environment",
        css="""
        .gradio-container { background: #0f172a; }
        h1, h2, h3 { color: #34d399 !important; }
        .score-box { background: #1e293b; border-radius: 8px; padding: 12px; }
        """
    ) as demo:

        state = gr.State(make_empty_state())

        # Header
        gr.Markdown("""
# 🌍 ESG Compliance & Sustainability RL Environment
**OpenEnv Hackathon — Long-Horizon Strategic Planning**

Train and evaluate AI agents on corporate ESG decision-making. Each month, choose an action
to reduce carbon emissions, boost renewable energy, and improve social scores — all while
managing a limited budget over 6–12 months.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Task selection
                task_selector = gr.Dropdown(
                    choices=list(TASK_OPTIONS.keys()),
                    value=list(TASK_OPTIONS.keys())[0],
                    label="Select Task Difficulty",
                )
                start_btn = gr.Button("🚀 Start New Episode", variant="primary", size="lg")

                # Observation display
                obs_display = gr.Markdown("**Click 'Start New Episode' to begin.**")

            with gr.Column(scale=2):
                # Action selection
                action_selector = gr.Radio(
                    choices=[ACTION_NAMES[i] for i in range(9)],
                    label="Choose Action for This Month",
                    value=ACTION_NAMES[0],
                )
                with gr.Row():
                    step_btn = gr.Button("▶️ Take Action", variant="primary")
                    auto_btn = gr.Button("🤖 Heuristic Agent Step", variant="secondary")

                status_display = gr.Markdown("Ready to play.")
                reward_display = gr.Markdown("")

        # Wire up events
        start_btn.click(
            start_episode,
            inputs=[task_selector, state],
            outputs=[state, obs_display, reward_display, status_display, action_selector],
        )
        step_btn.click(
            take_action,
            inputs=[action_selector, state],
            outputs=[state, obs_display, reward_display, status_display],
        )
        auto_btn.click(
            auto_play,
            inputs=[state],
            outputs=[state, obs_display, reward_display, status_display],
        )

        # Footer
        gr.Markdown("""
---
**Built for OpenEnv Hackathon** | Theme #2: Super Long-Horizon Planning | 
[GitHub](https://github.com/TharunBabu-05/OPEN-ENV) | Environment: ESG Compliance v1.0
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
