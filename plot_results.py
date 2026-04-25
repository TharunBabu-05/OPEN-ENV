"""
Results visualization for ESG RL training evidence.

Generates publication-quality plots for hackathon submission:
  1. Score comparison: random vs heuristic vs trained (bar chart)
  2. Per-task score breakdown (grouped bars)
  3. Reward history over episode steps (line chart)
  4. Reward component breakdown (stacked bar)

Usage:
  python plot_results.py --baseline results/baseline_heuristic.json --trained results/trained_v1.json

  # If you only have baseline:
  python plot_results.py --baseline results/baseline_heuristic.json --output results/baseline_only.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (works on servers/Colab)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Run: pip install matplotlib numpy")


# ---------------------------------------------------------------------------
# Color palette (premium look)
# ---------------------------------------------------------------------------

COLORS = {
    "random":    "#94a3b8",   # Gray
    "heuristic": "#60a5fa",   # Blue
    "trained":   "#34d399",   # Green
    "carbon":    "#f87171",   # Red
    "renewable": "#fbbf24",   # Yellow
    "diversity": "#a78bfa",   # Purple
    "waste":     "#34d399",   # Green
    "budget":    "#fb923c",   # Orange
}

TASK_LABELS = {
    "basic_compliance":         "Easy\n(6 months)",
    "aggressive_sustainability": "Medium\n(9 months)",
    "carbon_neutral_excellence": "Hard\n(12 months)",
}


# ---------------------------------------------------------------------------
# Plot 1: Overall score comparison bar chart
# ---------------------------------------------------------------------------

def plot_score_comparison(results: Dict[str, dict], output_path: str):
    """Bar chart comparing agent scores across all tasks."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    agents = list(results.keys())
    task_ids = list(TASK_LABELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")

    # Left: Overall mean
    ax = axes[0]
    overall_scores = []
    bar_colors = []
    for agent in agents:
        score = results[agent].get("overall_mean_score", 0)
        overall_scores.append(score)
        bar_colors.append(COLORS.get(agent, "#94a3b8"))

    bars = ax.bar(agents, overall_scores, color=bar_colors, width=0.5, zorder=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Score (0-1)", color="white", fontsize=12)
    ax.set_title("Overall Mean Score\nAll Tasks", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#475569")
    ax.spines["left"].set_color("#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#334155", zorder=0)

    for bar, score in zip(bars, overall_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{score:.3f}", ha="center", va="bottom", color="white", fontweight="bold")

    # Right: Per-task breakdown
    ax = axes[1]
    n_agents = len(agents)
    n_tasks = len(task_ids)
    x = np.arange(n_tasks)
    width = 0.8 / n_agents

    for i, agent in enumerate(agents):
        task_scores = []
        for task_id in task_ids:
            summary = results[agent].get("task_summary", {}).get(task_id, {})
            task_scores.append(summary.get("mean_score", 0))

        bars = ax.bar(x + i * width - (n_agents - 1) * width / 2,
                      task_scores, width * 0.9,
                      label=agent.capitalize(),
                      color=COLORS.get(agent, "#94a3b8"),
                      zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], color="white", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Score (0-1)", color="white", fontsize=12)
    ax.set_title("Per-Task Score Breakdown", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#475569")
    ax.spines["left"].set_color("#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#334155", zorder=0)
    legend = ax.legend(facecolor="#1e293b", labelcolor="white", edgecolor="#475569")

    plt.suptitle("ESG RL Agent -- Score Comparison\n(Higher is Better)",
                 color="white", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Score comparison -> {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Reward history over episode (per task)
# ---------------------------------------------------------------------------

def plot_reward_history(results: Dict[str, dict], output_path: str):
    """Line chart of cumulative reward per step, per task."""
    if not HAS_MATPLOTLIB:
        return

    task_ids = list(TASK_LABELS.keys())
    fig, axes = plt.subplots(1, len(task_ids), figsize=(15, 5))
    fig.patch.set_facecolor("#0f172a")

    for ax, task_id in zip(axes, task_ids):
        ax.set_facecolor("#1e293b")

        for agent, data in results.items():
            episodes = [e for e in data.get("episodes", []) if e["task_id"] == task_id]
            if not episodes:
                continue

            # Average reward history across seeds
            max_steps = max(len(e["reward_history"]) for e in episodes)
            avg_rewards = []
            for step in range(max_steps):
                step_rewards = [e["reward_history"][step]
                                for e in episodes
                                if step < len(e["reward_history"])]
                avg_rewards.append(sum(step_rewards) / len(step_rewards))

            cumulative = [sum(avg_rewards[:i+1]) for i in range(len(avg_rewards))]
            ax.plot(range(1, len(cumulative) + 1), cumulative,
                    label=agent.capitalize(),
                    color=COLORS.get(agent, "#94a3b8"),
                    linewidth=2.5, marker="o", markersize=5)

        ax.set_title(TASK_LABELS[task_id], color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step (Month)", color="white", fontsize=10)
        ax.set_ylabel("Cumulative Reward", color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#334155", alpha=0.5)
        ax.legend(facecolor="#1e293b", labelcolor="white", edgecolor="#475569", fontsize=9)

    plt.suptitle("Cumulative Reward per Episode Step",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Reward history -> {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Final ESG metrics comparison (radar/spider -- uses bar instead for simplicity)
# ---------------------------------------------------------------------------

def plot_esg_metrics(results: Dict[str, dict], output_path: str):
    """Grouped bar chart of final ESG metric values per agent."""
    if not HAS_MATPLOTLIB:
        return

    metrics = ["carbon_reduction_pct", "renewable_pct", "diversity_score", "waste_recycled_pct"]
    metric_labels = ["Carbon Reduction %", "Renewable %", "Diversity Score", "Waste Recycled %"]
    # Normalize diversity (0-100) to 0-100% for fair display
    normalizers = [100, 100, 100, 100]

    agents = list(results.keys())
    n_agents = len(agents)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    x = np.arange(n_metrics)
    width = 0.8 / n_agents

    for i, agent in enumerate(agents):
        avg_vals = []
        episodes = results[agent].get("episodes", [])
        for metric, norm in zip(metrics, normalizers):
            vals = [e["final_obs"].get(metric, 0) for e in episodes]
            avg_vals.append((sum(vals) / len(vals)) if vals else 0)

        bars = ax.bar(x + i * width - (n_agents - 1) * width / 2,
                      avg_vals, width * 0.9,
                      label=agent.capitalize(),
                      color=COLORS.get(agent, "#94a3b8"),
                      zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color="white", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Value (all on 0-100 scale)", color="white", fontsize=11)
    ax.set_title("Final ESG Metrics -- Agent Comparison\n(Higher is Better for All Metrics)",
                 color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#475569")
    ax.spines["left"].set_color("#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#334155", zorder=0)
    legend = ax.legend(facecolor="#1e293b", labelcolor="white", edgecolor="#475569", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] ESG metrics -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_result(path: str, label: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        print(f"WARNING: {path} not found. Skipping '{label}'.")
        return None
    with open(p) as f:
        data = json.load(f)
    data["_label"] = label
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ESG benchmark results")
    parser.add_argument("--random",    type=str, default=None, help="Path to random baseline JSON")
    parser.add_argument("--baseline",  type=str, default=None, help="Path to heuristic baseline JSON")
    parser.add_argument("--trained",   type=str, default=None, help="Path to trained model JSON")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    results = {}
    if args.random:
        d = load_result(args.random, "random")
        if d: results["random"] = d
    if args.baseline:
        d = load_result(args.baseline, "heuristic")
        if d: results["heuristic"] = d
    if args.trained:
        d = load_result(args.trained, "trained")
        if d: results["trained"] = d

    if not results:
        print("ERROR: Provide at least one result file (--random, --baseline, or --trained)")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_score_comparison(results, str(out_dir / "score_comparison.png"))
    plot_reward_history(results, str(out_dir / "reward_history.png"))
    plot_esg_metrics(results, str(out_dir / "esg_metrics.png"))

    print(f"\nAll plots saved to {out_dir}/")
