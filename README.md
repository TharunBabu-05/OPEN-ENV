<div align="center">

# 🌍 OPEN-ENV: ESG Compliance Reinforcement Learning Environment

<p align="center">
  <a href="https://github.com/TharunBabu-05/OPEN-ENV/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://huggingface.co/tharun5054/esg-rl-agent-grpo"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model%20on%20HF-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/spaces/tharun5054/esg-rl-train"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Space%20on%20HF-blue" alt="Hugging Face Space"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/GRPO-TRL%20%2B%20Unsloth-orange" alt="GRPO">
  <img src="https://img.shields.io/badge/GPU-A100%2080GB-76B900?logo=nvidia" alt="A100">
  <img src="https://img.shields.io/badge/OpenEnv-Hackathon%202026-purple" alt="OpenEnv">
</p>

<p align="center">
  <b>A reinforcement learning environment where an LLM agent acts as a corporate ESG sustainability strategist — making sequential multi-objective decisions to hit carbon, renewable energy, diversity, and waste targets within a fixed budget and timeline.</b>
</p>

<br/>

```
🏭 Carbon at 100% ──► 🤖 RL Agent Decides ──► ☀️ Solar Panels Installed
💰 Budget: $500K           (monthly)              🌿 Carbon drops to 47%
📅 6-Month Horizon     ────────────────────►      ✅ Target ACHIEVED
```

</div>

---

## 🎯 What Is This?

OPEN-ENV is an **OpenAI Gym-compatible** reinforcement learning environment designed to train LLMs to solve long-horizon, multi-objective corporate decision-making tasks. It was built for the **OpenEnv Hackathon 2026**.

The agent plays the role of a **corporate sustainability strategist**, choosing from 9 monthly interventions across 6–12 time steps to hit ESG (Environmental, Social, Governance) KPIs — all within a budget constraint.

The key insight: **greedy, one-step-optimal strategies fail**. Only agents capable of long-horizon planning (e.g., "install solar now → big dividend for 6 months") succeed.

---

## 🗂️ Repository Structure

```
OPEN-ENV/
├── env.py                   # Core ESG Gym environment (state, actions, rewards)
├── models.py                # Pydantic schemas (Observation, Action, RewardComponents)
├── tasks.py                 # Task configs (easy/medium/hard ESG targets)
├── reward_functions.py      # 4 independent GRPO reward functions
├── dataset_builder.py       # Builds GRPO training dataset from heuristic rollouts
├── train_rl.py              # GRPO training pipeline (Unsloth + TRL)
├── train_config_a100.yaml   # A100-optimized training config (150 steps, 3-stage curriculum)
├── benchmark.py             # Benchmark runner (random / heuristic / LLM)
├── plot_results.py          # Generates comparison charts
├── app.py                   # FastAPI server for OpenEnv judging API
├── space_train_app.py       # Gradio app for HF Space training UI
├── demo_script.py           # Live agent demo in terminal
├── run_benchmark.ps1        # One-click benchmark runner (Windows)
├── results/                 # Benchmark outputs and plots
│   ├── score_comparison.png
│   ├── reward_history.png
│   ├── esg_metrics.png
│   ├── baseline_random.json
│   ├── baseline_heuristic.json
│   └── trained_v2.json      # V2 A100 GRPO results
└── EVALUATION_REPORT.md     # Full model evaluation with scores
```

---

## 🏗️ Environment Architecture

### State Space (17 Fields)

| Category | Fields |
|----------|--------|
| 🌿 Environmental | `carbon_emissions_tons`, `renewable_energy_pct`, `energy_consumption_kwh`, `waste_recycled_pct`, `water_usage_liters` |
| 🤝 Social | `diversity_score`, `employee_satisfaction` |
| 🏛️ Governance | `compliance_violations`, `audit_score` |
| 💰 Financial | `available_budget`, `monthly_costs`, `total_investment` |
| 📅 Temporal | `current_month`, `quarters_completed` |
| 🎯 Targets | `target_carbon_reduction_pct`, `target_renewable_pct`, `baseline_carbon_emissions_tons` |

### Action Space (9 Discrete Actions)

| ID | Action | Cost | Effect Duration |
|----|--------|------|----------------|
| 0 | ☀️ Install Solar Panels | $150K | 6 months |
| 1 | 🏢 Upgrade HVAC | $80K | 12 months |
| 2 | ♻️ Recycling Program | $25K | Permanent |
| 3 | 💧 Water Recycling System | $60K | 12 months |
| 4 | 🌫️ Carbon Offset Credits | $40K | Immediate |
| 5 | 👥 Diversity Hiring Program | $50K | Permanent |
| 6 | 💆 Employee Wellness Program | $30K | Permanent |
| 7 | ⚡ Energy Audit | $15K | Ongoing |
| 8 | 💤 No Action | $0 | — |

### 3 Task Difficulty Levels

| Task | Difficulty | Steps | Budget | Targets |
|------|:----------:|:-----:|:------:|---------|
| `basic_compliance` | 🟢 Easy | 6 | $500K | -15% carbon, 30% renewable |
| `aggressive_sustainability` | 🟡 Medium | 9 | $750K | -40% carbon, 60% renewable, 70% recycling |
| `carbon_neutral_excellence` | 🔴 Hard | 12 | $1.2M | -90% carbon, 80% renewable, ALL metrics |

---

## 🧠 Training Pipeline

```
Dataset Builder
(350 samples: heuristic + random + adversarial)
        │
        ▼
ESGEnvironment.step()  ←──── LLM Generates Action JSON
        │
        ▼
4 Independent Reward Functions
  ├── reward_env_outcome()     (45%) — Simulator shaped reward
  ├── reward_format_compliance() (25%) — Valid JSON enforcement
  ├── reward_anti_cheat()       (15%) — Penalize NO_ACTION spam
  └── reward_task_progress()    (15%) — Terminal grader signal
        │
        ▼
TRL GRPOTrainer + Unsloth 4-bit LoRA
(Qwen2.5-0.5B-Instruct, bf16, A100 80GB)
        │
        ▼
Trained LoRA Adapter → tharun5054/esg-rl-agent-grpo
```

### A100 Training Config (V2)
- **150 steps** across a **3-stage curriculum**: 50 steps easy → 60 steps medium → 40 steps hard
- **6 GRPO generations** per prompt for better policy gradient signal
- **LR: 8e-6** with grad norm 0.8 for stable convergence
- **350-sample dataset** mixing heuristic rollouts, random exploration, and adversarial scenarios
- **Cost:** ~$4.50 on A100 Large

---

## 📊 Benchmark Results (V2 A100 Model)

All scores measured across **5 seeds per task** using the deterministic environment grader.

| Agent | Easy | Medium | Hard | Overall |
|-------|:----:|:------:|:----:|:-------:|
| 🎲 Random | 0.740 | 0.643 | 0.678 | **0.687** |
| 🧠 Heuristic Baseline | 1.000 | 0.847 | 0.852 | **0.900** |
| 🚀 **V2 GRPO (A100)** | **1.000** | **0.880** | **0.726** | **0.869** |

> 🏆 **The V2 GRPO Agent outperformed the hand-coded heuristic on the Medium task (0.880 vs 0.847)**, demonstrating genuine zero-shot generalization of learned ESG strategies.

### Improvement vs. Random Baseline

| Task | Random | V2 GRPO | Δ |
|------|:------:|:-------:|:-:|
| Easy | 0.740 | 1.000 | **+35.1%** |
| Medium | 0.643 | 0.880 | **+36.8%** |
| Hard | 0.678 | 0.726 | **+7.1%** |
| **Overall** | **0.687** | **0.869** | **+26.5%** |

---

## 🛡️ Anti-Reward-Hacking Measures

Following OpenEnv Hackathon Guide §8 requirements:

- ✅ **12 independent reward signals** (not one monolithic function)
- ✅ **Anti-cheat penalty** for `NO_ACTION` spamming
- ✅ **Anti-cheat penalty** for repeating cheap actions in sequence
- ✅ **Budget bankruptcy detection** (severe negative reward)
- ✅ **Per-step wall-clock timeout** (10s) in FastAPI server
- ✅ **Thread-safe session isolation** (UUID per session, no shared state)
- ✅ **Deterministic graders** (same input → same score, always)
- ✅ **Format compliance reward** (forces structured JSON output)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/TharunBabu-05/OPEN-ENV
cd OPEN-ENV
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Smoke Test (No GPU)
```bash
python train_rl.py --smoke_test
```

### 3. Live Demo — Watch the Agent Play
```bash
python demo_script.py
```

### 4. Run Full Benchmark
```bash
# Windows
.\run_benchmark.ps1

# Or manually
python benchmark.py --mode random --output results/baseline_random.json
python benchmark.py --mode heuristic --output results/baseline_heuristic.json
python benchmark.py --mode llm --model_path tharun5054/esg-rl-agent-grpo --output results/trained.json
```

### 5. Start the FastAPI Server
```bash
uvicorn app:app --reload --port 8000
```

### 6. Train on A100 (HuggingFace Space)
Go to the [Training Space](https://huggingface.co/spaces/tharun5054/esg-rl-train), enter your HF token, and click **🚀 Start Training**.

---

## 🤖 Use the Trained Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "tharun5054/esg-rl-agent-grpo", subfolder="lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("tharun5054/esg-rl-agent-grpo", subfolder="lora_adapter")

prompt = """You are an ESG sustainability strategist. Current state:
- Carbon Emissions: 1200 tons (target: -15%)
- Renewable Energy: 12% (target: 30%)
- Available Budget: $500,000

Choose an action (0-8) and explain your reasoning."""

inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# Output: {"action": 0, "reasoning": "Installing solar panels will boost renewable energy..."}
```

---

## 📐 Environment API

The environment is fully compatible with the OpenEnv judging API:

```
POST /reset         → { "obs": {...17 fields...}, "session_id": "..." }
POST /step          → { "obs": {...}, "reward": 1.23, "done": false, "info": {...} }
GET  /tasks         → List of available task configs
GET  /health        → Server health check
```

---

## 🔗 Links

| | |
|-|-|
| 📦 **GitHub** | [TharunBabu-05/OPEN-ENV](https://github.com/TharunBabu-05/OPEN-ENV) |
| 🤗 **Trained Model** | [tharun5054/esg-rl-agent-grpo](https://huggingface.co/tharun5054/esg-rl-agent-grpo) |
| 🚀 **Training Space** | [tharun5054/esg-rl-train](https://huggingface.co/spaces/tharun5054/esg-rl-train) |
| 📊 **Evaluation Report** | [EVALUATION_REPORT.md](./EVALUATION_REPORT.md) |
| 📋 **Submission Brief** | [SUBMISSION.md](./SUBMISSION.md) |

---

## 📜 License

MIT © 2026 Tharun Babu
