<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=OPEN-ENV&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=ESG%20Compliance%20Reinforcement%20Learning&descAlignY=60&descSize=20" width="100%"/>

<!-- Animated Title -->
<a href="https://github.com/TharunBabu-05/OPEN-ENV">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=26&pause=1000&color=00D9FF&center=true&vCenter=true&width=750&lines=ESG+Compliance+RL+Environment;Train+LLM+Agents+with+GRPO;A100+GPU+Curriculum+Learning;Beat+Heuristic+Baselines+with+RL!" alt="Typing SVG" />
</a>

<br/>

<!-- Badges Row 1 -->
<p>
  <a href="https://github.com/TharunBabu-05/OPEN-ENV/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License">
  </a>
  <a href="https://huggingface.co/tharun5054/esg-rl-agent-grpo">
    <img src="https://img.shields.io/badge/🤗_Model-HuggingFace-FFD21E?style=for-the-badge" alt="HF Model">
  </a>
  <a href="https://huggingface.co/spaces/tharun5054/esg-rl-train">
    <img src="https://img.shields.io/badge/🚀_Training_Space-HuggingFace-blue?style=for-the-badge" alt="HF Space">
  </a>
  <a href="https://huggingface.co/spaces/tharun5054/esg-compliance-env">
    <img src="https://img.shields.io/badge/🌍_Env_API-Live-success?style=for-the-badge" alt="API Live">
  </a>
</p>

<!-- Badges Row 2 -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/GRPO-TRL+Unsloth-FF6B35?style=flat-square" alt="GRPO">
  <img src="https://img.shields.io/badge/GPU-A100_80GB-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="A100">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenEnv-Hackathon_2026-9333EA?style=flat-square" alt="OpenEnv">
</p>

<br/>

> **An RL environment where an LLM agent plays the role of a corporate ESG sustainability strategist —**
> **making sequential multi-objective decisions under budget and time constraints.**

<br/>

</div>

---

## ⚡ At a Glance

<div align="center">

<table align="center">
<tr>
  <td colspan="3" align="center"><b>🔄 ESG RL ENVIRONMENT LOOP</b></td>
</tr>
<tr>
  <td align="center" width="260"><b>🏢 Company State</b></td>
  <td align="center" width="260"><b>🤖 LLM Agent</b></td>
  <td align="center" width="260"><b>✅ Results</b></td>
</tr>
<tr>
  <td>• Carbon: 1200 tons<br/>• Renewable: 12%<br/>• Budget: $500K<br/>• Month: 1/6</td>
  <td><code>{"action": 0,</code><br/><code> "reasoning": "Solar</code><br/><code> maximizes long-term</code><br/><code> ROI"}</code></td>
  <td>• Carbon &nbsp;&nbsp;&nbsp;: -53%<br/>• Renewable : 33%<br/>• Budget &nbsp;&nbsp;&nbsp;: $350K<br/>• <b>Score &nbsp;&nbsp;&nbsp;&nbsp;: 1.000</b></td>
</tr>
<tr>
  <td colspan="3" align="center"><i>↑ GRPO trains this policy ↑</i></td>
</tr>
</table>

<br/>

<table align="center">
<tr>
  <td align="center" width="260"><b>🎯 Score vs Heuristic</b></td>
  <td align="center" width="260"><b>💡 Key Innovation</b></td>
  <td align="center" width="260"><b>⚙️ Training</b></td>
</tr>
<tr>
  <td align="center">Medium task: <b>0.880 vs 0.847</b><br/><i>+3.9% over heuristic baseline!</i></td>
  <td align="center">No LLM judge —<br/><b>4 verifiable reward signals</b></td>
  <td align="center">150 steps, 3-stage<br/><b>A100 curriculum (~$4.50)</b></td>
</tr>
</table>

</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to expand</b></summary>

- [🗂️ Repository Structure](#️-repository-structure)
- [🏗️ Environment Design](#️-environment-design)
- [🧠 Training Pipeline](#-training-pipeline)
- [📊 Benchmark Results](#-benchmark-results)
- [🛡️ Anti-Reward-Hacking](#️-anti-reward-hacking)
- [🚀 Quick Start](#-quick-start)
- [🤖 Use the Trained Model](#-use-the-trained-model)
- [📐 API Reference](#-api-reference)
- [🔗 Links](#-links)

</details>

---

## 🗂️ Repository Structure

<details>
<summary><b>📁 Show full file tree</b></summary>

```
OPEN-ENV/
│
├── 🌐 ENVIRONMENT CORE
│   ├── env.py                    # ESG Gym environment (state, step, rewards)
│   ├── models.py                 # Pydantic schemas (Observation, Action, RewardComponents)
│   ├── tasks.py                  # 3 difficulty-level task configs
│   └── reward_functions.py       # 4 independent GRPO reward signals
│
├── 🤖 TRAINING PIPELINE
│   ├── train_rl.py               # GRPO training (Unsloth + TRL)
│   ├── dataset_builder.py        # Builds 350-sample training dataset
│   ├── train_config_a100.yaml    # A100 curriculum config (150 steps)
│   └── space_train_app.py        # Gradio UI for HF Space training
│
├── 📊 EVALUATION
│   ├── benchmark.py              # Benchmark runner (random/heuristic/LLM)
│   ├── plot_results.py           # Generates comparison charts
│   ├── EVALUATION_REPORT.md      # Full V2 A100 benchmark report
│   └── results/
│       ├── score_comparison.png
│       ├── reward_history.png
│       ├── esg_metrics.png
│       ├── baseline_random.json
│       ├── baseline_heuristic.json
│       └── trained_v2.json       # V2 A100 results (5 seeds × 3 tasks)
│
├── 🌐 API SERVER
│   ├── app.py                    # FastAPI server (OpenEnv-compliant)
│   └── server/app.py             # Uvicorn entrypoint shim
│
└── 📋 DOCUMENTATION
    ├── README.md
    ├── SUBMISSION.md
    ├── EVALUATION_REPORT.md
    └── run_benchmark.ps1          # One-click benchmark (Windows)
```

</details>

---

## 🏗️ Environment Design

### 🔭 State Space — 17 Observable Fields

<div align="center">

| Category | Fields | Description |
|:--------:|--------|-------------|
| 🌿 **Environmental** | `carbon_emissions_tons`<br/>`renewable_energy_pct`<br/>`waste_recycled_pct`<br/>`water_usage_liters` | Core ESG metrics tracked monthly |
| 🤝 **Social** | `diversity_score`<br/>`employee_satisfaction` | HR and workplace metrics |
| 🏛️ **Governance** | `compliance_violations`<br/>`audit_score` | Legal & regulatory standing |
| 💰 **Financial** | `available_budget`<br/>`monthly_costs`<br/>`total_investment` | Budget tracking |
| 📅 **Temporal** | `current_month`<br/>`quarters_completed` | Time horizon awareness |
| 🎯 **Targets** | `target_carbon_reduction_pct`<br/>`target_renewable_pct`<br/>`baseline_*` | Task-specific goals |

</div>

### 🎮 Action Space — 9 Discrete Interventions

<div align="center">

| ID | Action | Cost | Duration | Primary Effect |
|:--:|--------|:----:|:--------:|----------------|
| `0` | ☀️ **Install Solar Panels** | $150K | 6 months | +Renewable energy % |
| `1` | 🏢 **Upgrade HVAC** | $80K | 12 months | -Energy consumption |
| `2` | ♻️ **Recycling Program** | $25K | Permanent | +Waste recycling % |
| `3` | 💧 **Water Recycling** | $60K | 12 months | -Water usage |
| `4` | 🌫️ **Carbon Offset Credits** | $40K | Immediate | -Carbon emissions |
| `5` | 👥 **Diversity Hiring** | $50K | Permanent | +Diversity score |
| `6` | 💆 **Wellness Program** | $30K | Permanent | +Employee satisfaction |
| `7` | ⚡ **Energy Audit** | $15K | Ongoing | -Energy usage |
| `8` | 💤 **No Action** | $0 | — | Budget conservation |

</div>

### 📈 3 Task Difficulty Levels

<div align="center">

```
[EASY] --------> [MEDIUM] --------> [HARD]

basic_compliance     aggressive_sustainability  carbon_neutral_excellence
-----------------    ------------------------  -------------------------
6 steps | $500K      9 steps | $750K           12 steps | $1.2M
-15% carbon          -40% carbon               -90% carbon
30% renewable        60% renewable             80% renewable
                     70% recycling             ALL metrics
```

</div>

---

## 🧠 Training Pipeline

<div align="center">

```
         DATASET (350 samples)
         +---------------------------------+
         |  60% Heuristic rollouts         |
         |  25% Random exploration         |
         |  15% Adversarial edge cases     |
         +--------------+------------------+
                        |
                        v
         +----------------------------------+
         |     LLM generates JSON action    |
         |  {"action": 0, "reasoning": ...} |
         +--------------+-------------------+
                        |
                        v
         +----------------------------------------------+
         |            REWARD FUNCTIONS                  |
         |  +---------------------+------------------+  |
         |  | env_outcome   (45%) | format_comply(25%)|  |
         |  | anti_cheat    (15%) | task_progress(15%)|  |
         |  +---------------------+------------------+  |
         +--------------+-------------------------------+
                        |
                        v
         +----------------------------------+
         |   TRL GRPOTrainer + Unsloth      |
         |   Qwen2.5-0.5B  |  bf16  |  A100 |
         |   LoRA r=16, a=32, lr=8e-6        |
         +--------------+-------------------+
                        |
                        v
         🤗 tharun5054/esg-rl-agent-grpo
```

</div>

### 📅 3-Stage A100 Curriculum

<div align="center">

```
Step 0 --------- Step 50 --------- Step 110 --------- Step 150
  |                   |                  |                  |
  |   [EASY]          |   [MEDIUM]       |   [HARD]         |
  |   basic_comp.     |   aggressive     |   carbon_neutral  |
  |   50 steps        |   60 steps       |   40 steps       |
  +-------------------+------------------+------------------+

  Cost: ~$4.50  |  Time: ~40 min  |  GPU: A100 80GB
```

</div>

---

## 📊 Benchmark Results

<div align="center">

### 🏆 V2 A100 Model vs Baselines (5 seeds × 3 tasks)

| Agent | 🟢 Easy | 🟡 Medium | 🔴 Hard | 📊 Overall |
|:-----:|:-------:|:---------:|:-------:|:----------:|
| 🎲 Random | 0.740 | 0.643 | 0.678 | 0.687 |
| 🧠 Heuristic | 1.000 | 0.847 | 0.852 | 0.900 |
| 🚀 **V2 GRPO (A100)** | **1.000** | **0.880 ✨** | **0.726** | **0.869** |

> ✨ **The V2 model BEAT the heuristic baseline on the Medium task (0.880 > 0.847)** — proving genuine RL generalization!

</div>

### 📈 Improvement Over Random Baseline

<div align="center">

```
Label       Bar                                  Pct    Range
----------- ------------------------------------ ------ ------------------
Easy        [################################]   +35.1% (0.740 -> 1.000)
Medium      [####################################] +36.8% (0.643 -> 0.880) <- BEATS HEURISTIC!
Hard        [######]                             +7.1%  (0.678 -> 0.726)
Overall     [############################]       +26.5% (0.687 -> 0.869)
```

</div>

---

## 🛡️ Anti-Reward-Hacking

Following **OpenEnv Hackathon Guide §8** requirements — all checks are implemented:

<div align="center">

| # | Measure | Implementation |
|:-:|---------|----------------|
| ✅ | **12 independent reward signals** | No monolithic reward function |
| ✅ | **NO_ACTION spam detection** | Anti-cheat penalty for repeated no-ops |
| ✅ | **Action repetition penalty** | Penalizes cheap action loops |
| ✅ | **Budget bankruptcy** | Severe negative reward at $0 balance |
| ✅ | **Per-step timeout** | 10s wall-clock limit per API step |
| ✅ | **Thread-safe sessions** | UUID per session, zero shared state |
| ✅ | **Deterministic graders** | Same input → identical score, always |
| ✅ | **Format compliance** | Forces valid `{"action": int, "reasoning": str}` |

</div>

---

## 🚀 Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/TharunBabu-05/OPEN-ENV
cd OPEN-ENV
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Smoke Test (No GPU needed)

```bash
python train_rl.py --smoke_test
# Expected: Runs 2 steps, prints reward, exits cleanly
```

### 3️⃣ Watch the Agent Play

```bash
python demo_script.py
# Expected: Live terminal walkthrough of one episode
```

### 4️⃣ Run Benchmarks

```bash
# Run all baselines + LLM agent and generate plots
.\run_benchmark.ps1                     # Windows
# Or step by step:
python benchmark.py --mode random    --output results/baseline_random.json
python benchmark.py --mode heuristic --output results/baseline_heuristic.json
python benchmark.py --mode llm --model_path tharun5054/esg-rl-agent-grpo --output results/trained.json
python plot_results.py --random results/baseline_random.json \
                       --baseline results/baseline_heuristic.json \
                       --trained results/trained.json --output_dir results
```

### 5️⃣ Start the Environment API

```bash
uvicorn app:app --reload --port 8000
# → http://localhost:8000/health  ✅
# → http://localhost:8000/        (environment info)
```

### 6️⃣ Train on A100 (HuggingFace Space UI)

> Go to [🚀 Training Space](https://huggingface.co/spaces/tharun5054/esg-rl-train), enter your HF token, click **Start Training**.

---

## 🤖 Use the Trained Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── Load base + adapter ───────────────────────────────────────────────────────
base      = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct")
model     = PeftModel.from_pretrained(base, "tharun5054/esg-rl-agent-grpo",
                                      subfolder="lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("tharun5054/esg-rl-agent-grpo",
                                          subfolder="lora_adapter")
model.eval()

# ── Build a prompt ────────────────────────────────────────────────────────────
prompt = """You are an ESG sustainability strategist.

Current state:
  Carbon Emissions : 1200 tons  (target: -15%)
  Renewable Energy : 12%        (target: 30%)
  Budget Available : $500,000
  Month            : 1 of 6

Choose one action (0-8) and explain your reasoning. Output JSON only."""

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128,
                            temperature=0.7, do_sample=True,
                            pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
print(response)
# → {"action": 0, "reasoning": "Installing solar panels maximizes long-term ROI..."}
```

---

## 📐 API Reference

The environment is **OpenEnv judging-compatible** and live at:
`https://tharun5054-esg-compliance-env.hf.space`

<details>
<summary><b>📖 Show all endpoints</b></summary>

```
GET  /                    → Server info + available tasks
GET  /health              → {"status": "ok", "active_sessions": N}
POST /reset               → Start a new episode session
     Body: {"task_id": "basic_compliance", "seed": 42}
     Returns: {"session_id": "uuid", "observation": {...17 fields...}}

POST /step                → Take one action
     Body: {"session_id": "uuid", "action": 0}
     Returns: {"observation": {...}, "reward": 1.23,
               "terminated": false, "truncated": false, "info": {...}}

GET  /state/{session_id}  → Current observation without stepping
DEL  /session/{session_id}→ Close and cleanup a session
```

</details>

---

## 🔗 Links

<div align="center">

| | Resource | URL |
|-|----------|-----|
| 📦 | **GitHub Repository** | [TharunBabu-05/OPEN-ENV](https://github.com/TharunBabu-05/OPEN-ENV) |
| 🤗 | **Trained Model** | [tharun5054/esg-rl-agent-grpo](https://huggingface.co/tharun5054/esg-rl-agent-grpo) |
| 🌍 | **Environment API** | [tharun5054/esg-compliance-env](https://huggingface.co/spaces/tharun5054/esg-compliance-env) |
| 🚀 | **Training Space** | [tharun5054/esg-rl-train](https://huggingface.co/spaces/tharun5054/esg-rl-train) |
| 📊 | **Evaluation Report** | [EVALUATION_REPORT.md](./EVALUATION_REPORT.md) |
| 📋 | **Submission Brief** | [SUBMISSION.md](./SUBMISSION.md) |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Made with 💚 for OpenEnv Hackathon 2026**

*MIT License © 2026 Tharun Babu*

</div>
