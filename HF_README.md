---
license: mit
base_model: unsloth/Qwen2.5-0.5B-Instruct
tags:
  - reinforcement-learning
  - grpo
  - esg
  - sustainability
  - openenv
  - lora
  - unsloth
  - curriculum-learning
  - decision-making
language:
  - en
datasets:
  - custom
pipeline_tag: text-generation
library_name: peft
---

<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=ESG%20RL%20Agent&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=38&desc=A100%20GRPO-Trained%20%7C%20Qwen2.5-0.5B-Instruct&descAlignY=62&descSize=18" width="100%"/>

<!-- Animated Typing -->
<a href="https://github.com/TharunBabu-05/OPEN-ENV">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=00D9FF&center=true&vCenter=true&width=650&lines=🌍+Corporate+ESG+Sustainability+Strategist;🤖+GRPO+%2B+Curriculum+Learning+%2B+A100;📈+Beats+Heuristic+Baseline+on+Medium+Task!" alt="Typing SVG" />
</a>

<br/>

<p>
  <a href="https://github.com/TharunBabu-05/OPEN-ENV">
    <img src="https://img.shields.io/badge/GitHub-OPEN--ENV-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/spaces/tharun5054/esg-rl-train">
    <img src="https://img.shields.io/badge/🚀_Training_Space-HuggingFace-blue?style=for-the-badge" alt="Training Space">
  </a>
  <a href="https://huggingface.co/spaces/tharun5054/esg-compliance-env">
    <img src="https://img.shields.io/badge/🌍_Env_API-Live-22c55e?style=for-the-badge" alt="Env API">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative" alt="MIT">
</p>

</div>

---

## 🎯 What Is This Model?

This is a **LoRA adapter** on top of `unsloth/Qwen2.5-0.5B-Instruct`, trained with **GRPO (Group Relative Policy Optimization)** to solve long-horizon ESG corporate decision-making.

The agent acts as a **corporate sustainability strategist**, choosing monthly interventions to hit ESG targets within a fixed budget. The challenge: actions have **delayed compound effects** (e.g. solar panels take months to pay off), requiring multi-step reasoning.

```
 INPUT STATE                      MODEL OUTPUT
 ────────────────────────────── ──────────────────────────────────────
 Carbon Emissions : 1200 tons  {
 Renewable Energy : 12%           "action": 0,
 Available Budget : $500K          "reasoning": "Installing solar now
 Month            : 1 of 6         gives 6 months of renewable boost.
                                   Best ROI for our 30% target."
                               }
```

---

## 🏆 Benchmark Performance

<div align="center">

| Agent | 🟢 Easy | 🟡 Medium | 🔴 Hard | 📊 Overall |
|:-----:|:-------:|:---------:|:-------:|:----------:|
| 🎲 Random Baseline | 0.740 | 0.643 | 0.678 | 0.687 |
| 🧠 Heuristic Baseline | 1.000 | 0.847 | 0.852 | 0.900 |
| 🚀 **This Model (V2 GRPO)** | **1.000** | **0.880 ✨** | **0.726** | **0.869** |

</div>

> **✨ Key Achievement:** This model **outperformed the hand-coded heuristic** on the Medium task (0.880 vs 0.847), demonstrating genuine zero-shot RL generalization — not just memorization.

### 📈 Gains Over Random

```
Label       Performance Increase                   Pct     Range
----------- -------------------------------------- ------- --------------------
Easy        ██████████████████████████████████░░░░  +35.1% (0.740 → 1.000)
Medium      ████████████████████████████████████░░  +36.8% (0.643 → 0.880) ← BEATS HEURISTIC!
Hard        ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  + 7.1% (0.678 → 0.726)
Overall     ██████████████████████████░░░░░░░░░░░░  +26.5% (0.687 → 0.869)
```

---

## ⚙️ Training Configuration

<div align="center">

| Parameter | Value |
|-----------|:-----:|
| **Base Model** | `unsloth/Qwen2.5-0.5B-Instruct` |
| **Method** | GRPO (Group Relative Policy Optimization) |
| **Framework** | TRL + Unsloth |
| **Hardware** | NVIDIA A100 80GB |
| **Precision** | bf16 |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Learning Rate** | 8e-6 |
| **Batch Size** | 4 per device |
| **GRPO Generations** | 6 per prompt |
| **Max New Tokens** | 192 |
| **Total Steps** | 150 |
| **Curriculum** | 50 easy → 60 medium → 40 hard |
| **Dataset Size** | 350 samples (heuristic + random + adversarial) |
| **Training Time** | ~40 minutes |
| **Training Cost** | ~$4.50 |

</div>

### 📅 3-Stage Curriculum

```
Step 0 --------- Step 50 --------- Step 110 --------- Step 150
  |                  |                  |                  |
  |   [EASY]         |   [MEDIUM]       |   [HARD]         |
  |   50 steps       |   60 steps       |   40 steps       |
  +------------------+------------------+------------------+
```

---

## 🧪 Reward System (4 Independent Signals — No LLM Judge)

<div align="center">

| Signal | Weight | Type | Description |
|--------|:------:|:----:|-------------|
| ⚙️ **Environment Outcome** | 45% | Shaped | Reward from stepping the ESG simulator |
| 📋 **Format Compliance** | 25% | Binary | Valid `{"action": int, "reasoning": str}` JSON |
| 🛡️ **Anti-Cheat** | 15% | Penalty | Penalizes NO_ACTION spam & action repetition |
| 📈 **Task Progress** | 15% | Terminal | Final deterministic grader score |

</div>

All reward functions are **verifiable, rule-based, and LLM-free**. No GPT-as-judge.

---

## ⚡ Quick Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── 1. Load model ─────────────────────────────────────────────────
base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(
    base, "tharun5054/esg-rl-agent-grpo", subfolder="lora_adapter"
)
tokenizer = AutoTokenizer.from_pretrained(
    "tharun5054/esg-rl-agent-grpo", subfolder="lora_adapter"
)
model.eval()

# ── 2. Build ESG scenario ─────────────────────────────────────────
prompt = """You are an ESG sustainability strategist. Current state:
- Carbon Emissions: 1200 tons/month (target: -15%)
- Renewable Energy: 12% (target: 30%)
- Available Budget: $500,000
- Month: 1 of 6

Actions: 0=Solar($150K,6mo), 1=HVAC($80K,12mo), 2=Recycling($25K),
         3=Water($60K,12mo), 4=Carbon Offset($40K), 5=Diversity($50K),
         6=Wellness($30K), 7=Audit($15K), 8=NoAction

Output JSON only:"""

# ── 3. Generate ───────────────────────────────────────────────────
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=128,
        temperature=0.7, do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
# → {"action": 0, "reasoning": "Solar panels provide 6 months of renewable energy boost..."}
```

---

## 📂 Repository Structure

<details>
<summary><b>Click to see all files</b></summary>

```
tharun5054/esg-rl-agent-grpo/
│
├── 🔧 lora_adapter/
│   ├── adapter_config.json          ← LoRA config (rank=16, alpha=32)
│   ├── adapter_model.safetensors    ← Trained weights (~35MB)
│   ├── tokenizer.json               ← Qwen2.5 tokenizer
│   └── tokenizer_config.json
│
├── 📊 results/
│   ├── trained_v2.json              ← Full benchmark (5 seeds × 3 tasks)
│   ├── baseline_heuristic.json      ← Heuristic baseline
│   ├── baseline_random.json         ← Random baseline
│   ├── score_comparison.png         ← Bar chart comparison
│   ├── reward_history.png           ← Reward curves per step
│   └── esg_metrics.png              ← Final ESG metric breakdown
│
├── 📋 EVALUATION_REPORT.md          ← Detailed V2 analysis
└── 📋 SUBMISSION.md                 ← Hackathon submission brief
```

</details>

---

## 🌐 Related Links

<div align="center">

| | Resource | Link |
|-|----------|------|
| 📦 | **Source Code** | [TharunBabu-05/OPEN-ENV](https://github.com/TharunBabu-05/OPEN-ENV) |
| 🚀 | **Training Space** | [tharun5054/esg-rl-train](https://huggingface.co/spaces/tharun5054/esg-rl-train) |
| 🌍 | **Live Env API** | [tharun5054/esg-compliance-env](https://huggingface.co/spaces/tharun5054/esg-compliance-env) |
| 📊 | **Evaluation Report** | `EVALUATION_REPORT.md` |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**MIT License © 2026 Tharun Babu | OpenEnv Hackathon 2026**

</div>
