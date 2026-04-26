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

# 🌍 ESG RL Agent — A100 GRPO-Trained

**A Qwen2.5-0.5B-Instruct model fine-tuned with GRPO to act as a corporate ESG sustainability strategist**

[![GitHub](https://img.shields.io/badge/GitHub-OPEN--ENV-181717?logo=github)](https://github.com/TharunBabu-05/OPEN-ENV)
[![Training Space](https://img.shields.io/badge/%F0%9F%A4%97-Training%20Space-blue)](https://huggingface.co/spaces/tharun5054/esg-rl-train)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/TharunBabu-05/OPEN-ENV/blob/main/LICENSE)

</div>

---

## 🎯 Model Description

This model is a **LoRA adapter** trained on top of `unsloth/Qwen2.5-0.5B-Instruct` using **GRPO (Group Relative Policy Optimization)** to solve long-horizon ESG corporate decision-making tasks.

The agent acts as a **corporate sustainability strategist** for a fictional company, choosing from 9 monthly interventions (solar panels, HVAC upgrades, diversity hiring programs, etc.) to hit ESG targets within a fixed budget and time horizon.

The challenge: actions have **delayed compound effects** (solar panels take months to pay off), requiring genuine multi-step reasoning rather than greedy one-step optimization.

---

## 🏆 Benchmark Results (V2 A100, 5 seeds per task)

| Agent | Easy Task | Medium Task | Hard Task | Overall |
|-------|:---------:|:-----------:|:---------:|:-------:|
| 🎲 Random | 0.740 | 0.643 | 0.678 | 0.687 |
| 🧠 Heuristic Baseline | 1.000 | 0.847 | 0.852 | 0.900 |
| 🚀 **This Model (V2 GRPO)** | **1.000** | **0.880** | **0.726** | **0.869** |

> **Key highlight:** This model **outperformed the hand-coded heuristic** on the Medium task (0.880 vs 0.847), demonstrating genuine zero-shot generalization of ESG reasoning strategies learned through RL.

---

## 🏗️ Training Details

| Parameter | Value |
|-----------|-------|
| **Base Model** | `unsloth/Qwen2.5-0.5B-Instruct` |
| **Method** | GRPO (Group Relative Policy Optimization) |
| **Framework** | TRL + Unsloth (4-bit QLoRA) |
| **Hardware** | NVIDIA A100 80GB |
| **Precision** | bf16 |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Learning Rate** | 8e-6 |
| **Batch Size** | 4 (per device) |
| **GRPO Generations** | 6 per prompt |
| **Max New Tokens** | 192 |
| **Training Steps** | 150 |
| **Curriculum** | 50 easy → 60 medium → 40 hard |
| **Dataset Size** | 350 samples (heuristic + random + adversarial) |
| **Wall Clock Time** | ~40 minutes |
| **Training Cost** | ~$4.50 |

---

## 🧠 Reward System (4 Independent Signals — No LLM Judge)

| Signal | Weight | Description |
|--------|:------:|-------------|
| **Environment Outcome** | 45% | Shaped reward from stepping the ESG simulator |
| **Format Compliance** | 25% | Valid JSON with `action` (int) + `reasoning` (str) |
| **Anti-Cheat Penalty** | 15% | Penalizes `NO_ACTION` spam and action repetition |
| **Task Progress** | 15% | Terminal score from deterministic grader |

All reward functions are **verifiable, rule-based, and LLM-free** — no GPT-as-judge.

---

## ⚡ Quick Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model + adapter
base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "tharun5054/esg-rl-agent-grpo", subfolder="lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("tharun5054/esg-rl-agent-grpo", subfolder="lora_adapter")
model.eval()

# ESG scenario prompt
prompt = """You are an ESG sustainability strategist. Current state:
- Carbon Emissions: 1200 tons/month (target: reduce by 15%)
- Renewable Energy: 12% (target: 30%)
- Available Budget: $500,000
- Month: 1 of 6

Available actions:
0: Install Solar Panels ($150K, +renewable for 6 months)
1: Upgrade HVAC ($80K, -energy for 12 months)
2: Recycling Program ($25K, +waste recycling)
3: Water Recycling ($60K, -water usage)
4: Carbon Offset ($40K, -carbon immediate)
5: Diversity Hiring ($50K, +diversity)
6: Wellness Program ($30K, +satisfaction)
7: Energy Audit ($15K, -energy ongoing)
8: No Action ($0)

Choose ONE action and explain your reasoning. Output as JSON:"""

inputs = tokenizer(prompt, return_tensors="pt")
with __import__('torch').no_grad():
    out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# Expected: {"action": 0, "reasoning": "Installing solar panels now gives 6 months of renewable energy boost..."}
```

---

## 📂 Repository Contents

```
tharun5054/esg-rl-agent-grpo/
├── lora_adapter/
│   ├── adapter_config.json          # LoRA config (rank=16, alpha=32)
│   ├── adapter_model.safetensors    # Trained weights (~35MB)
│   ├── tokenizer.json               # Qwen2.5 tokenizer
│   └── tokenizer_config.json
├── results/
│   ├── trained_v2.json              # Full benchmark results (5 seeds × 3 tasks)
│   ├── baseline_heuristic.json      # Heuristic baseline scores
│   ├── baseline_random.json         # Random baseline scores
│   ├── score_comparison.png         # Bar chart comparison
│   ├── reward_history.png           # Reward per step curves
│   └── esg_metrics.png              # Final ESG metric heatmap
├── EVALUATION_REPORT.md             # Full evaluation with analysis
└── SUBMISSION.md                    # Hackathon submission brief
```

---

## 🔗 Related Resources

| Resource | Link |
|----------|------|
| 📦 **GitHub Repo** | [TharunBabu-05/OPEN-ENV](https://github.com/TharunBabu-05/OPEN-ENV) |
| 🚀 **Training Space** | [tharun5054/esg-rl-train](https://huggingface.co/spaces/tharun5054/esg-rl-train) |
| 📊 **Evaluation Report** | `EVALUATION_REPORT.md` (in this repo) |

---

## ⚖️ Ethical Considerations

This model is designed for **educational and research purposes** — simulating ESG decision-making in a controlled environment. Real-world ESG investment decisions should always involve domain experts, regulatory compliance review, and stakeholder consultation.

---

## 📜 License

MIT © 2026 Tharun Babu
