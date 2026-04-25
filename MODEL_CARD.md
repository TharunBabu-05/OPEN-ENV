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
datasets:
  - custom
pipeline_tag: text-generation
library_name: peft
---

# ESG RL Agent (GRPO-Trained)

A **Qwen2.5-0.5B-Instruct** model fine-tuned with **GRPO (Group Relative Policy Optimization)** to act as a corporate ESG sustainability strategist.

## Training Details

| Parameter | Value |
|-----------|-------|
| **Base Model** | `unsloth/Qwen2.5-0.5B-Instruct` |
| **Method** | GRPO via TRL + Unsloth (4-bit LoRA) |
| **LoRA Rank** | 8 |
| **Training Steps** | 150 |
| **Training Time** | 26 minutes (T4 GPU) |
| **Task** | `basic_compliance` (ESG environment) |
| **Reward Signals** | 4 independent verifiable functions |

## What It Does

The model makes monthly ESG investment decisions (solar panels, HVAC upgrades, diversity hiring, etc.) to optimize a company's environmental, social, and governance metrics under budget constraints.

### Input Format
The model receives an observation containing 17 fields (energy usage, carbon emissions, budget, diversity score, etc.) and must output:

```json
{"action": 0, "reasoning": "Installing solar panels to boost renewable energy percentage toward the 30% target."}
```

### Actions (9 discrete)
| ID | Action | Cost |
|----|--------|------|
| 0 | Install Solar Panels | $150K |
| 1 | Upgrade HVAC | $80K |
| 2 | Recycling Program | $25K |
| 3 | Water Recycling | $60K |
| 4 | Carbon Offset | $40K |
| 5 | Diversity Hiring | $50K |
| 6 | Wellness Program | $30K |
| 7 | Energy Audit | $15K |
| 8 | No Action | $0 |

## Reward System

4 independent, verifiable reward signals (no LLM-as-judge):

1. **Environment Outcome** (60%) — Shaped reward from stepping the ESG simulator
2. **Format Compliance** (20%) — Valid JSON with action + reasoning
3. **Anti-Cheat** (10%) — Penalizes NO_ACTION spam and action repetition
4. **Task Progress** (10%) — Terminal grader score

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "tharun5054/esg-rl-agent-grpo/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("tharun5054/esg-rl-agent-grpo/lora_adapter")
```

## Links

- **GitHub:** [TharunBabu-05/OPEN-ENV](https://github.com/TharunBabu-05/OPEN-ENV)
- **Live Space:** [tharun5054/esg-compliance-env](https://huggingface.co/spaces/tharun5054/esg-compliance-env)
- **Training Notebook:** [Colab](https://colab.research.google.com/github/TharunBabu-05/OPEN-ENV/blob/main/colab_train.ipynb)

## Benchmark Results

See `results/` folder for comparison plots (baseline heuristic vs trained model).

## License

MIT
