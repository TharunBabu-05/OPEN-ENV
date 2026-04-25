---
title: ESG Compliance RL Environment
emoji: 🌍
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: "4.44.0"
app_file: space_app.py
pinned: true
license: mit
tags:
  - reinforcement-learning
  - esg
  - sustainability
  - openenv
  - long-horizon-planning
  - grpo
  - environment
short_description: "RL environment for ESG optimization — OpenEnv Hackathon"
---

# 🌍 ESG Compliance & Sustainability RL Environment

**OpenEnv Hackathon Submission** | Theme: Super Long-Horizon Planning

## What This Is

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a
**corporate ESG (Environmental, Social, Governance) sustainability strategist**.

Each month (step), the agent chooses from 9 strategic actions to:
- 🌱 Reduce carbon emissions
- ☀️ Increase renewable energy usage  
- 👥 Improve workforce diversity & satisfaction
- 💧 Reduce water consumption
- ♻️ Increase waste recycling

All while managing a **limited budget** over 6–12 months.

## Why This is Interesting

Unlike simple Q&A tasks, this environment requires:

1. **Sequential planning** — actions have delayed multi-month effects (solar panels compound over 6 months)
2. **Multi-objective optimization** — balance E, S, G, and financial constraints simultaneously
3. **Budget management** — going bankrupt ends the episode with a penalty
4. **Curriculum learning** — 3 difficulty levels (Easy → Medium → Hard)

## Tasks

| Task | Difficulty | Duration | Budget | Key Targets |
|------|-----------|----------|--------|-------------|
| Basic Compliance | Easy | 6 months | $500K | -15% carbon, 30% renewable |
| Aggressive Sustainability | Medium | 9 months | $750K | -40% carbon, 60% renewable, 70% waste recycling |
| Carbon Neutral Excellence | Hard | 12 months | $1M | -90% carbon, 80% renewable, ALL metrics |

## Reward System (10 Independent Signals)

- ✅ Carbon reduction progress
- ✅ Renewable energy progress  
- ✅ Diversity improvement
- ✅ Waste recycling improvement
- ✅ Water usage reduction
- ✅ Quarterly milestone bonuses
- ✅ Synergy bonus (multi-metric improvements)
- ❌ Budget penalty (overspending)
- ❌ Compliance penalty
- ❌ Anti-cheat penalty (NO_ACTION spam)

## Training Stack

```
ESGEnvironment → reward_functions.py → TRL (GRPOTrainer) → Unsloth → Trained LLM
```

## How to Use This Space

1. Select a task difficulty
2. Click "Start New Episode"
3. Choose monthly actions manually — or click "Heuristic Agent Step" to watch an automatic agent
4. Watch ESG metrics evolve and see reward breakdowns in real time

## Links

- 📁 [GitHub Repository](https://github.com/TharunBabu-05/OPEN-ENV)
- 📓 [Training Notebook (Colab)](colab_train.ipynb)
- 📊 [Benchmark Results](results/)
