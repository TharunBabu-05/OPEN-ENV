---
title: ESG RL Training (A100)
emoji: 🏭
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
hardware: a100-large
---

# ESG RL Agent — A100 GRPO Training

Train an ESG sustainability RL agent using GRPO on NVIDIA A100 GPU.

## Features
- **Curriculum Learning**: Easy → Medium → Hard task progression
- **350 samples**: Heuristic (60%) + Random (25%) + Adversarial (15%)
- **Enhanced Rewards**: Waste boost + synergy bonus + anti-cheat
- **A100 Optimized**: BF16, batch 8, 1000 steps
