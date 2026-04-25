# 📋 SUBMISSION BRIEF — ESG Compliance RL Environment

**OpenEnv Hackathon | April 2026**

---

## 1. Problem Statement

Corporate ESG (Environmental, Social, Governance) reporting requires executives to make **sequential, multi-objective decisions** under budget constraints — weighing short-term vs. long-term trade-offs across months or years. This is exactly the kind of task where LLMs with RL training can outperform both rule-based systems and untrained base models.

We built an environment that simulates this: an AI agent acts as a **corporate sustainability strategist**, choosing monthly interventions (solar panels, HVAC upgrades, diversity hiring, etc.) to hit ESG targets within a fixed budget and time horizon.

---

## 2. Theme Selection

**Primary: Theme #2 — Super Long-Horizon Planning & Instruction Following**

- Episodes span 6–12 sequential decision steps (months)
- Actions have **delayed compound effects** (solar panels improve renewable % for 6 months)
- The agent must reason about budget exhaustion, quarterly milestone gates, and compounding benefits
- Greedy strategies fail — only multi-step planning succeeds

**Secondary: Theme #3.1 — World Modeling, Professional Tasks**

- Simulates a realistic corporate ESG system with dynamic state evolution
- Seasonal energy variation, compliance tracking, and budget management
- Matches complexity of professional decision-making tools

---

## 3. Environment Design

### What the agent observes (17 fields):
- Environmental: energy (kWh), renewable %, carbon emissions, waste recycled %, water usage
- Social: diversity score, employee satisfaction
- Governance: compliance violations, audit score
- Financial: available budget, monthly costs, total investment
- Temporal: current month, quarters completed
- Task: targets for each metric, baseline values

### Actions (9 discrete):
| ID | Action | Cost | Effect |
|----|--------|------|--------|
| 0 | Install Solar Panels | $150K | +renewable energy (6 months) |
| 1 | Upgrade HVAC | $80K | -energy consumption (12 months) |
| 2 | Recycling Program | $25K | +waste recycling % |
| 3 | Water Recycling | $60K | -water usage (12 months) |
| 4 | Carbon Offset | $40K | -carbon (immediate) |
| 5 | Diversity Hiring | $50K | +diversity score |
| 6 | Wellness Program | $30K | +employee satisfaction |
| 7 | Energy Audit | $15K | -energy (ongoing) |
| 8 | No Action | $0 | Budget conservation |

### Tasks (3 difficulty levels):
| Task | Difficulty | Steps | Budget | Targets |
|------|-----------|-------|--------|---------|
| basic_compliance | Easy | 6 | $500K | -15% carbon, 30% renewable |
| aggressive_sustainability | Medium | 9 | $750K | -40% carbon, 60% renewable, 70% recycling |
| carbon_neutral_excellence | Hard | 12 | $1M | -90% carbon, 80% renewable, ALL metrics |

---

## 4. Reward System (10+ Independent Signals)

| Signal | Type | Purpose |
|--------|------|---------|
| Carbon reduction progress | Positive | Primary ESG goal |
| Renewable energy progress | Positive | Key ESG goal |
| Diversity progress | Positive | Social component |
| Waste recycling reward | Positive | Environmental |
| Water reduction reward | Positive | Environmental |
| Quarterly milestone bonus | Positive | On-track signal |
| Synergy bonus | Positive | Multi-metric improvement |
| Task completion reward | Terminal | Final outcome signal |
| Budget penalty | Negative | Bankruptcy prevention |
| Compliance penalty | Negative | Governance |
| Anti-cheat penalty | Negative | Reward hacking prevention |
| Format compliance | Positive | Valid JSON action signal |

---

## 5. Training Pipeline

```
Dataset Builder → ESGEnvironment → reward_functions.py → TRL GRPOTrainer → Unsloth
```

**Stack:** Unsloth (4-bit LoRA) + TRL (GRPO) + OpenEnv environment

**Command:**
```bash
python train_rl.py --config train_config.yaml --task basic_compliance --max_steps 200
```

**Smoke test (no GPU needed):**
```bash
python train_rl.py --smoke_test
```

---

## 6. Benchmark Results (Baseline)

| Agent | Easy Score | Medium Score | Hard Score | Overall |
|-------|-----------|-------------|-----------|---------|
| Random | 0.74 | 0.64 | 0.68 | **0.687** |
| Heuristic | 1.00 | 0.85 | 0.85 | **0.900** |
| Trained LLM (GRPO) | ✅ trained | — | — | **See HF repo** |

> Plots available in `results/` directory.

---

## 7. Key Links

| Asset | Link |
|-------|------|
| GitHub Repository | https://github.com/TharunBabu-05/OPEN-ENV |
| HuggingFace Space | https://huggingface.co/spaces/tharun5054/esg-compliance-env |
| Colab Training Notebook | `colab_train.ipynb` |
| Benchmark Results | `results/` |
| Demo | `python demo_script.py` |

---

## 8. Anti-Reward-Hacking Measures

Following hackathon guide §8:
- ✅ 12 independent reward signals (not one monolithic reward)
- ✅ Anti-cheat penalty for NO_ACTION spam
- ✅ Anti-cheat penalty for repeating cheap actions
- ✅ Budget bankruptcy detection (severe penalty)
- ✅ Per-step wall-clock timeout (10s) in API server
- ✅ Thread-safe session isolation (no shared mutable state)
- ✅ Deterministic graders (same input always gives same score)

---

## 9. Reproducibility

```bash
# Clone and run smoke test
git clone https://github.com/TharunBabu-05/OPEN-ENV
cd OPEN-ENV
pip install -e .
python train_rl.py --smoke_test    # No GPU needed
python demo_script.py              # See agent in action
python run_benchmark.ps1           # Generate baseline results
```
