"""
HuggingFace Space: A100 GRPO Training for ESG RL Agent

This Space runs GRPO training on A100 GPU and uploads results to HuggingFace.
"""
import gradio as gr
import subprocess
import sys
import os
import threading
import time
import json
from pathlib import Path

# Global state
training_log = []
training_status = "idle"
training_thread = None


def run_training(hf_token: str, progress=gr.Progress()):
    """Run A100 GRPO training with live logging."""
    global training_log, training_status

    training_log = []
    training_status = "running"

    training_log.append("=" * 60)
    training_log.append("ESG RL AGENT - A100 GRPO TRAINING")
    training_log.append("=" * 60)

    # Step 1: Setup
    training_log.append("\n[1/6] Installing dependencies...")
    yield get_log_text(), "⏳ Installing..."

    deps = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "unsloth", "trl", "peft", "datasets", "pyyaml", "weave",
         "--upgrade", "huggingface_hub"],
        capture_output=True, text=True
    )
    training_log.append("Dependencies installed.")

    # Step 2: Patch imports
    training_log.append("\n[2/6] Patching module compatibility...")
    yield get_log_text(), "⏳ Patching..."

    import importlib, importlib.util, importlib.machinery, types
    _orig = importlib.util.find_spec
    def _safe(name, package=None):
        try: return _orig(name, package)
        except ValueError: return None
    importlib.util.find_spec = _safe

    mod = types.ModuleType("llm_blender")
    mod.__spec__ = importlib.machinery.ModuleSpec("llm_blender", None)
    mod.__version__ = "0.0.0"
    mod.__path__ = []
    sys.modules["llm_blender"] = mod

    training_log.append("Patches applied.")

    # Step 3: Build dataset
    training_log.append("\n[3/6] Building expanded dataset (350 samples)...")
    yield get_log_text(), "⏳ Building dataset..."

    sys.path.insert(0, str(Path(__file__).parent))
    import yaml
    with open("train_config_a100.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    from train_rl_a100 import build_expanded_dataset
    samples = build_expanded_dataset(cfg)
    training_log.append(f"Dataset ready: {len(samples)} samples")

    # Step 4: Run training
    training_log.append("\n[4/6] Starting GRPO training (1000 steps, 3 stages)...")
    training_log.append("  Stage 1: basic_compliance (300 steps)")
    training_log.append("  Stage 2: aggressive_sustainability (400 steps)")
    training_log.append("  Stage 3: carbon_neutral_excellence (300 steps)")
    yield get_log_text(), "🚀 Training started..."

    try:
        from train_rl_a100 import train_a100
        train_a100(cfg)
        training_log.append("\n✅ Training complete!")
    except Exception as e:
        training_log.append(f"\n❌ Training error: {e}")
        import traceback
        training_log.append(traceback.format_exc())
        training_status = "error"
        yield get_log_text(), "❌ Error"
        return get_log_text(), "❌ Training failed"

    # Step 5: Upload to HuggingFace
    training_log.append("\n[5/6] Uploading model to HuggingFace...")
    yield get_log_text(), "⏳ Uploading..."

    try:
        if hf_token:
            from huggingface_hub import HfApi, create_repo
            REPO = "tharun5054/esg-rl-agent-grpo"
            api = HfApi(token=hf_token)
            create_repo(REPO, exist_ok=True, private=False, token=hf_token)

            # Upload new A100 adapter
            adapter_path = "outputs/grpo_step_1000_a100_new/lora_adapter"
            if os.path.isdir(adapter_path):
                api.upload_folder(
                    folder_path=adapter_path,
                    repo_id=REPO, repo_type="model",
                    path_in_repo="lora_adapter_a100",
                )
                training_log.append("A100 LoRA adapter uploaded!")

            # Upload config
            config_path = "outputs/grpo_step_1000_a100_new/train_config_used.json"
            if os.path.exists(config_path):
                api.upload_file(
                    path_or_fileobj=config_path,
                    path_in_repo="a100_train_config.json",
                    repo_id=REPO, repo_type="model",
                )
            training_log.append(f"Model uploaded to: https://huggingface.co/{REPO}")
        else:
            training_log.append("No HF token provided, skipping upload.")
    except Exception as e:
        training_log.append(f"Upload error: {e}")

    # Step 6: Summary
    training_log.append("\n[6/6] Training Summary")
    training_log.append("=" * 60)
    training_log.append(f"  Total Steps: 1000 (3 curriculum stages)")
    training_log.append(f"  Dataset: {len(samples)} samples")
    training_log.append(f"  Model: outputs/grpo_step_1000_a100_new/lora_adapter")
    training_log.append("=" * 60)

    training_status = "complete"
    yield get_log_text(), "✅ Complete!"
    return get_log_text(), "✅ Training Complete!"


def get_log_text():
    return "\n".join(training_log)


# Gradio UI
with gr.Blocks(title="ESG RL Agent - A100 Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏭 ESG RL Agent — A100 GRPO Training
    
    Train an ESG sustainability agent using **GRPO** on **NVIDIA A100**.
    
    | Feature | Value |
    |---------|-------|
    | Base Model | Qwen2.5-0.5B-Instruct |
    | Method | GRPO + LoRA (4-bit) |
    | Steps | 1000 (curriculum: easy→medium→hard) |
    | Dataset | 350 samples (heuristic + random + adversarial) |
    | Hardware | NVIDIA A100 80GB |
    """)

    with gr.Row():
        token_input = gr.Textbox(
            label="HuggingFace Token",
            placeholder="hf_...",
            type="password",
            value=os.environ.get("HF_TOKEN", ""),
        )
        train_btn = gr.Button("🚀 Start Training", variant="primary", scale=0)

    status = gr.Textbox(label="Status", value="idle", interactive=False)
    log_output = gr.Textbox(label="Training Log", lines=25, max_lines=50, interactive=False)

    train_btn.click(
        fn=run_training,
        inputs=[token_input],
        outputs=[log_output, status],
    )

demo.launch()
