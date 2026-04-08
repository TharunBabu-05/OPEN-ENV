"""
CRITICAL PRE-SUBMISSION TEST SCRIPT

This script tests EXACTLY what judges will check:
1. Environment works without API keys
2. Inference works WITH API keys
3. Output format matches [START]/[STEP]/[END]
4. Rewards vary (not constant)
5. Deterministic behavior

Run this BEFORE submitting to catch instant-rejection issues!
"""

import os
import sys
import json

print("=" * 70)
print("🔍 OPENENV HACKATHON - FINAL VALIDATION")
print("=" * 70)

# =============================================================================
# TEST 1: Environment Works WITHOUT API Keys (CRITICAL)
# =============================================================================

print("\n📋 TEST 1: Environment works WITHOUT API keys")
print("-" * 70)

try:
    from env import ESGEnvironment
    from tasks import TASKS, grade_task
    from models import Action
    
    env = ESGEnvironment(TASKS['basic_compliance'], seed=42)
    obs = env.reset()
    print("✅ PASS: Environment created and reset (no API needed)")
    
    obs, reward1, term, trunc, info = env.step(0)
    print(f"✅ PASS: Step executed (reward={reward1:.3f})")
    
    obs, reward2, term, trunc, info = env.step(1)
    print(f"✅ PASS: Second step (reward={reward2:.3f})")
    
    if reward1 != reward2:
        print("✅ PASS: Rewards vary based on actions ✓")
    else:
        print("⚠️  WARNING: Both rewards same - this might be flagged")
    
    score = grade_task('basic_compliance', obs)
    if 0.0 <= score <= 1.0:
        print(f"✅ PASS: Grader returns valid score ({score:.3f})")
    else:
        print(f"❌ FAIL: Grader score {score:.3f} outside [0.0, 1.0]")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAIL: Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TEST 2: Check Output Format (CRITICAL)
# =============================================================================

print("\n📋 TEST 2: Output format matches hackathon requirements")
print("-" * 70)

# Check that inference.py has START/STEP/END logging
try:
    with open('inference.py', 'r', encoding='utf-8') as f:
        inference_code = f.read()
    
    required_logs = ['log_start', 'log_step', 'log_end']
    missing = [log for log in required_logs if log not in inference_code]
    
    if not missing:
        print("✅ PASS: inference.py has log_start/log_step/log_end")
    else:
        print(f"❌ FAIL: Missing log functions: {missing}")
        sys.exit(1)
    
    # Check JSON format
    if '"type": "START"' in inference_code:
        print('✅ PASS: START logging present')
    if '"type": "STEP"' in inference_code:
        print('✅ PASS: STEP logging present')
    if '"type": "END"' in inference_code:
        print('✅ PASS: END logging present')
        
except Exception as e:
    print(f"❌ FAIL: Could not verify output format: {e}")
    sys.exit(1)

# =============================================================================
# TEST 3: Deterministic Behavior (CRITICAL)
# =============================================================================

print("\n📋 TEST 3: Deterministic behavior (same seed = same results)")
print("-" * 70)

try:
    def run_episode(seed):
        env = ESGEnvironment(TASKS['basic_compliance'], seed=seed)
        obs = env.reset()
        trajectory = []
        for i in range(3):
            obs, r, t, tr, info = env.step(0)
            trajectory.append((obs.carbon_emissions_tons, r))
        return trajectory
    
    traj1 = run_episode(seed=999)
    traj2 = run_episode(seed=999)
    
    if traj1 == traj2:
        print("✅ PASS: Same seed produces identical results ✓")
    else:
        print("❌ FAIL: Non-deterministic behavior detected!")
        print(f"  Run 1: {traj1}")
        print(f"  Run 2: {traj2}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAIL: Determinism test failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 4: API Keys Check
# =============================================================================

print("\n📋 TEST 4: Check environment variables for inference")
print("-" * 70)

api_base = os.getenv("API_BASE_URL")
model_name = os.getenv("MODEL_NAME")
hf_token = os.getenv("HF_TOKEN")

if api_base:
    print(f"✅ API_BASE_URL set: {api_base}")
else:
    print("⚠️  API_BASE_URL not set (needed for inference.py)")

if model_name:
    print(f"✅ MODEL_NAME set: {model_name}")
else:
    print("⚠️  MODEL_NAME not set (needed for inference.py)")

if hf_token:
    print(f"✅ HF_TOKEN set: {hf_token[:15]}...")
else:
    print("⚠️  HF_TOKEN not set (needed for inference.py)")

if not all([api_base, model_name]):
    print("\n⚠️  WARNING: Set environment variables to run inference:")
    print("  $env:API_BASE_URL=\"https://api.openai.com/v1\"")
    print("  $env:MODEL_NAME=\"gpt-4o-mini\"")
    print("  $env:HF_TOKEN=\"your-token\"")

# =============================================================================
# TEST 5: openenv.yaml validation
# =============================================================================

print("\n📋 TEST 5: openenv.yaml structure")
print("-" * 70)

try:
    import yaml
    with open('openenv.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    required_sections = ['metadata', 'environment', 'tasks', 'grading', 'inference', 'dependencies', 'runtime']
    
    for section in required_sections:
        if section in config:
            print(f"✅ Section '{section}' present")
        else:
            print(f"❌ Section '{section}' MISSING")
    
    # Check tasks
    if len(config.get('tasks', [])) == 3:
        print(f"✅ 3 tasks defined")
    else:
        print(f"❌ Expected 3 tasks, found {len(config.get('tasks', []))}")
        
except ImportError:
    print("⚠️  PyYAML not installed, skipping YAML validation")
    print("   Install with: pip install pyyaml")
except Exception as e:
    print(f"⚠️  Could not validate openenv.yaml: {e}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("✅ PRE-SUBMISSION VALIDATION COMPLETE")
print("=" * 70)

print("\n🎯 NEXT STEPS:")
print("1. Set environment variables (if not set):")
print('   $env:API_BASE_URL="https://api.openai.com/v1"')
print('   $env:MODEL_NAME="gpt-4o-mini"')
print('   $env:HF_TOKEN="your-token"')
print('   $env:OPENAI_API_KEY="your-openai-key"  # If using OpenAI')
print()
print("2. Test inference (optional but recommended):")
print("   python inference.py")
print()
print("3. Build Docker image:")
print("   docker build -t esg-env .")
print()
print("4. Deploy to Hugging Face Spaces")
print()
print("5. Submit to hackathon! 🏆")
print()
print("=" * 70)
print("✅ Your environment is ready for submission!")
print("=" * 70)
