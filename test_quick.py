"""
Quick test script to verify core functionality.
Run this before submission to catch critical errors.
"""

from env import ESGEnvironment
from tasks import TASKS, grade_task
from models import Action

print("Testing ESG Environment...")

# Test 1: Environment creation
env = ESGEnvironment(TASKS['basic_compliance'], seed=42)
print("✓ Environment created")

# Test 2: Reset
obs = env.reset()
print(f"✓ Reset successful - Initial carbon: {obs.carbon_emissions_tons:.0f} tons")

# Test 3: Step
obs, reward, term, trunc, info = env.step(Action.INSTALL_SOLAR_PANELS)
print(f"✓ Step successful - Reward: {reward:.3f}, Renewable: {obs.renewable_energy_pct:.1f}%")

# Test 4: Grader
score = grade_task('basic_compliance', obs)
print(f"✓ Grader successful - Score: {score:.3f}")

# Test 5: All tasks exist
for task_id in ['basic_compliance', 'aggressive_sustainability', 'carbon_neutral_excellence']:
    assert task_id in TASKS, f"Missing task: {task_id}"
print("✓ All 3 tasks configured")

print("\n✓✓✓ All basic tests passed! ✓✓✓")
