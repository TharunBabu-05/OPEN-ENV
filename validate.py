"""
OpenEnv Hackathon Validation Script

Tests all critical requirements for submission:
1. Environment API (reset, step, state)
2. Graders return 0.0-1.0
3. Deterministic behavior
4. Tasks are properly configured
5. No blocking errors
"""

import sys
import random
from env import ESGEnvironment
from tasks import TASKS, GRADERS, grade_task
from models import Observation


def test_environment_api():
    """Test that environment implements reset/step/state correctly."""
    print("=" * 70)
    print("TEST 1: Environment API (reset, step, state)")
    print("=" * 70)
    
    env = ESGEnvironment(TASKS['basic_compliance'], seed=42)
    
    # Test reset
    try:
        obs = env.reset()
        assert isinstance(obs, Observation), "reset() must return Observation"
        print("✓ reset() returns Observation")
    except Exception as e:
        print(f"✗ reset() failed: {e}")
        return False
    
    # Test step
    try:
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, Observation), "step() must return Observation"
        assert isinstance(reward, (int, float)), "step() must return numeric reward"
        assert isinstance(terminated, bool), "step() must return bool terminated"
        assert isinstance(truncated, bool), "step() must return bool truncated"
        assert isinstance(info, dict), "step() must return dict info"
        print("✓ step() returns (obs, reward, terminated, truncated, info)")
        print(f"  Sample reward: {reward:.3f}")
    except Exception as e:
        print(f"✗ step() failed: {e}")
        return False
    
    # Test state
    try:
        current_obs = env.state()
        assert isinstance(current_obs, Observation), "state() must return Observation"
        print("✓ state() returns Observation")
    except Exception as e:
        print(f"✗ state() failed: {e}")
        return False
    
    print("\n✓ Environment API test PASSED\n")
    return True


def test_graders():
    """Test that all graders return scores in [0.0, 1.0]."""
    print("=" * 70)
    print("TEST 2: Grader Score Range (must be 0.0-1.0)")
    print("=" * 70)
    
    # Create sample observations
    test_cases = [
        ("perfect", Observation(
            energy_consumption_kwh=1000.0,
            renewable_energy_pct=95.0,
            carbon_emissions_tons=100.0,
            waste_generated_tons=400.0,
            waste_recycled_pct=85.0,
            water_usage_cubic_m=10000.0,
            diversity_score=95.0,
            employee_satisfaction=95.0,
            available_budget=200000.0,
            monthly_costs=30000.0,
            compliance_violations=0,
            audit_score=95.0,
            current_month=12,
            quarters_completed=4,
            target_carbon_reduction_pct=90.0,
            target_renewable_pct=80.0,
            target_diversity_score=85.0,
            baseline_carbon_emissions_tons=2000.0,
            baseline_water_usage_cubic_m=20000.0,
            actions_taken=[0, 1, 2, 3],
            total_investment=500000.0,
        )),
        ("poor", Observation(
            energy_consumption_kwh=6000.0,
            renewable_energy_pct=15.0,
            carbon_emissions_tons=1800.0,
            waste_generated_tons=550.0,
            waste_recycled_pct=30.0,
            water_usage_cubic_m=19000.0,
            diversity_score=50.0,
            employee_satisfaction=55.0,
            available_budget=100000.0,
            monthly_costs=55000.0,
            compliance_violations=5,
            audit_score=40.0,
            current_month=6,
            quarters_completed=2,
            target_carbon_reduction_pct=40.0,
            target_renewable_pct=60.0,
            target_diversity_score=75.0,
            baseline_carbon_emissions_tons=2000.0,
            baseline_water_usage_cubic_m=20000.0,
            actions_taken=[8, 8, 8],
            total_investment=50000.0,
        )),
    ]
    
    all_passed = True
    
    for task_id, grader_func in GRADERS.items():
        print(f"\nTesting grader: {task_id}")
        
        for case_name, obs in test_cases:
            try:
                score = grader_func(obs)
                
                if not isinstance(score, (int, float)):
                    print(f"  ✗ {case_name}: Grader must return numeric score, got {type(score)}")
                    all_passed = False
                    continue
                
                if score < 0.0 or score > 1.0:
                    print(f"  ✗ {case_name}: Score {score:.3f} outside valid range [0.0, 1.0]")
                    all_passed = False
                else:
                    print(f"  ✓ {case_name}: {score:.3f} (valid)")
                    
            except Exception as e:
                print(f"  ✗ {case_name}: Grader raised exception: {e}")
                all_passed = False
    
    if all_passed:
        print("\n✓ Grader test PASSED\n")
    else:
        print("\n✗ Grader test FAILED\n")
    
    return all_passed


def test_determinism():
    """Test that environment behavior is deterministic with same seed."""
    print("=" * 70)
    print("TEST 3: Deterministic Behavior (same seed = same results)")
    print("=" * 70)
    
    def run_episode(seed, task_id='basic_compliance'):
        env = ESGEnvironment(TASKS[task_id], seed=seed)
        obs = env.reset()
        
        trajectory = []
        for step in range(3):  # Short episode
            obs, reward, terminated, truncated, info = env.step(0)  # Always action 0
            trajectory.append({
                'carbon': obs.carbon_emissions_tons,
                'renewable': obs.renewable_energy_pct,
                'reward': reward,
            })
            if terminated or truncated:
                break
        
        return trajectory
    
    # Run twice with same seed
    traj1 = run_episode(seed=123)
    traj2 = run_episode(seed=123)
    
    # Run with different seed
    traj3 = run_episode(seed=456)
    
    # Check determinism
    if traj1 == traj2:
        print("✓ Same seed produces identical trajectories")
    else:
        print("✗ Same seed produces DIFFERENT trajectories (non-deterministic!)")
        print(f"  Trajectory 1: {traj1}")
        print(f"  Trajectory 2: {traj2}")
        return False
    
    # Check that different seeds differ
    if traj1 != traj3:
        print("✓ Different seeds produce different trajectories")
    else:
        print("⚠ Different seeds produce SAME trajectories (suspicious but not fatal)")
    
    print("\n✓ Determinism test PASSED\n")
    return True


def test_tasks():
    """Test that all 3 tasks are properly configured."""
    print("=" * 70)
    print("TEST 4: Task Configuration")
    print("=" * 70)
    
    required_tasks = ['basic_compliance', 'aggressive_sustainability', 'carbon_neutral_excellence']
    
    all_present = True
    for task_id in required_tasks:
        if task_id in TASKS:
            task = TASKS[task_id]
            print(f"✓ Task '{task_id}' found")
            print(f"  - Difficulty: {task.difficulty}")
            print(f"  - Max steps: {task.max_steps}")
            print(f"  - Budget: ${task.initial_budget:,.0f}")
        else:
            print(f"✗ Task '{task_id}' MISSING")
            all_present = False
    
    if all_present:
        print("\n✓ Task configuration PASSED\n")
    else:
        print("\n✗ Task configuration FAILED\n")
    
    return all_present


def test_full_episode():
    """Run a complete episode to check for runtime errors."""
    print("=" * 70)
    print("TEST 5: Full Episode Execution")
    print("=" * 70)
    
    try:
        env = ESGEnvironment(TASKS['basic_compliance'], seed=42)
        obs = env.reset()
        
        print(f"Starting episode (max_steps={TASKS['basic_compliance'].max_steps})")
        
        for step in range(TASKS['basic_compliance'].max_steps):
            # Simple policy: alternate between action 0 and 8
            action = 0 if step % 2 == 0 else 8
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {step + 1}: action={action}, reward={reward:.3f}, "
                  f"carbon={obs.carbon_emissions_tons:.0f}, "
                  f"renewable={obs.renewable_energy_pct:.1f}%")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
        
        # Grade final state
        score = grade_task('basic_compliance', obs)
        print(f"\nFinal score: {score:.3f}")
        
        if 0.0 <= score <= 1.0:
            print("✓ Final score in valid range")
        else:
            print(f"✗ Final score {score:.3f} outside [0.0, 1.0]")
            return False
        
        print("\n✓ Full episode test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Full episode test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_validation():
    """Test that environment handles all valid actions."""
    print("=" * 70)
    print("TEST 6: Action Validation (all 9 actions)")
    print("=" * 70)
    
    env = ESGEnvironment(TASKS['basic_compliance'], seed=42)
    
    all_passed = True
    for action_id in range(9):
        try:
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action_id)
            print(f"✓ Action {action_id}: executed successfully, reward={reward:.3f}")
        except Exception as e:
            print(f"✗ Action {action_id}: raised exception: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ Action validation PASSED\n")
    else:
        print("\n✗ Action validation FAILED\n")
    
    return all_passed


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("OPENENV HACKATHON VALIDATION")
    print("ESG Compliance & Sustainability Environment")
    print("=" * 70 + "\n")
    
    tests = [
        ("Environment API", test_environment_api),
        ("Grader Scores", test_graders),
        ("Determinism", test_determinism),
        ("Task Configuration", test_tasks),
        ("Full Episode", test_full_episode),
        ("Action Validation", test_action_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED - READY FOR SUBMISSION ✓✓✓")
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED - FIX BEFORE SUBMISSION ✗✗✗")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
