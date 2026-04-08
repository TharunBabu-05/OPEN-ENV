"""
Task definitions and deterministic graders for ESG environment.

This module defines three difficulty levels:
1. BASIC_COMPLIANCE (Easy) - Meet minimum ESG standards
2. AGGRESSIVE_SUSTAINABILITY (Medium) - Achieve ambitious targets with budget constraints
3. CARBON_NEUTRAL_EXCELLENCE (Hard) - Multi-objective optimization for carbon neutrality

Each task includes a deterministic grader that scores performance from 0.0 to 1.0.
"""

from typing import Dict, Any
from models import TaskConfig, Observation


# ============================================================================
# TASK 1: BASIC_COMPLIANCE (Easy)
# ============================================================================

TASK_BASIC_COMPLIANCE = TaskConfig(
    task_id="basic_compliance",
    difficulty="easy",
    max_steps=6,
    initial_budget=500000.0,
    target_carbon_reduction_pct=15.0,
    target_renewable_pct=30.0,
    target_diversity_score=60.0,
    target_waste_recycling_pct=0.0,  # Not required
    target_water_reduction_pct=0.0,  # Not required
    target_employee_satisfaction=0.0,  # Not required
    max_compliance_violations=2,
    description=(
        "BASIC COMPLIANCE: Meet minimum ESG standards in 6 months. "
        "Reduce carbon by 15%, achieve 30% renewable energy, maintain "
        "diversity score above 60, and minimize compliance violations. "
        "Budget: $500,000."
    ),
)


def grade_basic_compliance(final_obs: Observation) -> float:
    """
    Grade BASIC_COMPLIANCE task performance.
    
    Scoring breakdown:
    - 40%: Carbon reduction (15% target)
    - 30%: Renewable energy (30% target)
    - 20%: Diversity score (60 target)
    - 10%: Compliance violations (≤2 target)
    
    Args:
        final_obs: Final observation after episode completion
        
    Returns:
        Score between 0.0 (complete failure) and 1.0 (perfect completion)
    """
    score = 0.0
    
    # Component 1: Carbon reduction (40% of total score)
    carbon_reduction_pct = _calculate_carbon_reduction(final_obs)
    if carbon_reduction_pct >= 15.0:
        # Full points if target met
        score += 0.40
    else:
        # Partial credit proportional to achievement
        score += 0.40 * (carbon_reduction_pct / 15.0)
    
    # Component 2: Renewable energy (30% of total score)
    renewable_pct = final_obs.renewable_energy_pct
    if renewable_pct >= 30.0:
        score += 0.30
    else:
        # Partial credit (pro-rated from initial ~10%)
        # Scale from baseline 10 to target 30
        score += 0.30 * max(0.0, (renewable_pct - 10.0) / 20.0)
    
    # Component 3: Diversity score (20% of total score)
    diversity_score = final_obs.diversity_score
    if diversity_score >= 60.0:
        score += 0.20
    else:
        # Partial credit (pro-rated from initial ~45)
        # Scale from baseline 45 to target 60
        score += 0.20 * max(0.0, (diversity_score - 45.0) / 15.0)
    
    # Component 4: Compliance violations (10% of total score)
    violations = final_obs.compliance_violations
    if violations <= 2:
        score += 0.10
    elif violations <= 3:
        score += 0.05  # Partial credit
    # else: 0 points
    
    # Ensure score is in valid range
    return max(0.0, min(1.0, score))


# ============================================================================
# TASK 2: AGGRESSIVE_SUSTAINABILITY (Medium)
# ============================================================================

TASK_AGGRESSIVE_SUSTAINABILITY = TaskConfig(
    task_id="aggressive_sustainability",
    difficulty="medium",
    max_steps=9,
    initial_budget=750000.0,
    target_carbon_reduction_pct=40.0,
    target_renewable_pct=60.0,
    target_diversity_score=75.0,
    target_waste_recycling_pct=70.0,
    target_water_reduction_pct=0.0,  # Not required
    target_employee_satisfaction=0.0,  # Not required
    max_compliance_violations=1,
    description=(
        "AGGRESSIVE SUSTAINABILITY: Achieve ambitious ESG targets in 9 months "
        "with tight budget constraints. Reduce carbon by 40%, achieve 60% "
        "renewable energy, increase waste recycling to 70%, maintain diversity "
        "above 75, and stay within budget. Budget: $750,000."
    ),
)


def grade_aggressive_sustainability(final_obs: Observation) -> float:
    """
    Grade AGGRESSIVE_SUSTAINABILITY task performance.
    
    Scoring breakdown:
    - 35%: Carbon reduction (40% target) - most critical
    - 25%: Renewable energy (60% target)
    - 20%: Waste recycling (70% target)
    - 15%: Diversity score (75 target)
    - 5%: Budget efficiency (staying in budget)
    
    This task emphasizes multi-dimensional performance with tighter constraints.
    
    Args:
        final_obs: Final observation after episode completion
        
    Returns:
        Score between 0.0 (complete failure) and 1.0 (perfect completion)
    """
    score = 0.0
    
    # Component 1: Carbon reduction (35% of total score)
    carbon_reduction_pct = _calculate_carbon_reduction(final_obs)
    carbon_achievement = min(1.0, carbon_reduction_pct / 40.0)
    score += 0.35 * carbon_achievement
    
    # Component 2: Renewable energy (25% of total score)
    renewable_pct = final_obs.renewable_energy_pct
    # Scale from baseline ~10% to target 60%
    renewable_achievement = min(1.0, max(0.0, (renewable_pct - 10.0) / 50.0))
    score += 0.25 * renewable_achievement
    
    # Component 3: Waste recycling (20% of total score)
    waste_recycled_pct = final_obs.waste_recycled_pct
    # Scale from baseline ~25% to target 70%
    waste_achievement = min(1.0, max(0.0, (waste_recycled_pct - 25.0) / 45.0))
    score += 0.20 * waste_achievement
    
    # Component 4: Diversity score (15% of total score)
    diversity_score = final_obs.diversity_score
    # Scale from baseline ~45 to target 75
    diversity_achievement = min(1.0, max(0.0, (diversity_score - 45.0) / 30.0))
    score += 0.15 * diversity_achievement
    
    # Component 5: Budget efficiency (5% of total score)
    if final_obs.available_budget >= 0:
        score += 0.05  # Full points for staying in budget
    elif final_obs.available_budget >= -50000:
        score += 0.025  # Partial credit for minor overspend
    # else: 0 points for major overspend
    
    # Penalty for excessive compliance violations
    if final_obs.compliance_violations > 1:
        penalty = 0.1 * (final_obs.compliance_violations - 1)
        score -= penalty
    
    # Ensure score is in valid range
    return max(0.0, min(1.0, score))


# ============================================================================
# TASK 3: CARBON_NEUTRAL_EXCELLENCE (Hard)
# ============================================================================

TASK_CARBON_NEUTRAL_EXCELLENCE = TaskConfig(
    task_id="carbon_neutral_excellence",
    difficulty="hard",
    max_steps=12,
    initial_budget=1000000.0,
    target_carbon_reduction_pct=90.0,
    target_renewable_pct=80.0,
    target_diversity_score=85.0,
    target_waste_recycling_pct=75.0,
    target_water_reduction_pct=30.0,
    target_employee_satisfaction=90.0,
    max_compliance_violations=0,
    description=(
        "CARBON NEUTRAL EXCELLENCE: Achieve near carbon-neutrality and ESG "
        "excellence in 12 months. Reduce carbon by 90%, achieve 80% renewable "
        "energy, 75% waste recycling, 30% water reduction, 85+ diversity score, "
        "90+ employee satisfaction, and zero compliance violations. "
        "Budget: $1,000,000."
    ),
)


def grade_carbon_neutral_excellence(final_obs: Observation) -> float:
    """
    Grade CARBON_NEUTRAL_EXCELLENCE task performance.
    
    This is the hardest task requiring excellence across ALL ESG dimensions.
    Uses strict, multiplicative scoring to ensure balanced performance.
    
    Scoring breakdown (weighted):
    - 30%: Carbon reduction (90% target) - CRITICAL
    - 25%: Renewable energy (80% target)
    - 15%: Water reduction (30% target)
    - 15%: Diversity score (85 target)
    - 10%: Employee satisfaction (90 target)
    - 5%: Waste recycling (75% target)
    
    Additional strict requirements:
    - Zero compliance violations (else severe penalty)
    - Must achieve at least 80% carbon reduction for high scores
    
    Args:
        final_obs: Final observation after episode completion
        
    Returns:
        Score between 0.0 (complete failure) and 1.0 (perfect completion)
    """
    score = 0.0
    
    # Component 1: Carbon reduction (30% of total score)
    carbon_reduction_pct = _calculate_carbon_reduction(final_obs)
    carbon_achievement = min(1.0, carbon_reduction_pct / 90.0)
    score += 0.30 * carbon_achievement
    
    # Component 2: Renewable energy (25% of total score)
    renewable_pct = final_obs.renewable_energy_pct
    # Scale from baseline ~10% to target 80%
    renewable_achievement = min(1.0, max(0.0, (renewable_pct - 10.0) / 70.0))
    score += 0.25 * renewable_achievement
    
    # Component 3: Water reduction (15% of total score)
    water_reduction_pct = _calculate_water_reduction(final_obs)
    water_achievement = min(1.0, water_reduction_pct / 30.0)
    score += 0.15 * water_achievement
    
    # Component 4: Diversity score (15% of total score)
    diversity_score = final_obs.diversity_score
    # Scale from baseline ~45 to target 85
    diversity_achievement = min(1.0, max(0.0, (diversity_score - 45.0) / 40.0))
    score += 0.15 * diversity_achievement
    
    # Component 5: Employee satisfaction (10% of total score)
    satisfaction = final_obs.employee_satisfaction
    # Scale from baseline ~60 to target 90
    satisfaction_achievement = min(1.0, max(0.0, (satisfaction - 60.0) / 30.0))
    score += 0.10 * satisfaction_achievement
    
    # Component 6: Waste recycling (5% of total score)
    waste_recycled_pct = final_obs.waste_recycled_pct
    # Scale from baseline ~25% to target 75%
    waste_achievement = min(1.0, max(0.0, (waste_recycled_pct - 25.0) / 50.0))
    score += 0.05 * waste_achievement
    
    # STRICT REQUIREMENTS
    
    # Penalty 1: Compliance violations (ZERO tolerance)
    if final_obs.compliance_violations > 0:
        # Severe penalty - violations are unacceptable
        violation_penalty = 0.3 * final_obs.compliance_violations
        score -= violation_penalty
    
    # Penalty 2: Critical threshold (carbon reduction must be at least 80%)
    if carbon_reduction_pct < 80.0:
        # Severe penalty for missing critical environmental target
        score *= 0.5  # Cut score in half
    
    # Penalty 3: Budget management (bankruptcy is failure)
    if final_obs.available_budget < -100000:
        score *= 0.3  # Severe penalty for poor financial management
    
    # Bonus: Exceptional performance (all targets exceeded)
    if (
        carbon_reduction_pct >= 90.0
        and renewable_pct >= 80.0
        and water_reduction_pct >= 30.0
        and diversity_score >= 85.0
        and satisfaction >= 90.0
        and waste_recycled_pct >= 75.0
        and final_obs.compliance_violations == 0
    ):
        score = min(1.0, score + 0.05)  # Perfect execution bonus
    
    # Ensure score is in valid range
    return max(0.0, min(1.0, score))


# ============================================================================
# Helper Functions
# ============================================================================

def _calculate_carbon_reduction(obs: Observation) -> float:
    """
    Calculate percentage reduction in carbon emissions from baseline.
    
    Args:
        obs: Current or final observation
        
    Returns:
        Percentage reduction (0-100+)
    """
    if obs.baseline_carbon_emissions_tons == 0:
        return 0.0
    
    reduction = (
        (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
        / obs.baseline_carbon_emissions_tons
        * 100.0
    )
    return max(0.0, reduction)


def _calculate_water_reduction(obs: Observation) -> float:
    """
    Calculate percentage reduction in water usage from baseline.
    
    Args:
        obs: Current or final observation
        
    Returns:
        Percentage reduction (0-100+)
    """
    if obs.baseline_water_usage_cubic_m == 0:
        return 0.0
    
    reduction = (
        (obs.baseline_water_usage_cubic_m - obs.water_usage_cubic_m)
        / obs.baseline_water_usage_cubic_m
        * 100.0
    )
    return max(0.0, reduction)


# ============================================================================
# Task Registry
# ============================================================================

TASKS: Dict[str, TaskConfig] = {
    "basic_compliance": TASK_BASIC_COMPLIANCE,
    "aggressive_sustainability": TASK_AGGRESSIVE_SUSTAINABILITY,
    "carbon_neutral_excellence": TASK_CARBON_NEUTRAL_EXCELLENCE,
}


GRADERS: Dict[str, Any] = {
    "basic_compliance": grade_basic_compliance,
    "aggressive_sustainability": grade_aggressive_sustainability,
    "carbon_neutral_excellence": grade_carbon_neutral_excellence,
}


def get_task_config(task_id: str) -> TaskConfig:
    """
    Get task configuration by ID.
    
    Args:
        task_id: Task identifier
        
    Returns:
        TaskConfig for the specified task
        
    Raises:
        KeyError: If task_id is not found
    """
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task_id: {task_id}. "
            f"Available tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def get_grader(task_id: str):
    """
    Get grader function for a task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Grader function that takes Observation and returns float (0.0-1.0)
        
    Raises:
        KeyError: If task_id is not found
    """
    if task_id not in GRADERS:
        raise KeyError(
            f"Unknown task_id: {task_id}. "
            f"Available graders: {list(GRADERS.keys())}"
        )
    return GRADERS[task_id]


def grade_task(task_id: str, final_obs: Observation) -> float:
    """
    Grade a completed task.
    
    Args:
        task_id: Task identifier
        final_obs: Final observation after episode completion
        
    Returns:
        Score between 0.0 and 1.0
    """
    grader = get_grader(task_id)
    return grader(final_obs)


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example usage and determinism tests.
    """
    from env import ESGEnvironment
    
    print("=" * 70)
    print("ESG TASK DEFINITIONS AND GRADER EXAMPLES")
    print("=" * 70)
    
    # Display all tasks
    for task_id, config in TASKS.items():
        print(f"\n{'='*70}")
        print(f"TASK: {task_id.upper()}")
        print(f"{'='*70}")
        print(f"Difficulty: {config.difficulty}")
        print(f"Max Steps: {config.max_steps} months")
        print(f"Budget: ${config.initial_budget:,.0f}")
        print(f"\nTargets:")
        print(f"  - Carbon Reduction: {config.target_carbon_reduction_pct}%")
        print(f"  - Renewable Energy: {config.target_renewable_pct}%")
        if config.target_diversity_score > 0:
            print(f"  - Diversity Score: {config.target_diversity_score}")
        if config.target_waste_recycling_pct > 0:
            print(f"  - Waste Recycling: {config.target_waste_recycling_pct}%")
        if config.target_water_reduction_pct > 0:
            print(f"  - Water Reduction: {config.target_water_reduction_pct}%")
        if config.target_employee_satisfaction > 0:
            print(f"  - Employee Satisfaction: {config.target_employee_satisfaction}")
        print(f"  - Max Compliance Violations: {config.max_compliance_violations}")
        print(f"\nDescription:")
        print(f"  {config.description}")
    
    # Test deterministic grading
    print(f"\n{'='*70}")
    print("DETERMINISTIC GRADING TEST")
    print(f"{'='*70}")
    
    # Create mock final observations with known values
    from models import Observation
    
    # Perfect performance scenario
    perfect_obs = Observation(
        energy_consumption_kwh=1000.0,
        renewable_energy_pct=90.0,
        carbon_emissions_tons=200.0,
        waste_generated_tons=400.0,
        waste_recycled_pct=80.0,
        water_usage_cubic_m=10000.0,
        diversity_score=90.0,
        employee_satisfaction=95.0,
        available_budget=100000.0,
        monthly_costs=30000.0,
        compliance_violations=0,
        audit_score=95.0,
        current_month=12,
        quarters_completed=4,
        target_carbon_reduction_pct=90.0,
        target_renewable_pct=80.0,
        target_diversity_score=85.0,
        baseline_carbon_emissions_tons=2000.0,  # 90% reduction
        baseline_water_usage_cubic_m=15000.0,   # 33% reduction
        actions_taken=[0, 1, 2, 3, 4, 5],
        total_investment=500000.0,
    )
    
    # Mediocre performance scenario
    mediocre_obs = Observation(
        energy_consumption_kwh=3500.0,
        renewable_energy_pct=40.0,
        carbon_emissions_tons=1200.0,
        waste_generated_tons=450.0,
        waste_recycled_pct=50.0,
        water_usage_cubic_m=18000.0,
        diversity_score=65.0,
        employee_satisfaction=70.0,
        available_budget=50000.0,
        monthly_costs=45000.0,
        compliance_violations=2,
        audit_score=60.0,
        current_month=9,
        quarters_completed=3,
        target_carbon_reduction_pct=40.0,
        target_renewable_pct=60.0,
        target_diversity_score=75.0,
        baseline_carbon_emissions_tons=2000.0,  # 40% reduction
        baseline_water_usage_cubic_m=20000.0,   # 10% reduction
        actions_taken=[0, 8, 8],
        total_investment=200000.0,
    )
    
    print("\nPerfect Performance Scores:")
    for task_id in TASKS.keys():
        score = grade_task(task_id, perfect_obs)
        print(f"  {task_id}: {score:.3f}")
    
    print("\nMediocre Performance Scores:")
    for task_id in TASKS.keys():
        score = grade_task(task_id, mediocre_obs)
        print(f"  {task_id}: {score:.3f}")
    
    # Test determinism (same input = same output)
    print("\nDeterminism Test (running grader 3 times on same input):")
    test_obs = mediocre_obs
    scores = []
    for i in range(3):
        score = grade_basic_compliance(test_obs)
        scores.append(score)
        print(f"  Run {i+1}: {score:.6f}")
    
    if len(set(scores)) == 1:
        print("  ✓ DETERMINISTIC: All scores identical")
    else:
        print("  ✗ NON-DETERMINISTIC: Scores differ!")
    
    print("\n" + "=" * 70)


__all__ = [
    "TASK_BASIC_COMPLIANCE",
    "TASK_AGGRESSIVE_SUSTAINABILITY",
    "TASK_CARBON_NEUTRAL_EXCELLENCE",
    "grade_basic_compliance",
    "grade_aggressive_sustainability",
    "grade_carbon_neutral_excellence",
    "TASKS",
    "GRADERS",
    "get_task_config",
    "get_grader",
    "grade_task",
]
