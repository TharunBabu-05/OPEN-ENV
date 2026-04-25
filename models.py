"""
Pydantic models for AI ESG Compliance & Sustainability Evaluation Environment.

This module defines the observation space, action space, and reward structure
for an OpenEnv-compliant environment where AI agents optimize ESG metrics.
"""

from enum import IntEnum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class Action(IntEnum):
    """
    Discrete action space for ESG improvement interventions.
    
    Each action represents a strategic decision the agent can make to improve
    environmental, social, or governance metrics. Actions have associated costs,
    impacts, and implementation timelines.
    """
    
    INSTALL_SOLAR_PANELS = 0  # High upfront cost, long-term renewable energy gain
    UPGRADE_HVAC_EFFICIENCY = 1  # Moderate cost, reduces energy consumption
    IMPLEMENT_RECYCLING_PROGRAM = 2  # Low cost, improves waste recycling %
    INSTALL_WATER_RECYCLING = 3  # Moderate cost, reduces water usage
    CARBON_OFFSET_PURCHASE = 4  # Variable cost, immediate carbon reduction
    DIVERSITY_HIRING_INITIATIVE = 5  # Moderate cost, improves diversity score
    EMPLOYEE_WELLNESS_PROGRAM = 6  # Low cost, improves employee satisfaction
    ENERGY_AUDIT = 7  # Low cost, reveals optimization opportunities
    NO_ACTION = 8  # Skip month, conserve budget


class Observation(BaseModel):
    """
    Complete observable state of the company's ESG performance.
    
    Includes environmental metrics (energy, emissions, waste, water),
    social metrics (diversity, satisfaction), governance metrics (compliance),
    financial state (budget, costs), and task-specific targets.
    """
    
    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        json_schema_extra={
            "description": "ESG company state observation"
        }
    )
    
    # === Environmental Metrics ===
    energy_consumption_kwh: float = Field(
        ...,
        ge=0.0,
        le=20000.0,
        description="Current monthly energy consumption in kilowatt-hours"
    )
    
    renewable_energy_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of energy from renewable sources"
    )
    
    carbon_emissions_tons: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Monthly carbon dioxide emissions in metric tons"
    )
    
    waste_generated_tons: float = Field(
        ...,
        ge=0.0,
        le=2000.0,
        description="Monthly waste generated in metric tons"
    )
    
    waste_recycled_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of waste that is recycled"
    )
    
    water_usage_cubic_m: float = Field(
        ...,
        ge=0.0,
        le=100000.0,
        description="Monthly water consumption in cubic meters"
    )
    
    # === Social Metrics ===
    diversity_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite diversity and inclusion score (0-100)"
    )
    
    employee_satisfaction: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Employee satisfaction index (0-100)"
    )
    
    # === Financial State ===
    available_budget: float = Field(
        ...,
        description="Remaining budget for ESG improvements (can be negative)"
    )
    
    monthly_costs: float = Field(
        ...,
        ge=0.0,
        description="Current monthly operating costs in USD"
    )
    
    # === Governance & Compliance ===
    compliance_violations: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of active compliance violations"
    )
    
    audit_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Most recent ESG audit score (0-100)"
    )
    
    # === Temporal Information ===
    current_month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Current month in the simulation (1-12)"
    )
    
    quarters_completed: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of completed quarters (0-4)"
    )
    
    # === Task-Specific Targets ===
    target_carbon_reduction_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Required carbon emission reduction percentage for task completion"
    )
    
    target_renewable_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Required renewable energy percentage for task completion"
    )
    
    target_diversity_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Required diversity score for task completion"
    )
    
    # === Baseline Tracking (for calculating reductions) ===
    baseline_carbon_emissions_tons: float = Field(
        ...,
        ge=0.0,
        description="Initial carbon emissions at environment reset (for % reduction calc)"
    )
    
    baseline_water_usage_cubic_m: float = Field(
        ...,
        ge=0.0,
        description="Initial water usage at environment reset (for % reduction calc)"
    )
    
    # === Historical Context ===
    actions_taken: List[int] = Field(
        default_factory=list,
        description="History of actions taken (action IDs)"
    )
    
    total_investment: float = Field(
        default=0.0,
        ge=0.0,
        description="Cumulative amount spent on ESG initiatives"
    )


class ActionEffect(BaseModel):
    """
    Defines the impact of an action on company metrics.
    
    Each action has immediate effects (applied on execution) and ongoing effects
    (applied each subsequent month for a duration).
    """
    
    model_config = ConfigDict(frozen=True)
    
    action: Action = Field(..., description="The action being defined")
    
    cost: float = Field(
        ...,
        ge=0.0,
        description="Upfront cost in USD"
    )
    
    # Immediate effects (applied once)
    immediate_energy_delta: float = Field(
        default=0.0,
        description="Immediate change in energy consumption (kWh)"
    )
    
    immediate_renewable_delta: float = Field(
        default=0.0,
        description="Immediate change in renewable energy %"
    )
    
    immediate_carbon_delta: float = Field(
        default=0.0,
        description="Immediate change in carbon emissions (tons)"
    )
    
    immediate_waste_recycling_delta: float = Field(
        default=0.0,
        description="Immediate change in waste recycling %"
    )
    
    immediate_water_delta: float = Field(
        default=0.0,
        description="Immediate change in water usage (cubic meters)"
    )
    
    immediate_diversity_delta: float = Field(
        default=0.0,
        description="Immediate change in diversity score"
    )
    
    immediate_satisfaction_delta: float = Field(
        default=0.0,
        description="Immediate change in employee satisfaction"
    )
    
    immediate_cost_delta: float = Field(
        default=0.0,
        description="Immediate change in monthly operating costs"
    )
    
    # Ongoing effects (applied per month)
    ongoing_energy_delta_per_month: float = Field(
        default=0.0,
        description="Monthly change in energy consumption (kWh/month)"
    )
    
    ongoing_renewable_delta_per_month: float = Field(
        default=0.0,
        description="Monthly change in renewable energy %/month"
    )
    
    ongoing_cost_delta_per_month: float = Field(
        default=0.0,
        description="Monthly change in operating costs (USD/month)"
    )
    
    duration_months: int = Field(
        default=1,
        ge=1,
        description="Number of months the ongoing effects persist"
    )
    
    compliance_improvement: int = Field(
        default=0,
        description="Number of compliance violations resolved"
    )


class RewardComponents(BaseModel):
    """
    Breakdown of shaped reward components for transparency and debugging.
    
    This allows the agent (and evaluators) to understand which aspects of
    performance contributed to the total reward.
    """
    
    model_config = ConfigDict(frozen=True)
    
    carbon_progress_reward: float = Field(
        default=0.0,
        description="Reward for reducing carbon emissions"
    )
    
    renewable_progress_reward: float = Field(
        default=0.0,
        description="Reward for increasing renewable energy"
    )
    
    diversity_progress_reward: float = Field(
        default=0.0,
        description="Reward for improving diversity score"
    )
    
    waste_recycling_reward: float = Field(
        default=0.0,
        description="Reward for increasing waste recycling"
    )
    
    water_reduction_reward: float = Field(
        default=0.0,
        description="Reward for reducing water usage"
    )
    
    budget_penalty: float = Field(
        default=0.0,
        description="Penalty for exceeding budget (negative value)"
    )
    
    compliance_penalty: float = Field(
        default=0.0,
        description="Penalty for compliance violations (negative value)"
    )
    
    quarterly_bonus: float = Field(
        default=0.0,
        description="Bonus for meeting quarterly milestones"
    )
    
    synergy_bonus: float = Field(
        default=0.0,
        description="Bonus for simultaneous improvements across metrics"
    )
    
    task_completion_reward: float = Field(
        default=0.0,
        description="Final reward for completing task objectives"
    )
    
    anti_cheat_penalty: float = Field(
        default=0.0,
        description="Penalty for reward hacking (e.g. spamming NO_ACTION or repeating same cheap action)"
    )
    
    format_compliance_reward: float = Field(
        default=0.0,
        description="Small reward for taking a valid, non-trivial action (format compliance signal)"
    )
    
    total_reward: float = Field(
        ...,
        description="Sum of all reward components"
    )


class TaskConfig(BaseModel):
    """
    Configuration for a specific ESG task/difficulty level.
    
    Defines targets, constraints, and success criteria for a task instance.
    """
    
    model_config = ConfigDict(frozen=True)
    
    task_id: str = Field(
        ...,
        description="Unique identifier for the task"
    )
    
    difficulty: str = Field(
        ...,
        description="Difficulty level: 'easy', 'medium', or 'hard'"
    )
    
    max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum number of months (steps) allowed"
    )
    
    initial_budget: float = Field(
        ...,
        gt=0.0,
        description="Starting budget in USD"
    )
    
    # Task targets
    target_carbon_reduction_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Required carbon emission reduction %"
    )
    
    target_renewable_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Required renewable energy %"
    )
    
    target_diversity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Required diversity score (0 = not required)"
    )
    
    target_waste_recycling_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Required waste recycling % (0 = not required)"
    )
    
    target_water_reduction_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Required water usage reduction % (0 = not required)"
    )
    
    target_employee_satisfaction: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Required employee satisfaction (0 = not required)"
    )
    
    max_compliance_violations: int = Field(
        default=999,
        ge=0,
        description="Maximum allowed compliance violations"
    )
    
    description: str = Field(
        default="",
        description="Human-readable task description"
    )


class EnvironmentState(BaseModel):
    """
    Complete internal state of the ESG environment.
    
    This extends Observation with additional internal tracking for action effects,
    random seeds, and implementation details not exposed to the agent.
    """
    
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    # Core observation (what agent sees)
    observation: Observation = Field(
        ...,
        description="Current observable state"
    )
    
    # Task configuration
    task_config: TaskConfig = Field(
        ...,
        description="Current task parameters"
    )
    
    # Active ongoing effects from previous actions
    active_effects: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Active ongoing action effects with remaining duration"
    )
    
    # Previous state for reward shaping
    previous_carbon_emissions: float = Field(
        default=0.0,
        description="Carbon emissions from previous step"
    )
    
    previous_renewable_pct: float = Field(
        default=0.0,
        description="Renewable energy % from previous step"
    )
    
    previous_diversity_score: float = Field(
        default=0.0,
        description="Diversity score from previous step"
    )
    
    previous_waste_recycled_pct: float = Field(
        default=0.0,
        description="Waste recycling % from previous step"
    )
    
    previous_water_usage: float = Field(
        default=0.0,
        description="Water usage from previous step"
    )
    
    # Determinism support
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken in current episode"
    )
    
    rng_seed: int = Field(
        default=42,
        description="Random seed for deterministic behavior"
    )


# Export all models
__all__ = [
    "Action",
    "Observation",
    "ActionEffect",
    "RewardComponents",
    "TaskConfig",
    "EnvironmentState",
]
