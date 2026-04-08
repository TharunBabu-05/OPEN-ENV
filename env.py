"""
AI ESG Compliance & Sustainability Evaluation Environment.

OpenEnv-compliant environment for training AI agents to optimize
Environmental, Social, and Governance (ESG) metrics in a simulated company.
"""

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action,
    ActionEffect,
    Observation,
    RewardComponents,
    TaskConfig,
    EnvironmentState,
)


class ESGEnvironment:
    """
    ESG Compliance & Sustainability Environment.
    
    Simulates a company's ESG performance where an AI agent takes monthly
    actions to improve environmental, social, and governance metrics while
    managing budget constraints and compliance requirements.
    """
    
    # Action effect definitions
    ACTION_EFFECTS: Dict[Action, ActionEffect] = {
        Action.INSTALL_SOLAR_PANELS: ActionEffect(
            action=Action.INSTALL_SOLAR_PANELS,
            cost=150000.0,
            immediate_renewable_delta=15.0,
            immediate_energy_delta=-300.0,
            immediate_cost_delta=-800.0,
            ongoing_renewable_delta_per_month=2.5,
            ongoing_energy_delta_per_month=-150.0,
            ongoing_cost_delta_per_month=-400.0,
            duration_months=6,
            compliance_improvement=1,
        ),
        Action.UPGRADE_HVAC_EFFICIENCY: ActionEffect(
            action=Action.UPGRADE_HVAC_EFFICIENCY,
            cost=80000.0,
            immediate_energy_delta=-500.0,
            immediate_cost_delta=-300.0,
            ongoing_energy_delta_per_month=-200.0,
            ongoing_cost_delta_per_month=-150.0,
            duration_months=12,
            compliance_improvement=0,
        ),
        Action.IMPLEMENT_RECYCLING_PROGRAM: ActionEffect(
            action=Action.IMPLEMENT_RECYCLING_PROGRAM,
            cost=25000.0,
            immediate_waste_recycling_delta=20.0,
            immediate_cost_delta=-100.0,
            ongoing_waste_recycling_delta_per_month=0.0,
            ongoing_cost_delta_per_month=-50.0,
            duration_months=12,
            compliance_improvement=1,
        ),
        Action.INSTALL_WATER_RECYCLING: ActionEffect(
            action=Action.INSTALL_WATER_RECYCLING,
            cost=60000.0,
            immediate_water_delta=-5000.0,
            immediate_cost_delta=-200.0,
            ongoing_water_delta_per_month=0.0,
            ongoing_cost_delta_per_month=-100.0,
            duration_months=12,
            compliance_improvement=0,
        ),
        Action.CARBON_OFFSET_PURCHASE: ActionEffect(
            action=Action.CARBON_OFFSET_PURCHASE,
            cost=40000.0,
            immediate_carbon_delta=-400.0,
            immediate_renewable_delta=0.0,
            ongoing_renewable_delta_per_month=0.0,
            duration_months=1,
            compliance_improvement=1,
        ),
        Action.DIVERSITY_HIRING_INITIATIVE: ActionEffect(
            action=Action.DIVERSITY_HIRING_INITIATIVE,
            cost=50000.0,
            immediate_diversity_delta=8.0,
            immediate_satisfaction_delta=3.0,
            ongoing_renewable_delta_per_month=0.0,
            duration_months=6,
            compliance_improvement=0,
        ),
        Action.EMPLOYEE_WELLNESS_PROGRAM: ActionEffect(
            action=Action.EMPLOYEE_WELLNESS_PROGRAM,
            cost=30000.0,
            immediate_satisfaction_delta=10.0,
            immediate_diversity_delta=2.0,
            ongoing_renewable_delta_per_month=0.0,
            duration_months=6,
            compliance_improvement=0,
        ),
        Action.ENERGY_AUDIT: ActionEffect(
            action=Action.ENERGY_AUDIT,
            cost=15000.0,
            immediate_energy_delta=-100.0,
            ongoing_energy_delta_per_month=-50.0,
            duration_months=3,
            compliance_improvement=0,
        ),
        Action.NO_ACTION: ActionEffect(
            action=Action.NO_ACTION,
            cost=0.0,
            duration_months=1,
            compliance_improvement=0,
        ),
    }
    
    def __init__(self, task_config: TaskConfig, seed: int = 42):
        """
        Initialize the ESG environment.
        
        Args:
            task_config: Configuration defining task objectives and constraints
            seed: Random seed for reproducibility
        """
        self.task_config = task_config
        self.seed = seed
        self.rng = random.Random(seed)
        self.state_internal: Optional[EnvironmentState] = None
        
    def reset(self) -> Observation:
        """
        Reset the environment to initial state.
        
        Creates a new company with baseline ESG metrics and returns the
        initial observation.
        
        Returns:
            Initial observation of the environment
        """
        # Reset RNG for deterministic episodes
        self.rng = random.Random(self.seed)
        
        # Generate initial company state (with some controlled variation)
        base_energy = 5000.0 + self.rng.uniform(-500, 500)
        base_renewable = 10.0 + self.rng.uniform(-5, 5)
        base_water = 20000.0 + self.rng.uniform(-2000, 2000)
        base_diversity = 45.0 + self.rng.uniform(-5, 5)
        base_satisfaction = 60.0 + self.rng.uniform(-10, 10)
        
        # Calculate initial carbon emissions based on energy and renewables
        carbon_factor = 0.4  # tons CO2 per kWh for non-renewable energy
        base_carbon = base_energy * (1 - base_renewable / 100.0) * carbon_factor
        
        initial_obs = Observation(
            # Environmental metrics
            energy_consumption_kwh=base_energy,
            renewable_energy_pct=base_renewable,
            carbon_emissions_tons=base_carbon,
            waste_generated_tons=500.0,
            waste_recycled_pct=25.0,
            water_usage_cubic_m=base_water,
            
            # Social metrics
            diversity_score=base_diversity,
            employee_satisfaction=base_satisfaction,
            
            # Financial state
            available_budget=self.task_config.initial_budget,
            monthly_costs=50000.0,
            
            # Governance
            compliance_violations=3,  # Start with some violations to fix
            audit_score=55.0,
            
            # Temporal
            current_month=1,
            quarters_completed=0,
            
            # Task targets
            target_carbon_reduction_pct=self.task_config.target_carbon_reduction_pct,
            target_renewable_pct=self.task_config.target_renewable_pct,
            target_diversity_score=self.task_config.target_diversity_score,
            
            # Baselines (for % reduction calculations)
            baseline_carbon_emissions_tons=base_carbon,
            baseline_water_usage_cubic_m=base_water,
            
            # History
            actions_taken=[],
            total_investment=0.0,
        )
        
        # Initialize internal state
        self.state_internal = EnvironmentState(
            observation=initial_obs,
            task_config=self.task_config,
            active_effects=[],
            previous_carbon_emissions=base_carbon,
            previous_renewable_pct=base_renewable,
            previous_diversity_score=base_diversity,
            previous_waste_recycled_pct=25.0,
            previous_water_usage=base_water,
            step_count=0,
            rng_seed=self.seed,
        )
        
        return initial_obs
    
    def step(
        self, action: int
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action ID to execute (from Action enum)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            - observation: New state after action
            - reward: Shaped reward for this step
            - terminated: True if episode completed (success/failure)
            - truncated: True if max steps reached
            - info: Additional information including reward components
        """
        if self.state_internal is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Convert action to enum
        action_enum = Action(action)
        
        # Store previous state for reward calculation
        prev_state = deepcopy(self.state_internal.observation)
        
        # Apply action effects
        self._apply_action(action_enum)
        
        # Apply ongoing effects from previous actions
        self._apply_ongoing_effects()
        
        # Simulate natural dynamics (monthly evolution)
        self._simulate_monthly_dynamics()
        
        # Update temporal information
        self._update_temporal_state()
        
        # Calculate carbon emissions based on current state
        self._update_carbon_emissions()
        
        # Update audit score based on overall performance
        self._update_audit_score()
        
        # Calculate shaped reward
        reward_components = self._calculate_reward(prev_state)
        reward = reward_components.total_reward
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Prepare info dict
        info = {
            "reward_components": reward_components.model_dump(),
            "carbon_reduction_pct": self._get_carbon_reduction_pct(),
            "water_reduction_pct": self._get_water_reduction_pct(),
            "task_progress": self._get_task_progress(),
            "budget_remaining": self.state_internal.observation.available_budget,
            "months_remaining": self.task_config.max_steps - self.state_internal.step_count,
        }
        
        # Update step count
        self.state_internal.step_count += 1
        
        return (
            self.state_internal.observation,
            reward,
            terminated,
            truncated,
            info,
        )
    
    def state(self) -> Observation:
        """
        Get current observation without stepping.
        
        Returns:
            Current observation
        """
        if self.state_internal is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state_internal.observation
    
    def _apply_action(self, action: Action) -> None:
        """Apply immediate and register ongoing effects of an action."""
        effect = self.ACTION_EFFECTS[action]
        obs = self.state_internal.observation
        
        # Deduct cost
        obs.available_budget -= effect.cost
        obs.total_investment += effect.cost
        
        # Apply immediate effects
        obs.energy_consumption_kwh = max(
            0, obs.energy_consumption_kwh + effect.immediate_energy_delta
        )
        obs.renewable_energy_pct = min(
            100, max(0, obs.renewable_energy_pct + effect.immediate_renewable_delta)
        )
        obs.carbon_emissions_tons = max(
            0, obs.carbon_emissions_tons + effect.immediate_carbon_delta
        )
        obs.waste_recycled_pct = min(
            100, max(0, obs.waste_recycled_pct + effect.immediate_waste_recycling_delta)
        )
        obs.water_usage_cubic_m = max(
            0, obs.water_usage_cubic_m + effect.immediate_water_delta
        )
        obs.diversity_score = min(
            100, max(0, obs.diversity_score + effect.immediate_diversity_delta)
        )
        obs.employee_satisfaction = min(
            100, max(0, obs.employee_satisfaction + effect.immediate_satisfaction_delta)
        )
        obs.monthly_costs = max(
            0, obs.monthly_costs + effect.immediate_cost_delta
        )
        obs.compliance_violations = max(
            0, obs.compliance_violations - effect.compliance_improvement
        )
        
        # Register ongoing effects
        if effect.duration_months > 1:
            self.state_internal.active_effects.append({
                "energy_delta": effect.ongoing_energy_delta_per_month,
                "renewable_delta": effect.ongoing_renewable_delta_per_month,
                "cost_delta": effect.ongoing_cost_delta_per_month,
                "remaining_months": effect.duration_months - 1,
            })
        
        # Track action
        obs.actions_taken.append(int(action))
    
    def _apply_ongoing_effects(self) -> None:
        """Apply ongoing effects from previous actions."""
        obs = self.state_internal.observation
        remaining_effects = []
        
        for effect in self.state_internal.active_effects:
            # Apply deltas
            obs.energy_consumption_kwh = max(
                0, obs.energy_consumption_kwh + effect["energy_delta"]
            )
            obs.renewable_energy_pct = min(
                100, max(0, obs.renewable_energy_pct + effect["renewable_delta"])
            )
            obs.monthly_costs = max(
                0, obs.monthly_costs + effect["cost_delta"]
            )
            
            # Decrement duration
            effect["remaining_months"] -= 1
            if effect["remaining_months"] > 0:
                remaining_effects.append(effect)
        
        self.state_internal.active_effects = remaining_effects
    
    def _simulate_monthly_dynamics(self) -> None:
        """
        Simulate natural monthly evolution of company metrics.
        
        Includes seasonal energy fluctuations, employee churn effects,
        and natural drift in metrics.
        """
        obs = self.state_internal.observation
        
        # Seasonal energy variation (deterministic based on month)
        month_factor = 1.0 + 0.1 * ((obs.current_month - 1) % 12 / 12.0 - 0.5)
        seasonal_variation = self.rng.uniform(-100, 200) * month_factor
        obs.energy_consumption_kwh = max(
            1000, obs.energy_consumption_kwh + seasonal_variation
        )
        
        # Waste generation varies slightly
        obs.waste_generated_tons = max(
            100, obs.waste_generated_tons + self.rng.uniform(-20, 20)
        )
        
        # Natural employee satisfaction drift
        satisfaction_drift = self.rng.uniform(-2, 1)  # Easier to lose than gain
        obs.employee_satisfaction = min(
            100, max(20, obs.employee_satisfaction + satisfaction_drift)
        )
        
        # Diversity score has slight natural improvement trend
        diversity_drift = self.rng.uniform(-0.5, 1.0)
        obs.diversity_score = min(
            100, max(0, obs.diversity_score + diversity_drift)
        )
        
        # Water usage variation
        water_variation = self.rng.uniform(-500, 500)
        obs.water_usage_cubic_m = max(
            1000, obs.water_usage_cubic_m + water_variation
        )
        
        # Recycling percentage slowly degrades without maintenance
        if obs.waste_recycled_pct > 25:
            obs.waste_recycled_pct = max(25, obs.waste_recycled_pct - 0.5)
    
    def _update_temporal_state(self) -> None:
        """Update month and quarter counters."""
        obs = self.state_internal.observation
        obs.current_month += 1
        
        # Update quarters
        if obs.current_month % 3 == 1 and obs.current_month > 1:
            obs.quarters_completed += 1
    
    def _update_carbon_emissions(self) -> None:
        """
        Recalculate carbon emissions based on current energy and renewable %.
        
        Carbon emissions = energy consumption * (1 - renewable%) * emission factor
        """
        obs = self.state_internal.observation
        carbon_factor = 0.4  # tons CO2 per kWh for non-renewable
        obs.carbon_emissions_tons = (
            obs.energy_consumption_kwh
            * (1.0 - obs.renewable_energy_pct / 100.0)
            * carbon_factor
        )
    
    def _update_audit_score(self) -> None:
        """
        Update ESG audit score based on current performance.
        
        Audit score is a weighted combination of all ESG metrics.
        """
        obs = self.state_internal.observation
        
        # Environmental component (40%)
        carbon_reduction_pct = self._get_carbon_reduction_pct()
        env_score = (
            0.3 * min(100, obs.renewable_energy_pct)
            + 0.3 * min(100, carbon_reduction_pct * 2)  # Scale to 0-100
            + 0.2 * obs.waste_recycled_pct
            + 0.2 * min(100, (1 - obs.water_usage_cubic_m / obs.baseline_water_usage_cubic_m) * 100)
        )
        
        # Social component (30%)
        social_score = 0.6 * obs.diversity_score + 0.4 * obs.employee_satisfaction
        
        # Governance component (30%)
        governance_score = max(0, 100 - obs.compliance_violations * 10)
        
        # Weighted audit score
        obs.audit_score = 0.4 * env_score + 0.3 * social_score + 0.3 * governance_score
    
    def _calculate_reward(self, prev_state: Observation) -> RewardComponents:
        """
        Calculate shaped reward based on progress toward targets.
        
        Reward components:
        - Progress toward carbon reduction
        - Progress toward renewable energy target
        - Progress toward diversity target
        - Waste recycling improvements
        - Water usage reduction
        - Budget management penalties
        - Compliance penalties
        - Quarterly milestone bonuses
        - Synergy bonuses for multi-metric improvements
        - Task completion bonus (if applicable)
        """
        obs = self.state_internal.observation
        
        # Track previous state for reward shaping
        prev_carbon = self.state_internal.previous_carbon_emissions
        prev_renewable = self.state_internal.previous_renewable_pct
        prev_diversity = self.state_internal.previous_diversity_score
        prev_waste_recycled = self.state_internal.previous_waste_recycled_pct
        prev_water = self.state_internal.previous_water_usage
        
        # 1. Carbon reduction progress
        carbon_progress = prev_carbon - obs.carbon_emissions_tons
        carbon_reward = 0.1 * (carbon_progress / 100.0) if carbon_progress > 0 else 0.0
        
        # 2. Renewable energy progress
        renewable_progress = obs.renewable_energy_pct - prev_renewable
        renewable_reward = 0.05 * renewable_progress if renewable_progress > 0 else 0.0
        
        # 3. Diversity progress
        diversity_progress = obs.diversity_score - prev_diversity
        diversity_reward = 0.05 * diversity_progress if diversity_progress > 0 else 0.0
        
        # 4. Waste recycling progress
        waste_progress = obs.waste_recycled_pct - prev_waste_recycled
        waste_reward = 0.03 * waste_progress if waste_progress > 0 else 0.0
        
        # 5. Water reduction progress
        water_progress = prev_water - obs.water_usage_cubic_m
        water_reward = 0.02 * (water_progress / 1000.0) if water_progress > 0 else 0.0
        
        # 6. Budget penalty (bankruptcy is bad)
        budget_penalty = -1.0 if obs.available_budget < 0 else 0.0
        
        # 7. Compliance penalty
        compliance_penalty = -0.2 * obs.compliance_violations
        
        # 8. Quarterly milestone bonus
        quarterly_bonus = 0.0
        if obs.current_month % 3 == 0 and obs.current_month > 1:
            # Check if we're on track for task completion
            progress = self._get_task_progress()
            expected_progress = obs.current_month / self.task_config.max_steps
            if progress >= expected_progress * 0.8:  # 80% of expected progress
                quarterly_bonus = 0.5
            else:
                quarterly_bonus = -0.3
        
        # 9. Synergy bonus (multiple improvements at once)
        synergy_bonus = 0.0
        improvements = sum([
            carbon_progress > 50,
            renewable_progress > 2,
            diversity_progress > 1,
            waste_progress > 2,
        ])
        if improvements >= 2:
            synergy_bonus = 0.15 * improvements
        
        # 10. Task completion bonus (terminal reward)
        task_completion_reward = 0.0
        if self._is_task_complete():
            task_completion_reward = 5.0
        elif self.state_internal.step_count >= self.task_config.max_steps - 1:
            # Final step - partial credit or penalty
            progress = self._get_task_progress()
            if progress >= 0.8:
                task_completion_reward = 2.0
            else:
                task_completion_reward = -1.0
        
        # Total reward
        total = (
            carbon_reward
            + renewable_reward
            + diversity_reward
            + waste_reward
            + water_reward
            + budget_penalty
            + compliance_penalty
            + quarterly_bonus
            + synergy_bonus
            + task_completion_reward
        )
        
        # Update previous state tracking
        self.state_internal.previous_carbon_emissions = obs.carbon_emissions_tons
        self.state_internal.previous_renewable_pct = obs.renewable_energy_pct
        self.state_internal.previous_diversity_score = obs.diversity_score
        self.state_internal.previous_waste_recycled_pct = obs.waste_recycled_pct
        self.state_internal.previous_water_usage = obs.water_usage_cubic_m
        
        return RewardComponents(
            carbon_progress_reward=carbon_reward,
            renewable_progress_reward=renewable_reward,
            diversity_progress_reward=diversity_reward,
            waste_recycling_reward=waste_reward,
            water_reduction_reward=water_reward,
            budget_penalty=budget_penalty,
            compliance_penalty=compliance_penalty,
            quarterly_bonus=quarterly_bonus,
            synergy_bonus=synergy_bonus,
            task_completion_reward=task_completion_reward,
            total_reward=total,
        )
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check if episode should terminate.
        
        Returns:
            Tuple of (terminated, truncated)
            - terminated: True if task completed or failed
            - truncated: True if max steps reached
        """
        obs = self.state_internal.observation
        
        # Success termination
        if self._is_task_complete():
            return True, False
        
        # Failure termination (bankruptcy)
        if obs.available_budget < -100000:  # Severe debt
            return True, False
        
        # Truncation (max steps)
        if self.state_internal.step_count >= self.task_config.max_steps:
            return False, True
        
        return False, False
    
    def _is_task_complete(self) -> bool:
        """Check if all task objectives are met."""
        obs = self.state_internal.observation
        config = self.task_config
        
        # Carbon reduction target
        carbon_reduction_pct = self._get_carbon_reduction_pct()
        if carbon_reduction_pct < config.target_carbon_reduction_pct:
            return False
        
        # Renewable energy target
        if obs.renewable_energy_pct < config.target_renewable_pct:
            return False
        
        # Diversity target (if required)
        if config.target_diversity_score > 0:
            if obs.diversity_score < config.target_diversity_score:
                return False
        
        # Waste recycling target (if required)
        if config.target_waste_recycling_pct > 0:
            if obs.waste_recycled_pct < config.target_waste_recycling_pct:
                return False
        
        # Water reduction target (if required)
        if config.target_water_reduction_pct > 0:
            water_reduction_pct = self._get_water_reduction_pct()
            if water_reduction_pct < config.target_water_reduction_pct:
                return False
        
        # Employee satisfaction target (if required)
        if config.target_employee_satisfaction > 0:
            if obs.employee_satisfaction < config.target_employee_satisfaction:
                return False
        
        # Compliance violations
        if obs.compliance_violations > config.max_compliance_violations:
            return False
        
        return True
    
    def _get_task_progress(self) -> float:
        """
        Calculate overall task progress (0.0 to 1.0).
        
        Returns weighted average of progress toward each target.
        """
        obs = self.state_internal.observation
        config = self.task_config
        
        # Carbon reduction progress
        carbon_reduction_pct = self._get_carbon_reduction_pct()
        carbon_progress = min(1.0, carbon_reduction_pct / config.target_carbon_reduction_pct)
        
        # Renewable energy progress
        renewable_progress = min(1.0, obs.renewable_energy_pct / config.target_renewable_pct)
        
        # Start with required targets
        total_weight = 0.5 + 0.3  # Carbon + renewable always required
        total_progress = 0.5 * carbon_progress + 0.3 * renewable_progress
        
        # Add optional targets if specified
        if config.target_diversity_score > 0:
            diversity_progress = min(1.0, obs.diversity_score / config.target_diversity_score)
            total_progress += 0.1 * diversity_progress
            total_weight += 0.1
        
        if config.target_waste_recycling_pct > 0:
            waste_progress = min(1.0, obs.waste_recycled_pct / config.target_waste_recycling_pct)
            total_progress += 0.1 * waste_progress
            total_weight += 0.1
        
        return total_progress / total_weight if total_weight > 0 else 0.0
    
    def _get_carbon_reduction_pct(self) -> float:
        """Calculate percentage reduction in carbon emissions from baseline."""
        obs = self.state_internal.observation
        if obs.baseline_carbon_emissions_tons == 0:
            return 0.0
        reduction = (
            (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
            / obs.baseline_carbon_emissions_tons
            * 100.0
        )
        return max(0.0, reduction)
    
    def _get_water_reduction_pct(self) -> float:
        """Calculate percentage reduction in water usage from baseline."""
        obs = self.state_internal.observation
        if obs.baseline_water_usage_cubic_m == 0:
            return 0.0
        reduction = (
            (obs.baseline_water_usage_cubic_m - obs.water_usage_cubic_m)
            / obs.baseline_water_usage_cubic_m
            * 100.0
        )
        return max(0.0, reduction)


__all__ = ["ESGEnvironment"]
