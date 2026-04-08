"""
Inference script for ESG Compliance Environment.

This script runs an AI agent (via OpenAI API) on all three ESG tasks
and reports structured logs and final scores.

Environment Variables:
    API_BASE_URL: Base URL for OpenAI-compatible API
    MODEL_NAME: Name of the model to use
    HF_TOKEN: Hugging Face token for authentication
"""

import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple

from openai import OpenAI

from env import ESGEnvironment
from models import Action, Observation
from tasks import TASKS, grade_task


def log_start(task_id: str, config: Dict[str, Any]) -> None:
    """Log task start in structured format."""
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "difficulty": config["difficulty"],
        "max_steps": config["max_steps"],
        "budget": config["initial_budget"],
        "timestamp": time.time(),
    }))
    sys.stdout.flush()


def log_step(
    task_id: str,
    step: int,
    action: int,
    action_name: str,
    observation: Dict[str, Any],
    reward: float,
    info: Dict[str, Any],
) -> None:
    """Log environment step in structured format."""
    print(json.dumps({
        "type": "STEP",
        "task_id": task_id,
        "step": step,
        "action": action,
        "action_name": action_name,
        "observation": {
            "energy_kwh": observation["energy_consumption_kwh"],
            "renewable_pct": observation["renewable_energy_pct"],
            "carbon_tons": observation["carbon_emissions_tons"],
            "diversity": observation["diversity_score"],
            "budget": observation["available_budget"],
            "violations": observation["compliance_violations"],
        },
        "reward": reward,
        "carbon_reduction_pct": info.get("carbon_reduction_pct", 0.0),
        "task_progress": info.get("task_progress", 0.0),
        "timestamp": time.time(),
    }))
    sys.stdout.flush()


def log_end(task_id: str, score: float, total_steps: int, final_obs: Dict[str, Any]) -> None:
    """Log task completion in structured format."""
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "score": score,
        "total_steps": total_steps,
        "final_state": {
            "carbon_reduction_pct": (
                (final_obs["baseline_carbon_emissions_tons"] - final_obs["carbon_emissions_tons"])
                / final_obs["baseline_carbon_emissions_tons"] * 100
                if final_obs["baseline_carbon_emissions_tons"] > 0 else 0.0
            ),
            "renewable_pct": final_obs["renewable_energy_pct"],
            "diversity": final_obs["diversity_score"],
            "budget_remaining": final_obs["available_budget"],
            "violations": final_obs["compliance_violations"],
        },
        "timestamp": time.time(),
    }))
    sys.stdout.flush()


def create_system_prompt() -> str:
    """Create system prompt for the ESG agent."""
    return """You are an expert ESG (Environmental, Social, Governance) sustainability strategist.

Your goal is to optimize a company's ESG metrics by taking strategic actions each month.

Available Actions:
0. INSTALL_SOLAR_PANELS - $150K, increases renewable energy significantly
1. UPGRADE_HVAC_EFFICIENCY - $80K, reduces energy consumption
2. IMPLEMENT_RECYCLING_PROGRAM - $25K, increases waste recycling
3. INSTALL_WATER_RECYCLING - $60K, reduces water usage
4. CARBON_OFFSET_PURCHASE - $40K, immediately reduces carbon emissions
5. DIVERSITY_HIRING_INITIATIVE - $50K, improves diversity score
6. EMPLOYEE_WELLNESS_PROGRAM - $30K, improves employee satisfaction
7. ENERGY_AUDIT - $15K, identifies inefficiencies
8. NO_ACTION - $0, conserve budget

Key Strategies:
- Solar panels have ongoing benefits (increase renewable % each month for 6 months)
- HVAC upgrades reduce energy consumption over time
- Balance immediate impact vs. long-term benefits
- Watch your budget carefully
- Prioritize actions that address task targets

You must respond with ONLY a valid JSON object in this exact format:
{
    "action": <action_number>,
    "reasoning": "<brief explanation>"
}

Do not include any other text before or after the JSON."""


def create_task_prompt(task_config: Dict[str, Any], obs: Observation, step: int) -> str:
    """Create task-specific prompt for current state."""
    carbon_reduction = (
        (obs.baseline_carbon_emissions_tons - obs.carbon_emissions_tons)
        / obs.baseline_carbon_emissions_tons * 100
        if obs.baseline_carbon_emissions_tons > 0 else 0.0
    )
    
    water_reduction = (
        (obs.baseline_water_usage_cubic_m - obs.water_usage_cubic_m)
        / obs.baseline_water_usage_cubic_m * 100
        if obs.baseline_water_usage_cubic_m > 0 else 0.0
    )
    
    prompt = f"""TASK: {task_config['task_id']} ({task_config['difficulty']})
Month: {step + 1}/{task_config['max_steps']}

TARGETS:
- Carbon Reduction: {task_config['target_carbon_reduction_pct']}%
- Renewable Energy: {task_config['target_renewable_pct']}%
"""
    
    if task_config.get('target_diversity_score', 0) > 0:
        prompt += f"- Diversity Score: {task_config['target_diversity_score']}\n"
    if task_config.get('target_waste_recycling_pct', 0) > 0:
        prompt += f"- Waste Recycling: {task_config['target_waste_recycling_pct']}%\n"
    if task_config.get('target_water_reduction_pct', 0) > 0:
        prompt += f"- Water Reduction: {task_config['target_water_reduction_pct']}%\n"
    if task_config.get('target_employee_satisfaction', 0) > 0:
        prompt += f"- Employee Satisfaction: {task_config['target_employee_satisfaction']}\n"
    
    prompt += f"- Max Violations: {task_config['max_compliance_violations']}\n\n"
    
    prompt += f"""CURRENT STATE:
- Energy Consumption: {obs.energy_consumption_kwh:.0f} kWh
- Renewable Energy: {obs.renewable_energy_pct:.1f}% (target: {task_config['target_renewable_pct']}%)
- Carbon Emissions: {obs.carbon_emissions_tons:.0f} tons
- Carbon Reduction: {carbon_reduction:.1f}% (target: {task_config['target_carbon_reduction_pct']}%)
- Waste Recycled: {obs.waste_recycled_pct:.1f}%
- Water Usage: {obs.water_usage_cubic_m:.0f} m³ (reduction: {water_reduction:.1f}%)
- Diversity Score: {obs.diversity_score:.1f}
- Employee Satisfaction: {obs.employee_satisfaction:.1f}
- Available Budget: ${obs.available_budget:,.0f}
- Compliance Violations: {obs.compliance_violations}

What action should you take this month? Consider:
1. Which targets are furthest from goals?
2. How much time remains?
3. Budget constraints
4. Long-term vs. immediate impact

Respond with JSON only."""
    
    return prompt


def get_llm_action(
    client: OpenAI,
    model_name: str,
    task_config: Dict[str, Any],
    obs: Observation,
    step: int,
    max_retries: int = 3,
) -> Tuple[int, str]:
    """
    Get action from LLM via OpenAI API.
    
    Returns:
        Tuple of (action_id, reasoning)
    """
    system_prompt = create_system_prompt()
    user_prompt = create_task_prompt(task_config, obs, step)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # Deterministic
                max_tokens=200,
                timeout=30.0,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON (handle cases where LLM adds markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            result = json.loads(content)
            action = int(result["action"])
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Validate action
            if 0 <= action <= 8:
                return action, reasoning
            else:
                print(f"Warning: Invalid action {action}, defaulting to NO_ACTION", file=sys.stderr)
                return 8, "Invalid action, defaulting to NO_ACTION"
                
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON decode error: {e}", file=sys.stderr)
            print(f"Response content: {content}", file=sys.stderr)
            if attempt == max_retries - 1:
                print("Max retries reached, using NO_ACTION", file=sys.stderr)
                return 8, "Failed to parse LLM response"
                
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error getting LLM action: {e}", file=sys.stderr)
            if attempt == max_retries - 1:
                print("Max retries reached, using NO_ACTION", file=sys.stderr)
                return 8, f"Error: {str(e)}"
        
        time.sleep(1)  # Brief delay before retry
    
    return 8, "Failed after max retries"


def run_task(
    client: OpenAI,
    model_name: str,
    task_id: str,
    seed: int = 42,
) -> float:
    """
    Run a single task with LLM agent.
    
    Args:
        client: OpenAI client
        model_name: Model name to use
        task_id: Task identifier
        seed: Random seed for reproducibility
        
    Returns:
        Final score (0.0 to 1.0)
    """
    # Get task configuration
    task_config = TASKS[task_id]
    
    # Log task start
    log_start(task_id, task_config.model_dump())
    
    # Create environment
    env = ESGEnvironment(task_config=task_config, seed=seed)
    
    # Reset environment
    obs = env.reset()
    
    total_reward = 0.0
    step_count = 0
    
    # Run episode
    for step in range(task_config.max_steps):
        # Get action from LLM
        action, reasoning = get_llm_action(
            client=client,
            model_name=model_name,
            task_config=task_config.model_dump(),
            obs=obs,
            step=step,
        )
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Log step
        action_name = Action(action).name
        log_step(
            task_id=task_id,
            step=step + 1,
            action=action,
            action_name=action_name,
            observation=obs.model_dump(),
            reward=reward,
            info=info,
        )
        
        # Check if episode ended
        if terminated or truncated:
            break
    
    # Grade final performance
    final_score = grade_task(task_id, obs)
    
    # Log task end
    log_end(
        task_id=task_id,
        score=final_score,
        total_steps=step_count,
        final_obs=obs.model_dump(),
    )
    
    return final_score


def run_inference() -> int:
    """
    Run inference on all tasks and compute average score.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Read environment variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")
    
    # Validate environment variables
    if not api_base_url:
        print("Error: API_BASE_URL environment variable not set", file=sys.stderr)
        return 1
    
    if not model_name:
        print("Error: MODEL_NAME environment variable not set", file=sys.stderr)
        return 1
    
    # Create OpenAI client
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token or "dummy-key",  # Some endpoints don't require auth
    )
    
    print(json.dumps({
        "type": "INFO",
        "message": "Starting ESG Environment Inference",
        "api_base_url": api_base_url,
        "model_name": model_name,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()
    
    # Run all tasks
    task_scores = {}
    
    for task_id in ["basic_compliance", "aggressive_sustainability", "carbon_neutral_excellence"]:
        try:
            print(json.dumps({
                "type": "INFO",
                "message": f"Running task: {task_id}",
                "timestamp": time.time(),
            }))
            sys.stdout.flush()
            
            score = run_task(
                client=client,
                model_name=model_name,
                task_id=task_id,
                seed=42,
            )
            
            task_scores[task_id] = score
            
            print(json.dumps({
                "type": "INFO",
                "message": f"Completed task: {task_id}",
                "score": score,
                "timestamp": time.time(),
            }))
            sys.stdout.flush()
            
        except Exception as e:
            print(json.dumps({
                "type": "ERROR",
                "task_id": task_id,
                "error": str(e),
                "timestamp": time.time(),
            }), file=sys.stderr)
            task_scores[task_id] = 0.0
    
    # Calculate average score
    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    
    # Print final summary
    print(json.dumps({
        "type": "SUMMARY",
        "task_scores": task_scores,
        "average_score": avg_score,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()
    
    return 0


if __name__ == "__main__":
    exit_code = run_inference()
    sys.exit(exit_code)
