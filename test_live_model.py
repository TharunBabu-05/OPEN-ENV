import httpx
import time
import json
from huggingface_hub import InferenceClient

API_URL = "https://tharun5054-esg-compliance-env.hf.space"
MODEL_ID = "tharun5054/esg-rl-agent-grpo"
HF_TOKEN = "hf_REPLACE_ME"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

def build_prompt(obs, task_id):
    targets = {
        "basic_compliance": {"carbon": -15, "renewable": 30},
        "aggressive_sustainability": {"carbon": -40, "renewable": 60},
        "carbon_neutral_excellence": {"carbon": -90, "renewable": 80}
    }
    t = targets.get(task_id, {"carbon": 0, "renewable": 0})
    
    prompt = f"""You are an ESG sustainability strategist.

Current state:
  Carbon Emissions: {obs['carbon_emissions_tons']:.0f} tons (target: {t['carbon']}%)
  Renewable Energy: {obs['renewable_energy_pct']:.1f}% (target: {t['renewable']}%)
  Available Budget: ${obs['available_budget']/1000:.0f}K
  Month: {obs['current_month']}

Actions: 
0=Solar, 1=HVAC, 2=Recycling, 3=Water, 4=CarbonOffset, 5=Diversity, 6=Wellness, 7=Audit, 8=NoAction

Choose an action (0-8) and explain your reasoning. Output JSON only in format: {{"action": int, "reasoning": "string"}}
"""
    return prompt

def test_live():
    print(f"\nConnecting to Live API: {API_URL}")
    
    task_id = "aggressive_sustainability"
    
    # 1. Reset Environment
    print(f"Starting new session for task: {task_id}")
    r = httpx.post(f"{API_URL}/reset", json={"task_id": task_id, "seed": 42}, timeout=30.0)
    obs = r.json()
    
    print("Environment reset successfully.\n")
    
    step = 1
    total_reward = 0.0
    
    while True:
        print(f"--- Step {step} ---")
        prompt = build_prompt(obs, task_id)
        
        # 2. Generate Action using Inference API
        try:
            response = client.text_generation(prompt, max_new_tokens=128, temperature=0.7)
        except Exception as e:
            print(f"Inference API Error: {e}")
            break
            
        try:
            # Extract JSON from response
            import re
            match = re.search(r'\{.*\}', response.replace('\n', ''))
            if match:
                action_data = json.loads(match.group(0))
            else:
                action_data = json.loads(response)
            action = action_data["action"]
            reasoning = action_data.get("reasoning", "")
        except:
            print(f"Failed to parse model output: {response}")
            action = 8 # Fallback
            reasoning = "Fallback"
            
        print(f"🤖 Model Action: {action} | Reasoning: {reasoning}")
        
        # 3. Send Action to API
        r = httpx.post(f"{API_URL}/step", json={"action": action}, timeout=30.0)
        res = r.json()
        
        obs = res["observation"]
        reward = res["reward"]
        total_reward += reward
        
        print(f"🌍 Env Reward: {reward:.3f} | Carbon: {obs['carbon_emissions_tons']:.0f}t | Budget: ${obs['available_budget']/1000:.0f}K")
        
        if res["terminated"] or res["truncated"]:
            print("\n" + "="*40)
            print("🏁 EPISODE FINISHED")
            print(f"Final Score (Reward): {total_reward:.3f}")
            info = res.get("info", {})
            if "target_carbon" in info:
                 print(f"Final Carbon: {obs['carbon_emissions_tons']} (Target: <= {info['target_carbon']})")
                 print(f"Final Renewable: {obs['renewable_energy_pct']:.1f}% (Target: >= {info['target_renewable']}%)")
            break
            
        step += 1
        time.sleep(1) # Small pause for readability

if __name__ == "__main__":
    test_live()
