import httpx
import json

# Change this to "http://127.0.0.1:7860" if you are running the API locally!
API_URL = "https://tharun5054-esg-compliance-env.hf.space"

ACTIONS = {
    0: "INSTALL_SOLAR_PANELS",
    1: "UPGRADE_HVAC_EFFICIENCY",
    2: "IMPLEMENT_RECYCLING_PROGRAM",
    3: "WATER_CONSERVATION_INITIATIVE",
    4: "CARBON_OFFSET_PURCHASE",
    5: "DIVERSITY_HIRING_INITIATIVE",
    6: "EMPLOYEE_WELLNESS_PROGRAM",
    7: "COMPLIANCE_AUDIT",
    8: "DO_NOTHING"
}

def play_game():
    print(f"Connecting to Environment API: {API_URL}")
    print("\nStarting new game session...")
    
    # 1. Reset
    r = httpx.post(f"{API_URL}/reset", json={"task_id": "medium"}, timeout=30.0)
    data = r.json()
    session_id = data.get("session_id", "demo-session")
    obs = data.get("observation", data)
    
    print("\n" + "="*50)
    print("🌍 ESG COMPLIANCE SIMULATOR - INTERACTIVE MODE")
    print("="*50)
    
    step = 1
    total_reward = 0.0
    
    while True:
        print(f"\n--- MONTH {obs['current_month']} ---")
        print(f"Budget Remaining: ${obs['available_budget']/1000:.0f}K")
        print(f"Carbon Emissions: {obs['carbon_emissions_tons']:.0f} tons")
        print(f"Renewable Energy: {obs['renewable_energy_pct']:.1f}%")
        print(f"Diversity Score:  {obs['diversity_score']:.1f}")
        print(f"Violations:       {obs['compliance_violations']}")
        print("\nAvailable Actions:")
        for k, v in ACTIONS.items():
            print(f"  [{k}] {v}")
            
        # Get User Input
        try:
            action_input = input("\nEnter Action ID (0-8) or 'q' to quit: ")
            if action_input.lower() == 'q':
                break
            action = int(action_input)
            if action < 0 or action > 8:
                print("Invalid action! Must be 0-8.")
                continue
        except ValueError:
            print("Please enter a valid number.")
            continue
            
        print(f"\nExecuting: {ACTIONS[action]}...")
        
        # 2. Step
        if "session_id" in data:
            r = httpx.post(f"{API_URL}/step", json={"session_id": session_id, "action": action}, timeout=30.0)
        else:
            r = httpx.post(f"{API_URL}/step", json={"action": action}, timeout=30.0)
            
        res = r.json()
        obs = res["observation"]
        reward = res["reward"]
        total_reward += reward
        
        print(f"\n✅ Step Complete! Reward: {reward:+.3f}")
        
        if res["terminated"] or res["truncated"]:
            print("\n" + "="*50)
            print("🏁 GAME OVER!")
            print(f"Final Score: {total_reward:.3f}")
            print(f"Final Month: {obs['current_month']}")
            print("="*50)
            break

if __name__ == "__main__":
    play_game()
