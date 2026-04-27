import httpx

API_URL = 'https://tharun5054-esg-compliance-env.hf.space'
print(f'Connecting to Live API: {API_URL}')

# 1. Reset Environment
r = httpx.post(f'{API_URL}/reset', json={'task_id': 'aggressive_sustainability', 'seed': 42})
obs = r.json()
print(f'\n[RESET] Month {obs["current_month"]}: Budget ${obs["available_budget"]/1000:.0f}K, Carbon {obs["carbon_emissions_tons"]:.0f}t')

# 2. Take a Step (Action 0 = Install Solar)
print('\nSending Action 0 (Install Solar) to Live API...')
r = httpx.post(f'{API_URL}/step', json={'action': 0})
res = r.json()
new_obs = res['observation']
print(f'[STEP] Reward: +{res["reward"]:.3f} | New Budget: ${new_obs["available_budget"]/1000:.0f}K | Carbon: {new_obs["carbon_emissions_tons"]:.0f}t')
