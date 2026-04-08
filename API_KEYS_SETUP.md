# 🚨 CRITICAL: API KEYS REQUIRED FOR HACKATHON

**YOU ARE RIGHT - I SHOULD HAVE TOLD YOU THIS FIRST!**

## ⚠️ MANDATORY ENVIRONMENT VARIABLES

Before running inference, you **MUST** set these:

### 1. For Windows PowerShell:

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your-huggingface-token-here"
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

### 2. What You Need:

| Variable | What It Is | Where to Get It |
|----------|-----------|-----------------|
| `API_BASE_URL` | LLM endpoint | https://api.openai.com/v1 (for OpenAI) |
| `MODEL_NAME` | Model to use | `gpt-4o-mini` or `gpt-4o` |
| `HF_TOKEN` | Hugging Face token | Get from https://huggingface.co/settings/tokens |
| `OPENAI_API_KEY` | OpenAI key | Get from https://platform.openai.com/api-keys |

### 3. Alternative: Use Free Hugging Face Inference

If you don't want to pay for OpenAI, use Hugging Face's free inference:

```powershell
$env:API_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
$env:MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
$env:HF_TOKEN="your-huggingface-token-here"
```

**Note**: Free tier has rate limits but works for testing!

---

## ✅ QUICK TEST BEFORE SUBMISSION

Run these commands in order:

### Step 1: Test environment (NO API needed)
```bash
python test_final.py
```
Expected: ✅ All tests pass

### Step 2: Set your API keys
```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your-huggingface-token-here"
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

### Step 3: Test inference
```bash
python inference.py
```

Expected output:
```json
{"type": "INFO", "message": "Starting ESG Environment Inference"}
{"type": "START", "task_id": "basic_compliance", "max_steps": 6}
{"type": "STEP", "step": 1, "action": 0, "reward": 0.25, ...}
{"type": "STEP", "step": 2, "action": 1, "reward": 0.18, ...}
...
{"type": "END", "task_id": "basic_compliance", "score": 0.85}
{"type": "SUMMARY", "average_score": 0.73}
```

---

## 🚨 WHAT JUDGES WILL CHECK

✅ Your environment works WITHOUT API keys  
✅ Inference works WITH API keys  
✅ Output has [START], [STEP], [END] markers  
✅ Rewards vary (not same every time)  
✅ Score is 0.0-1.0  
✅ Deterministic (same seed = same result)  

---

## 🎯 YOUR NEXT STEPS

1. **Get OpenAI API key** (if you want to use OpenAI):
   - Go to: https://platform.openai.com/api-keys
   - Create new key
   - Add $5-10 credit (enough for testing)

2. **OR use free Hugging Face** (recommended for testing):
   - Get token from: https://huggingface.co/settings/tokens
   - Use free models like Llama-3

3. **Test locally**:
   ```bash
   python test_final.py  # Must pass
   ```

4. **Deploy to HF Spaces**:
   - Upload all files
   - Set environment variables in Space settings
   - Let it auto-build

5. **Submit to hackathon** 🏆

---

## ❓ NEED HELP?

Run this and send me the output:
```bash
python test_final.py
```

I'll tell you exactly what to fix!
