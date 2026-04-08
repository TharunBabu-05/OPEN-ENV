# How to Test Your Environment - STEP BY STEP

## 🎯 Goal: Make sure everything works before submission

---

## ✅ STEP 1: Test Environment (No API needed)

Open PowerShell and run:

```powershell
cd c:\Users\tharu\open_ENV
python test_final.py
```

**Expected output:**
```
🔍 OPENENV HACKATHON - FINAL VALIDATION
======================================================================
✅ PASS: Environment created and reset (no API needed)
✅ PASS: Step executed (reward=0.123)
✅ PASS: Second step (reward=0.089)
✅ PASS: Rewards vary based on actions ✓
✅ PASS: Grader returns valid score (0.234)
...
✅ PRE-SUBMISSION VALIDATION COMPLETE
```

**If this fails**: Send me the error message!

---

## ✅ STEP 2: Set Environment Variables

### Option A: Use OpenAI (costs money, ~$0.50 for full test)

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXX"
$env:HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXX"
```

Where to get OpenAI key:
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. Add $5-10 to your account

### Option B: Use Hugging Face (FREE but slower)

```powershell
$env:API_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct/v1"
$env:MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
$env:HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXX"
```

**Note**: Free tier has rate limits. If you get errors, wait 1 minute and try again.

---

## ✅ STEP 3: Test Inference (optional but recommended)

```powershell
python inference.py
```

**What you should see:**
```json
{"type": "INFO", "message": "Starting ESG Environment Inference"}
{"type": "START", "task_id": "basic_compliance", ...}
{"type": "STEP", "step": 1, "action": 0, "reward": 0.25, ...}
{"type": "STEP", "step": 2, "action": 1, "reward": 0.18, ...}
...
{"type": "END", "score": 0.85, ...}
{"type": "SUMMARY", "average_score": 0.73}
```

**Good signs:**
- ✅ Each step shows different reward
- ✅ Final score between 0.5-1.0
- ✅ No errors

**Bad signs:**
- ❌ All rewards the same (0.0 or 1.0)
- ❌ API timeout errors
- ❌ Python crashes

---

## ✅ STEP 4: Deploy to Hugging Face Spaces

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - Name: `esg-compliance-env`
   - SDK: **Docker**
   - Visibility: Public

4. Upload these files:
   - ✅ Dockerfile
   - ✅ models.py
   - ✅ env.py
   - ✅ tasks.py
   - ✅ inference.py
   - ✅ openenv.yaml
   - ✅ requirements.txt
   - ✅ pyproject.toml
   - ✅ README.md

5. Go to "Settings" → "Variables and secrets"
6. Add these secrets:
   - `API_BASE_URL`: Your API endpoint
   - `MODEL_NAME`: Your model name
   - `HF_TOKEN`: Your Hugging Face token (starts with hf_)
   - `OPENAI_API_KEY`: Your OpenAI key (if using OpenAI)

7. Wait for build to complete (5-10 minutes)

8. Check logs - should see same output as local test

---

## ✅ STEP 5: Submit to Hackathon

1. Copy your Hugging Face Space URL
   - Example: `https://huggingface.co/spaces/your-username/esg-compliance-env`

2. Go to hackathon submission form

3. Paste URL

4. Submit! 🏆

---

## 🚨 TROUBLESHOOTING

### Problem: "Module not found" error
**Solution**: Make sure you're in the right directory
```powershell
cd c:\Users\tharu\open_ENV
python -m pip install -r requirements.txt
```

### Problem: "API_BASE_URL not set" error
**Solution**: Set environment variables again (they reset when you close PowerShell)

### Problem: "OpenAI API key invalid"
**Solution**: 
1. Check you copied the full key (starts with `sk-`)
2. Make sure you added billing to your OpenAI account
3. Try Hugging Face option instead (free)

### Problem: "Rate limit exceeded" (Hugging Face)
**Solution**: Wait 60 seconds and try again, or switch to OpenAI

### Problem: All rewards are 0.0
**Solution**: This is a bug - send me your `env.py` file and I'll fix it

---

## 📞 STILL STUCK?

Run this and send me the output:

```powershell
python test_final.py > test_results.txt 2>&1
```

Then show me `test_results.txt` and I'll help immediately!
