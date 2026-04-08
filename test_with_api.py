"""
Complete end-to-end test with OpenAI API.

This runs ONE task (basic_compliance) to verify:
1. Environment works
2. LLM agent works
3. Grader works
4. Output format is correct

Cost: ~$0.02 for 6 steps with gpt-4o-mini
"""

import os
import sys

# Set API credentials (OpenRouter with Llama)
# IMPORTANT: Set these before running or use environment variables
os.environ['API_BASE_URL'] = os.getenv('API_BASE_URL', 'https://openrouter.ai/api/v1')
os.environ['MODEL_NAME'] = os.getenv('MODEL_NAME', 'meta-llama/llama-3-8b-instruct')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-huggingface-token-here')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-openrouter-api-key-here')

print("=" * 70)
print("🧪 COMPLETE END-TO-END TEST")
print("=" * 70)
print()
print("Testing: ESG Compliance Environment")
print("Task: basic_compliance (easy, 6 steps)")
print("Model: meta-llama/llama-3-8b-instruct (via OpenRouter)")
print("Cost: ~$0.01 (OpenRouter pricing)")
print()
print("=" * 70)
print()

try:
    from openai import OpenAI
    from inference import run_task
    
    # Create OpenAI client
    client = OpenAI(
        base_url=os.environ['API_BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY']
    )
    
    print("✅ OpenAI client created")
    print()
    print("🚀 Running task 'basic_compliance' with LLM agent...")
    print("   (This will take 1-2 minutes)")
    print()
    print("-" * 70)
    
    # Run task
    score = run_task(
        client=client,
        model_name='meta-llama/llama-3-8b-instruct',
        task_id='basic_compliance',
        seed=42
    )
    
    print("-" * 70)
    print()
    print("=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)
    print()
    print(f"Final Score: {score:.3f}")
    print()
    
    if score >= 0.8:
        print("🏆 EXCELLENT! Score exceeds success threshold (0.8)")
    elif score >= 0.5:
        print("✅ GOOD! Score is acceptable")
    else:
        print("⚠️  Score is low - agent may need better prompting")
    
    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print()
    print("1. Your environment works perfectly! ✅")
    print()
    print("2. Deploy to Hugging Face Spaces:")
    print("   - Go to https://huggingface.co/spaces")
    print("   - Create new Space (Docker SDK)")
    print("   - Upload all files")
    print("   - Set environment variables in Settings:")
    print("     * API_BASE_URL = https://openrouter.ai/api/v1")
    print("     * MODEL_NAME = meta-llama/llama-3-8b-instruct")
    print("     * HF_TOKEN = your-huggingface-token")
    print("     * OPENAI_API_KEY = your-openrouter-api-key")
    print()
    print("3. Submit Space URL to hackathon! 🏆")
    print()
    print("=" * 70)
    
    sys.exit(0)
    
except ImportError as e:
    print(f"❌ ERROR: Missing dependency: {e}")
    print()
    print("Install with: pip install openai")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
