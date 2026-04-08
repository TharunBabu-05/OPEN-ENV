# Quick Test Script - Using OpenRouter API
# USAGE: Set your API keys first, then run .\quick_test_openrouter.ps1

$env:API_BASE_URL="https://openrouter.ai/api/v1"
$env:OPENAI_API_KEY="your-openrouter-api-key-here"
$env:MODEL_NAME="meta-llama/llama-3-8b-instruct"
$env:HF_TOKEN="your-huggingface-token-here"

Write-Host "🚀 Running complete test with OpenRouter API..."
Write-Host "   Using model: meta-llama/llama-3-8b-instruct (via OpenRouter)"
Write-Host ""

python test_with_api.py
