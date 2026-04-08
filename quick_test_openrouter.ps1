# Quick Test Script - Using OpenRouter API
# USAGE: Set your API keys first, then run .\quick_test_openrouter.ps1

$env:API_BASE_URL="https://openrouter.ai/api/v1"
$env:OPENAI_API_KEY="YOUR_OPENROUTER_API_KEY"
$env:MODEL_NAME="meta-llama/llama-3-8b-instruct"
$env:HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

if ($env:OPENAI_API_KEY -like "YOUR_*") {
	Write-Error "Set OPENAI_API_KEY to a real key before running this script."
	exit 1
}

if ($env:HF_TOKEN -like "YOUR_*") {
	Write-Error "Set HF_TOKEN to a real token before running this script."
	exit 1
}

Write-Host "🚀 Running complete test with OpenRouter API..."
Write-Host "   Using model: meta-llama/llama-3-8b-instruct (via OpenRouter)"
Write-Host ""

python test_with_api.py
