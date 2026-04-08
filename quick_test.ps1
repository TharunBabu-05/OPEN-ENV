# Quick Test Script - Just set environment variables and run test_with_api.py
# USAGE: .\quick_test.ps1

$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXX"
$env:OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXX"

Write-Host "🚀 Running complete test with OpenAI API..."
Write-Host ""

python test_with_api.py
