# Complete Project Test with OpenAI API
# Run this in PowerShell

Write-Host "======================================================================"
Write-Host "🧪 COMPLETE END-TO-END TEST - ESG Compliance Environment"
Write-Host "======================================================================"
Write-Host ""

# Set environment variables (replace with your actual keys)
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your-huggingface-token-here"
$env:OPENAI_API_KEY="your-openai-api-key-here"

Write-Host "✅ Environment variables set:"
Write-Host "   API_BASE_URL: $env:API_BASE_URL"
Write-Host "   MODEL_NAME: $env:MODEL_NAME"
Write-Host "   HF_TOKEN: $($env:HF_TOKEN.Substring(0,15))..."
Write-Host "   OPENAI_API_KEY: sk-proj-***"
Write-Host ""

Write-Host "======================================================================"
Write-Host "[1/2] Testing environment (no API cost)..."
Write-Host "======================================================================"
Write-Host ""

python test_final.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Environment test failed!"
    exit 1
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "[2/2] Testing with OpenAI API (1 task, ~$0.02 cost)..."
Write-Host "======================================================================"
Write-Host ""
Write-Host "Running: basic_compliance task with gpt-4o-mini"
Write-Host "Expected time: 1-2 minutes"
Write-Host ""

python test_with_api.py

Write-Host ""
Write-Host "======================================================================"
Write-Host "✅ ALL TESTS COMPLETE!"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Your environment is ready for deployment! 🚀"
Write-Host ""
