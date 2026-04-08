@echo off
REM Complete Project Test Script
REM This tests everything end-to-end with your OpenAI API key

echo ======================================================================
echo COMPLETE PROJECT TEST - ESG Compliance Environment
echo ======================================================================
echo.

REM Set environment variables (replace with your actual keys)
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your-huggingface-token-here
set OPENAI_API_KEY=your-openai-api-key-here

echo [1/4] Testing environment (no API needed)...
echo.
python test_final.py
if errorlevel 1 (
    echo.
    echo ERROR: Environment test failed!
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo [2/4] Testing single task with LLM agent (basic_compliance)...
echo This will use OpenAI API and cost ~$0.02
echo ======================================================================
echo.

python test_with_api.py

echo.
echo ======================================================================
echo [3/4] Summary
echo ======================================================================
echo.
echo All tests completed!
echo.
echo Next steps:
echo 1. Deploy to Hugging Face Spaces
echo 2. Set same environment variables in Space settings
echo 3. Submit to hackathon!
echo.
echo ======================================================================
pause
