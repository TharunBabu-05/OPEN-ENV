# run_benchmark.ps1
# One-click benchmark runner for Windows
# Runs random + heuristic baselines and generates comparison plots

Write-Host "=== ESG RL Benchmark Runner ===" -ForegroundColor Cyan

$python = ".venv\Scripts\python.exe"

# Step 1: Random baseline
Write-Host "`n[1/4] Running random agent baseline..." -ForegroundColor Yellow
& $python benchmark.py --mode random --seeds 42 43 44 45 46 --output results/baseline_random.json
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Random baseline failed" -ForegroundColor Red; exit 1 }

# Step 2: Heuristic baseline
Write-Host "`n[2/4] Running heuristic agent baseline..." -ForegroundColor Yellow
& $python benchmark.py --mode heuristic --seeds 42 43 44 45 46 --output results/baseline_heuristic.json
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Heuristic baseline failed" -ForegroundColor Red; exit 1 }

# Step 3: Trained model (if available)
$trainedModel = "outputs/esg_rl_model/lora_adapter"
if (Test-Path $trainedModel) {
    Write-Host "`n[3/4] Running trained model benchmark..." -ForegroundColor Yellow
    & $python benchmark.py --mode llm --model_path $trainedModel --seeds 42 43 44 45 46 --output results/trained_v1.json
    if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: Trained model benchmark failed (skipping)" -ForegroundColor Yellow }
} else {
    Write-Host "`n[3/4] No trained model found at $trainedModel (skipping LLM benchmark)" -ForegroundColor Gray
}

# Step 4: Generate plots
Write-Host "`n[4/4] Generating comparison plots..." -ForegroundColor Yellow
$plotArgs = @("--random", "results/baseline_random.json", "--baseline", "results/baseline_heuristic.json", "--output_dir", "results")
if (Test-Path "results/trained_v1.json") {
    $plotArgs += @("--trained", "results/trained_v1.json")
}
& $python plot_results.py @plotArgs
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Plot generation failed" -ForegroundColor Red; exit 1 }

Write-Host "`n=== Benchmark Complete ===" -ForegroundColor Green
Write-Host "Results saved in: results/" -ForegroundColor Cyan
Write-Host "  - results/baseline_random.json" -ForegroundColor White
Write-Host "  - results/baseline_heuristic.json" -ForegroundColor White
Write-Host "  - results/score_comparison.png" -ForegroundColor White
Write-Host "  - results/reward_history.png" -ForegroundColor White
Write-Host "  - results/esg_metrics.png" -ForegroundColor White
