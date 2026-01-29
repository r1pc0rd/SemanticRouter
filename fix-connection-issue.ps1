# Fix MCP Router Connection Issue
# This script kills stale Node.js processes that are blocking port 5598

Write-Host "=== MCP Router Connection Fix ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check for running Node.js processes
Write-Host "Step 1: Checking for Node.js processes..." -ForegroundColor Yellow
$nodeProcesses = Get-Process | Where-Object {$_.ProcessName -like "*node*"}

if ($nodeProcesses) {
    Write-Host "Found $($nodeProcesses.Count) Node.js process(es):" -ForegroundColor Yellow
    $nodeProcesses | Select-Object ProcessName, Id, StartTime | Format-Table -AutoSize
    
    # Step 2: Kill the processes
    Write-Host "Step 2: Stopping Node.js processes..." -ForegroundColor Yellow
    try {
        $nodeProcesses | Stop-Process -Force
        Write-Host "✅ Successfully stopped all Node.js processes" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error stopping processes: $_" -ForegroundColor Red
        Write-Host "You may need to run this script as Administrator" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✅ No Node.js processes found" -ForegroundColor Green
}

Write-Host ""

# Step 3: Verify port is free
Write-Host "Step 3: Checking if port 5598 is free..." -ForegroundColor Yellow
$portCheck = netstat -ano | findstr :5598

if ($portCheck) {
    Write-Host "⚠️  Port 5598 is still in use:" -ForegroundColor Yellow
    Write-Host $portCheck
    Write-Host ""
    Write-Host "You may need to:" -ForegroundColor Yellow
    Write-Host "  1. Close Kiro completely" -ForegroundColor Yellow
    Write-Host "  2. Run this script again" -ForegroundColor Yellow
    Write-Host "  3. Or restart your computer" -ForegroundColor Yellow
} else {
    Write-Host "✅ Port 5598 is free" -ForegroundColor Green
}

Write-Host ""

# Step 4: Instructions
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Close Kiro completely (if running)" -ForegroundColor White
Write-Host "2. Wait 5 seconds" -ForegroundColor White
Write-Host "3. Start Kiro" -ForegroundColor White
Write-Host "4. Check MCP Server view for 'semantic-router'" -ForegroundColor White
Write-Host "5. Status should show 'Connected' or 'Running'" -ForegroundColor White
Write-Host ""
Write-Host "Expected startup time: 10-20 seconds" -ForegroundColor Cyan
Write-Host ""
Write-Host "If you still have issues, see:" -ForegroundColor Yellow
Write-Host "  temp/summaries/connection-issue-diagnosis.md" -ForegroundColor Yellow
Write-Host ""
