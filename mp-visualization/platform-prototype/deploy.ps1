# Motion Pixels — deploy to Vercel (public URL)
# Run this once from inside platform-prototype/:
#   .\deploy.ps1
#
# First run: opens browser for Vercel auth. After that it's instant.

Set-Location $PSScriptRoot
Write-Host "`n motion pixels — deploying to Vercel..." -ForegroundColor Cyan

# Build production bundle first
npm run build
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

# Deploy (--prod = production URL, not preview URL)
npx vercel --prod --yes
