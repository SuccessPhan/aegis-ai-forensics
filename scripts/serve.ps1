# Start the AEGIS web-demo inference server (docs/WEB_API_CONTRACT.md).
# Prerequisite: data/cache/serving/aegis_v3_aug.pt (python -m src.learning.export --head-variant aug).
#
# Example (class demo, everyone behind one school Wi-Fi = one public IP):
#   .\scripts\serve.ps1 -Origins "https://<project>.vercel.app,http://localhost:3000" -RateLimit 300 -MaxWaiting 64
#
# To expose it to the Vercel frontend, open a SECOND terminal and run:
#   cloudflared tunnel --url http://localhost:8000
# then set NEXT_PUBLIC_AEGIS_API_URL on Vercel to the printed https://*.trycloudflare.com URL and redeploy.
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Origins = "http://localhost:3000",
    [int]$RateLimit = 300,     # requests / minute / public IP (a whole class may share one IP)
    [int]$MaxWaiting = 64      # queued requests; ~0.1 s each on the RTX 4060
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:AEGIS_ALLOWED_ORIGINS = $Origins
& "$root\.venv\Scripts\python.exe" -m src.serving.app --host $BindHost --port $Port --rate-limit $RateLimit --max-waiting $MaxWaiting
