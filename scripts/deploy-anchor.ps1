<#
.SYNOPSIS
  Deploy the current PrismFormer build to the always-on anchor box (Oracle Always-Free).

.DESCRIPTION
  The anchor runs a self-contained single-file binary at /home/ubuntu/prismgym via the
  `prism-anchor` systemd service (there is NO git repo on the box). This script:
    1. publishes a linux-x64 single-file PrismGym from THIS source tree,
    2. scp's it to /tmp on the box,
    3. stops the service, installs the new binary, KEEPS the checkpoint by default (it upgrades in
       place - Shifts/Context grow via LoadUpgrade); pass -WipeCheckpoint only for a hard fork, restarts,
    4. verifies the spec + answering mode it comes up with.

  Requires: dotnet SDK locally; ssh/scp on PATH; the anchor SSH key; passwordless sudo for the
  SSH user on the box (Oracle's default ubuntu user has it).

.EXAMPLE
  .\scripts\deploy-anchor.ps1                  # deploy; KEEP the checkpoint (upgrades in place - the safe default)
  .\scripts\deploy-anchor.ps1 -WipeCheckpoint  # deploy AND wipe the checkpoint (only for a hard fork / clean reset)
  .\scripts\deploy-anchor.ps1 -SkipPublish     # reuse the last publish (fast re-deploy)
#>
param(
  [string]$Server           = "79.72.78.90",
  [string]$User             = "ubuntu",
  [string]$Key              = "$env:USERPROFILE\.ssh\prism_anchor",
  [string]$Service          = "prism-anchor",
  [string]$RemoteBin        = "/home/ubuntu/prismgym",
  [string]$RemoteCheckpoint = "/home/ubuntu/.local/share/Prism/prism-anchor.bin",
  [switch]$SkipPublish,
  [switch]$WipeCheckpoint   # default keeps + upgrades the checkpoint in place; pass this only for a hard fork / clean reset
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$proj = Join-Path $repo "studio\PrismGym\PrismGym.csproj"
$pub  = Join-Path $env:TEMP "prism-anchor-publish"
$sshOpts = @("-i", $Key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15")

# Run a (possibly multi-line) command on the box. Base64-wrapped so PowerShell never mangles quotes/pipes.
# A saturated 1-vCPU box starves sshd (TCP connects, but the SSH banner never lands) — so retry: a short command
# eventually slips through a scheduling gap, and the first thing we do is stop the service, freeing the core.
function Remote([string]$cmd, [int]$tries = 1) {
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  for ($i = 1; $i -le $tries; $i++) {
    ssh @sshOpts "$User@$Server" "echo $b64 | base64 -d | bash"
    if ($LASTEXITCODE -eq 0) { return }
    if ($i -lt $tries) { Write-Host ("  ssh attempt {0}/{1} failed (exit {2}) - retrying..." -f $i, $tries, $LASTEXITCODE) -ForegroundColor DarkYellow; Start-Sleep -Seconds 2 }
  }
  throw "remote command failed after $tries attempt(s)"
}
function Upload([string]$src, [string]$dst, [int]$tries = 1) {
  for ($i = 1; $i -le $tries; $i++) {
    scp @sshOpts $src "$User@${Server}:$dst"
    if ($LASTEXITCODE -eq 0) { return }
    if ($i -lt $tries) { Write-Host ("  scp attempt {0}/{1} failed - retrying..." -f $i, $tries) -ForegroundColor DarkYellow; Start-Sleep -Seconds 2 }
  }
  throw "scp failed after $tries attempt(s)"
}

if (-not $SkipPublish) {
  Write-Host "== publish linux-x64 single-file PrismGym ==" -ForegroundColor Cyan
  dotnet publish $proj -c Release -r linux-x64 --self-contained true -p:PublishSingleFile=true -o $pub --nologo -v q
  if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }
}
$bin = Join-Path $pub "PrismGym"
if (-not (Test-Path $bin)) { throw "published binary not found: $bin (run without -SkipPublish)" }
Write-Host ("binary: {0} ({1:N0} bytes)" -f $bin, (Get-Item $bin).Length)

# STOP FIRST — free the core before we transfer, so the (busy) box can actually service the scp. Retry hard: this short
# command is the one that has to punch through the saturated sshd; once it lands, the service is down and the rest is calm.
Write-Host "== stop service (frees the core so the transfer can land) ==" -ForegroundColor Cyan
Remote "sudo systemctl stop '$Service'; echo '  stopped'" -tries 40

Write-Host "== upload to /tmp/prismgym.new ==" -ForegroundColor Cyan
Upload $bin "/tmp/prismgym.new" -tries 10

$wipe = if ($WipeCheckpoint) { "sudo rm -f '$RemoteCheckpoint'; echo '  WIPED checkpoint (fresh start)'" } else { "echo '  kept checkpoint (upgrades in place)'" }
Write-Host "== install / restart ==" -ForegroundColor Cyan
Remote @"
set -e
sudo mv /tmp/prismgym.new '$RemoteBin'
sudo chown ubuntu:ubuntu '$RemoteBin'
sudo chmod +x '$RemoteBin'
$wipe
sudo systemctl start '$Service'
echo '  service (re)started'
"@ -tries 5

Write-Host "== verify ==" -ForegroundColor Cyan
Start-Sleep -Seconds 4
Remote "echo -n '  active: '; systemctl is-active '$Service'; journalctl -u '$Service' --no-pager -n 12 | grep -iaE 'anchor node . spec|joined the mesh' | tail -3"
Write-Host "== done ==" -ForegroundColor Green
