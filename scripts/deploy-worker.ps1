<#
.SYNOPSIS
  Stand up a heavy compute WORKER node (e.g. an Azure/Oracle 32-64 vCPU box) on the Prism mesh, seeded
  from the anchor's live model in a way that CANNOT corrupt the worker OR the mesh.

.DESCRIPTION
  A worker runs the self-contained linux-x64 PrismGym as `prismgym headless` under the `prism-worker`
  systemd service. It auto-joins the colony (broker + room default to the anchor, zero config), grinds
  the shipped curriculum on all its cores, donates gradients, and bleeds its improved weights back into
  the swarm. This script:
    1. publishes a linux-x64 single-file PrismGym from THIS source tree,
    2. SAFELY copies the SEED model onto the box:
         - default: pulls the anchor's prism-anchor.bin (atomic snapshot via `cp`, read-only to the
           anchor's live model), verifies its embedded spec signature locally, THEN installs it atomically
           on the worker as prism-headless.bin. Override with -SeedFrom <localFile> to seed from a local
           checkpoint (e.g. your prism.bin) instead.
    3. ships the curriculum (studio/PrismStudio/data) as the worker's training corpus,
    4. writes + (re)starts the prism-worker service,
    5. GATES on a verified load: the worker must log `loaded`/`seeded` with a spec matching the anchor's.
       If it comes up `fresh` (seed did not take), the service is STOPPED before its first weight-bleed and
       the deploy aborts — so a mis-seeded node can never dilute the mesh.

  The self-contained publish bundles the .NET runtime, so the box needs NO dotnet install — just a stock
  Ubuntu with passwordless sudo for the SSH user (Azure `azureuser` / Oracle `ubuntu` both have it).

  Requires locally: dotnet SDK; ssh/scp/tar on PATH; the SSH key; reachability to BOTH the worker and
  (for the default anchor seed) the anchor.

.EXAMPLE
  .\scripts\deploy-worker.ps1 -Server 20.0.0.5 -User azureuser        # Azure box, seed from the anchor
  .\scripts\deploy-worker.ps1 -Server 20.0.0.5 -SeedFrom "$env:LOCALAPPDATA\Prism\prism.bin"  # seed from local
  .\scripts\deploy-worker.ps1 -Server 20.0.0.5 -SkipPublish           # reuse the last publish (fast redeploy)
  # PASSIVE weight-share hub (no training): runs `anchor` mode, holds a copy of YOUR model, bleeds/absorbs weights only.
  .\scripts\deploy-worker.ps1 -Server 20.42.102.22 -Passive -SeedFrom "$env:LOCALAPPDATA\Prism\prism.bin"
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$Server,                                                   # the worker box public IP
  [string]$User             = "azureuser",                           # Azure default admin (Oracle would be 'ubuntu')
  [string]$Key              = "$env:USERPROFILE\.ssh\prism_anchor",  # reuse the anchor key
  [string]$SeedFrom         = "anchor",                              # 'anchor' (pull live model) or a local checkpoint path
  [string]$AnchorServer     = "79.72.78.90",
  [string]$AnchorUser       = "ubuntu",
  [string]$AnchorCheckpoint = "/home/ubuntu/.local/share/Prism/prism-anchor.bin",
  [string]$Service          = "prism-worker",
  [switch]$SkipPublish,
  [switch]$Passive,                                                 # PASSIVE weight-share only: run `anchor` (no curriculum training), just hold the model + bleed/absorb weights
  [switch]$NoBackup,                                                 # skip the periodic worker->anchor checkpoint backup
  [string]$AnchorBackupDir  = "/home/ubuntu/worker-backups",        # anchor folder for worker backups (NEVER the anchor's own model)
  [string]$BackupEvery      = "30min"                               # systemd timer interval for the backup push
)
$ErrorActionPreference = "Stop"
$repo    = Split-Path $PSScriptRoot -Parent
$proj    = Join-Path $repo "studio\PrismGym\PrismGym.csproj"
$dataSrc = Join-Path $repo "studio\PrismStudio\data"
$pub     = Join-Path $env:TEMP "prism-worker-publish"
$localSeed = Join-Path $env:TEMP "prism-worker-seed.bin"
$localData = Join-Path $env:TEMP "prism-worker-data.tgz"

$RemoteHome = "/home/$User"
$RemoteBin  = "$RemoteHome/prismgym"
$RemoteDir  = "$RemoteHome/.local/share/Prism"          # StudioModel's LocalApplicationData/Prism
$RemoteCkpt = "$RemoteDir/prism-headless.bin"           # HeadlessNode loads this first + saves progress here
$RemoteData = "$RemoteHome/prism-data"                  # passed as the dataDir arg (pairs/ text/ …)
$Node       = "prism-" + (($Server -replace '[^0-9A-Za-z]','-').Trim('-'))   # unique backup label per box (from its IP)

# explicit colony room code (base64 of "PZR1;<room>") — passed to headless so systemd never has to quote an empty arg
$RoomCode = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("PZR1;prism-colony"))

# PASSIVE mode: run `anchor` (holds the model, bleeds/absorbs weight slices, NO curriculum training). anchor loads/saves
# prism-anchor.bin (not prism-headless.bin), takes only the room arg, and needs no curriculum corpus.
if ($Passive) {
  $RemoteCkpt = "$RemoteDir/prism-anchor.bin"
  $ExecArgs   = "anchor ${RoomCode}"
  $SvcDesc    = "Prism passive node (weight-share only, no training)"
} else {
  $ExecArgs   = "headless ${RoomCode} ${RemoteData}"
  $SvcDesc    = "Prism worker node (headless swarm trainer)"
}

$sshWorker = @("-i", $Key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15")
$sshAnchor = $sshWorker

function Remote([string]$cmd, [int]$tries = 3) {          # run a (multi-line) command on the WORKER; base64-wrapped so quotes/pipes survive
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  for ($i = 1; $i -le $tries; $i++) {
    ssh @sshWorker "$User@$Server" "echo $b64 | base64 -d | bash"
    if ($LASTEXITCODE -eq 0) { return }
    if ($i -lt $tries) { Write-Host ("  ssh {0}/{1} failed (exit {2}) - retrying..." -f $i, $tries, $LASTEXITCODE) -ForegroundColor DarkYellow; Start-Sleep -Seconds 2 }
  }
  throw "remote command failed after $tries attempt(s)"
}
function RemoteOut([string]$cmd) {                         # capture combined output of a WORKER command
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  return (ssh @sshWorker "$User@$Server" "echo $b64 | base64 -d | bash" | Out-String)
}
function Upload([string]$src, [string]$dst, [int]$tries = 4) {
  for ($i = 1; $i -le $tries; $i++) {
    scp @sshWorker $src "$User@${Server}:$dst"
    if ($LASTEXITCODE -eq 0) { return }
    if ($i -lt $tries) { Write-Host ("  scp {0}/{1} failed - retrying..." -f $i, $tries) -ForegroundColor DarkYellow; Start-Sleep -Seconds 2 }
  }
  throw "scp failed after $tries attempt(s)"
}
function AnchorRemote([string]$cmd) {                     # run a command on the ANCHOR (base64-wrapped), capture output
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  return (ssh @sshAnchor "$AnchorUser@$AnchorServer" "echo $b64 | base64 -d | bash" | Out-String)
}

# Read the spec signature embedded at the head of a checkpoint (BinaryWriter length-prefixed string:
# 1 length byte for our <128-char sig, then ASCII). Returns the signature or throws if it isn't a Prism file.
function CheckpointSignature([string]$path) {
  if (-not (Test-Path $path)) { throw "checkpoint not found: $path" }
  $b = [System.IO.File]::ReadAllBytes($path)
  if ($b.Length -lt 24) { throw "checkpoint too small ($($b.Length) bytes) - not a real model: $path" }
  $len = $b[0]
  if ($len -lt 10 -or $len -gt 120 -or $b.Length -lt (1 + $len)) { throw "bad signature header in $path" }
  $sig = [System.Text.Encoding]::UTF8.GetString($b, 1, $len)
  if (-not $sig.StartsWith("PRISM-")) { throw "not a PRISM checkpoint (sig='$sig'): $path" }
  return $sig
}

# ── 1. publish ────────────────────────────────────────────────────────────────────────────────────────
if (-not $SkipPublish) {
  Write-Host "== publish linux-x64 single-file PrismGym ==" -ForegroundColor Cyan
  dotnet publish $proj -c Release -r linux-x64 --self-contained true -p:PublishSingleFile=true -o $pub --nologo -v q
  if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }
}
$bin = Join-Path $pub "PrismGym"
if (-not (Test-Path $bin)) { throw "published binary not found: $bin (run without -SkipPublish)" }
Write-Host ("binary: {0} ({1:N0} bytes)" -f $bin, (Get-Item $bin).Length)

# ── 2. obtain + verify the SEED model ───────────────────────────────────────────────────────────────────
if ($SeedFrom -eq "anchor") {
  Write-Host "== pull the anchor's live model (atomic snapshot, read-only to the anchor) ==" -ForegroundColor Cyan
  # cp on the anchor gives us a STABLE snapshot: the anchor writes prism-anchor.bin atomically (File.Replace),
  # so this copy is always a whole checkpoint even if the anchor saves mid-transfer. We never write the anchor.
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("set -e; cp -f '$AnchorCheckpoint' /tmp/prism-seed.bin; ls -l /tmp/prism-seed.bin"))
  ssh @sshAnchor "$AnchorUser@$AnchorServer" "echo $b64 | base64 -d | bash"
  if ($LASTEXITCODE -ne 0) { throw "could not snapshot the anchor checkpoint (is the anchor up / key valid?)" }
  if (Test-Path $localSeed) { Remove-Item $localSeed -Force }
  scp @sshAnchor "$AnchorUser@${AnchorServer}:/tmp/prism-seed.bin" $localSeed
  if ($LASTEXITCODE -ne 0) { throw "scp of the anchor seed failed" }
} else {
  Write-Host "== seed from local checkpoint: $SeedFrom ==" -ForegroundColor Cyan
  if (-not (Test-Path $SeedFrom)) { throw "seed file not found: $SeedFrom" }
  Copy-Item $SeedFrom $localSeed -Force
}
$seedSig = CheckpointSignature $localSeed
Write-Host ("  seed OK - spec {0}, {1:N0} bytes" -f $seedSig, (Get-Item $localSeed).Length) -ForegroundColor Green

# ── 3. package the curriculum (skipped for passive nodes — they never train) ────────────────────────────
if (-not $Passive) {
  Write-Host "== package curriculum (studio/PrismStudio/data) ==" -ForegroundColor Cyan
  if (-not (Test-Path $dataSrc)) { throw "curriculum folder not found: $dataSrc" }
  if (Test-Path $localData) { Remove-Item $localData -Force }
  tar -czf $localData -C $dataSrc .          # archive the CONTENTS (pairs/ text/ gossip/) so it expands into prism-data/
  if ($LASTEXITCODE -ne 0) { throw "tar of the curriculum failed" }
  Write-Host ("  data: {0:N0} bytes" -f (Get-Item $localData).Length)
} else {
  Write-Host "== passive node: skipping curriculum (no training) ==" -ForegroundColor DarkGray
}

# ── 4. stop service (frees the binary lock), then upload everything ──────────────────────────────────────
Write-Host "== stop any running worker + prep dirs ==" -ForegroundColor Cyan
Remote "sudo systemctl stop '$Service' 2>/dev/null; mkdir -p '$RemoteDir' '$RemoteData'; echo '  ready'"

Write-Host "== upload binary + seed$(if (-not $Passive) {' + curriculum'}) ==" -ForegroundColor Cyan
Upload $bin       "/tmp/prismgym.new"
Upload $localSeed "/tmp/prism-seed.bin"
if (-not $Passive) { Upload $localData "/tmp/prism-data.tgz" }

# ── 5. install (atomic seed install), write unit, start ─────────────────────────────────────────────────
Write-Host "== install + (re)start service ==" -ForegroundColor Cyan
# curriculum extract only for training workers; passive nodes get an empty block
$CurriculumBlock = if ($Passive) { "" } else { "tar -xzf /tmp/prism-data.tgz -C '$RemoteData'`nchown -R ${User}:${User} '$RemoteData'" }
Remote @"
set -e
# binary
sudo mv /tmp/prismgym.new '$RemoteBin'
sudo chown ${User}:${User} '$RemoteBin'
sudo chmod +x '$RemoteBin'
# seed model — ATOMIC install: land in the Prism dir then rename (same-fs), so the service never sees a half-file
cp -f /tmp/prism-seed.bin '$RemoteDir/.seed.tmp'
mv -f '$RemoteDir/.seed.tmp' '$RemoteCkpt'
chown ${User}:${User} '$RemoteCkpt'
$CurriculumBlock
# systemd unit — auto-joins the colony (broker+room default to the anchor)
sudo tee /etc/systemd/system/${Service}.service >/dev/null <<UNIT
[Unit]
Description=${SvcDesc}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${User}
Environment=DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
Environment=PRISM_EVALAPP_KEY=
ExecStart=${RemoteBin} ${ExecArgs}
Restart=always
RestartSec=5
Nice=0

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable '$Service' >/dev/null 2>&1
sudo systemctl start '$Service'
echo '  service started'
"@

# ── 6. VERIFY THE SEED LOADED — the anti-dilution gate ──────────────────────────────────────────────────
Write-Host "== verify the seed loaded (before the first weight-bleed) ==" -ForegroundColor Cyan
Start-Sleep -Seconds 8
$log = RemoteOut "journalctl -u '$Service' --no-pager -n 80 | tr -d '\r'"
$active = (RemoteOut "systemctl is-active '$Service'").Trim()
Write-Host "  service: $active"

if ($log -match 'fresh model') {
  Remote "sudo systemctl stop '$Service'; echo '  STOPPED (mis-seed)'"
  Write-Host $log -ForegroundColor DarkGray
  throw "ABORTED: worker came up FRESH (seed did not load) - stopped it before it could bleed init-noise into the mesh. Check the seed's spec ($seedSig) matches the deployed build."
}
if ($log -notmatch 'loaded|seeded from') {
  Remote "sudo systemctl stop '$Service'; echo '  STOPPED (unconfirmed)'"
  Write-Host $log -ForegroundColor DarkGray
  throw "ABORTED: could not confirm the seed loaded within 8s - stopped the service to be safe. Journal above."
}

# spec cross-check: the worker's reported spec must match the seed we shipped
$workerSpec = ([regex]::Match($log, 'spec\s+(PRISM-\S+)')).Groups[1].Value
if ($workerSpec -and $workerSpec -ne $seedSig) {
  Write-Host ("  NOTE: worker spec {0} vs seed {1} - upgrade-in-place path (Shifts/Context grew); OK if intended." -f $workerSpec, $seedSig) -ForegroundColor DarkYellow
}

Write-Host "== worker is on the mesh ==" -ForegroundColor Green
($log -split "`n" | Select-String -Pattern 'spec |loaded|seeded from|joined swarm|params' | Select-Object -Last 6) | ForEach-Object { Write-Host "  $_" }

# ── 7. DURABLE BACKUP: worker pushes its OWN checkpoint to the anchor (restricted key; never touches prism-anchor.bin) ──
if (-not $NoBackup -and -not $Passive) {
  Write-Host "== set up periodic backup (worker -> anchor, restricted key) ==" -ForegroundColor Cyan
  try {
    $backupSh = @"
#!/bin/bash
CKPT="$RemoteCkpt"
[ -f "`$CKPT" ] || exit 0
ssh -i "$RemoteHome/.ssh/prism_backup" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 $AnchorUser@$AnchorServer < "`$CKPT"
"@
    $svcUnit = @"
[Unit]
Description=Prism worker checkpoint backup to anchor
[Service]
Type=oneshot
User=$User
ExecStart=$RemoteHome/prism-backup.sh
"@
    $tmrUnit = @"
[Unit]
Description=Periodic Prism worker checkpoint backup
[Timer]
OnBootSec=3min
OnUnitActiveSec=$BackupEvery
Persistent=true
[Install]
WantedBy=timers.target
"@
    $shB64  = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($backupSh))
    $svcB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($svcUnit))
    $tmrB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($tmrUnit))

    # 7a. worker: dedicated backup key + script + timer; print the PUBLIC key
    $wbSetup = RemoteOut @"
set -e
mkdir -p ~/.ssh; chmod 700 ~/.ssh
[ -f ~/.ssh/prism_backup ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/prism_backup -C 'prism-backup:$Node' >/dev/null
echo $shB64 | base64 -d > $RemoteHome/prism-backup.sh; chmod +x $RemoteHome/prism-backup.sh
echo $svcB64 | base64 -d | sudo tee /etc/systemd/system/prism-backup.service >/dev/null
echo $tmrB64 | base64 -d | sudo tee /etc/systemd/system/prism-backup.timer >/dev/null
sudo systemctl daemon-reload; sudo systemctl enable --now prism-backup.timer >/dev/null 2>&1
echo '---PUBKEY---'; cat ~/.ssh/prism_backup.pub
"@
    $pub = ($wbSetup -split "`n" | Where-Object { $_ -match '^ssh-ed25519 ' } | Select-Object -First 1).Trim()
    if (-not $pub) { throw "could not read the worker's backup public key" }

    # 7b. anchor: install the key under a FORCED COMMAND that can ONLY write this node's backup file
    $akLine = "command=`"F=$AnchorBackupDir/$Node.bin; cat > `$F.tmp && mv `$F.tmp `$F`",restrict $pub prism-backup:$Node"
    $akB64  = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($akLine))
    $akRes = AnchorRemote @"
set -e
mkdir -p '$AnchorBackupDir'; chmod 700 ~/.ssh 2>/dev/null || true; touch ~/.ssh/authorized_keys
grep -v 'prism-backup:$Node' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp 2>/dev/null || true
mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys
echo $akB64 | base64 -d >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys
echo INSTALLED
"@
    if ($akRes -notmatch 'INSTALLED') { throw "anchor did not confirm key install" }

    # 7c. push one backup now + 7d. verify it landed on the anchor
    Remote "$RemoteHome/prism-backup.sh; echo '  first backup pushed'"
    $chk = (AnchorRemote "ls -l '$AnchorBackupDir/$Node.bin' 2>/dev/null || echo MISSING").Trim()
    if ($chk -match 'MISSING') { throw "backup file did not appear on the anchor" }
    Write-Host "  backup OK -> ${AnchorUser}@${AnchorServer}:$AnchorBackupDir/$Node.bin (every $BackupEvery)" -ForegroundColor Green
    Write-Host "  $chk"
    Write-Host ("  restore:  scp -i `"{0}`" {1}@{2}:{3}/{4}.bin ." -f $Key, $AnchorUser, $AnchorServer, $AnchorBackupDir, $Node) -ForegroundColor DarkCyan
  } catch {
    Write-Host ("  WARNING: backup setup failed ({0})." -f $_.Exception.Message) -ForegroundColor Yellow
    Write-Host "  Worker is TRAINING + MESHING FINE - only the off-box backup is missing. Rerun to retry." -ForegroundColor Yellow
  }
}

# ── 8. confirm live participation: mesh + group chat + distil (bleed timer fires ~30s after start) ──
Write-Host "== participation (mesh / group chat / distil) ==" -ForegroundColor Cyan
$act = RemoteOut "journalctl -u '$Service' --no-pager -n 140 | tr -d '\r'"
$hits = ($act -split "`n") | Select-String -Pattern 'joined swarm|hello|peer|\[bleed\]|group|distil|mesh'
if ($hits) { $hits | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  (no mesh lines yet - the bleed/group/distil timer fires ~30s in; watch the live log)" -ForegroundColor DarkGray }
Write-Host "  headless nodes serve group chat, learn from queries they're asked, and distil peers on the bleed timer." -ForegroundColor DarkGray

Write-Host ("watch it:  ssh -i `"{0}`" {1}@{2} 'journalctl -u {3} -f'" -f $Key, $User, $Server, $Service) -ForegroundColor Cyan
Write-Host "== done ==" -ForegroundColor Green
