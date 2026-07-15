<#
.SYNOPSIS
  Wire up a periodic checkpoint backup for an ALREADY-DEPLOYED Prism worker: the worker pushes its own
  model to the anchor on a timer, using a RESTRICTED key that can only write that one backup file.

.DESCRIPTION
  No redeploy. This:
    1. generates a dedicated ed25519 backup key on the worker (kept private on the worker),
    2. installs its PUBLIC key on the anchor under a FORCED COMMAND that can ONLY write
       <AnchorBackupDir>/<node>.bin — it cannot run a shell, read other files, or touch prism-anchor.bin,
    3. installs a systemd timer on the worker that pushes the checkpoint every -BackupEvery,
    4. pushes one backup immediately and verifies it appeared on the anchor.
  Idempotent + multi-node safe: each worker gets a distinct backup file + a distinct authorized_keys line
  (keyed by the node label derived from its IP), so re-running or adding another worker never clobbers.

.EXAMPLE
  .\scripts\setup-worker-backup.ps1 -Server 20.42.102.22
#>
param(
  [Parameter(Mandatory = $true)][string]$Server,
  [string]$User            = "azureuser",
  [string]$Key             = "$env:USERPROFILE\.ssh\prism_anchor",
  [string]$AnchorServer    = "79.72.78.90",
  [string]$AnchorUser      = "ubuntu",
  [string]$AnchorBackupDir = "/home/ubuntu/worker-backups",
  [string]$BackupEvery     = "30min"
)
$ErrorActionPreference = "Stop"
$RemoteHome = "/home/$User"
$RemoteCkpt = "$RemoteHome/.local/share/Prism/prism-headless.bin"
$Node       = "prism-" + (($Server -replace '[^0-9A-Za-z]','-').Trim('-'))
$sshWorker  = @("-i", $Key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15")
$sshAnchor  = $sshWorker

function Remote([string]$cmd) {
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  ssh @sshWorker "$User@$Server" "echo $b64 | base64 -d | bash"
  if ($LASTEXITCODE -ne 0) { throw "worker command failed" }
}
function RemoteOut([string]$cmd) {
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  return (ssh @sshWorker "$User@$Server" "echo $b64 | base64 -d | bash" | Out-String)
}
function AnchorRemote([string]$cmd) {
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd))
  return (ssh @sshAnchor "$AnchorUser@$AnchorServer" "echo $b64 | base64 -d | bash" | Out-String)
}

Write-Host "== worker $Node -> anchor backup (restricted key, every $BackupEvery) ==" -ForegroundColor Cyan

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

# 1. worker: dedicated backup key + script + timer; print the PUBLIC key
Write-Host "-- worker: key + timer" -ForegroundColor DarkCyan
$wb = RemoteOut @"
set -e
mkdir -p ~/.ssh; chmod 700 ~/.ssh
[ -f ~/.ssh/prism_backup ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/prism_backup -C 'prism-backup:$Node' >/dev/null
echo $shB64 | base64 -d > $RemoteHome/prism-backup.sh; chmod +x $RemoteHome/prism-backup.sh
echo $svcB64 | base64 -d | sudo tee /etc/systemd/system/prism-backup.service >/dev/null
echo $tmrB64 | base64 -d | sudo tee /etc/systemd/system/prism-backup.timer >/dev/null
sudo systemctl daemon-reload; sudo systemctl enable --now prism-backup.timer >/dev/null 2>&1
echo '---PUBKEY---'; cat ~/.ssh/prism_backup.pub
"@
$pub = ($wb -split "`n" | Where-Object { $_ -match '^ssh-ed25519 ' } | Select-Object -First 1).Trim()
if (-not $pub) { throw "could not read the worker's backup public key" }

# 2. anchor: install the restricted forced-command key (write-one-file only)
Write-Host "-- anchor: restricted key" -ForegroundColor DarkCyan
$akLine = "command=`"F=$AnchorBackupDir/$Node.bin; cat > `$F.tmp && mv `$F.tmp `$F`",restrict $pub prism-backup:$Node"
$akB64  = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($akLine))
$ak = AnchorRemote @"
set -e
mkdir -p '$AnchorBackupDir'; chmod 700 ~/.ssh 2>/dev/null || true; touch ~/.ssh/authorized_keys
grep -v 'prism-backup:$Node' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp 2>/dev/null || true
mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys
echo $akB64 | base64 -d >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys
echo INSTALLED
"@
if ($ak -notmatch 'INSTALLED') { throw "anchor did not confirm key install" }

# 3. push one backup now + verify it landed
Write-Host "-- first push + verify" -ForegroundColor DarkCyan
Remote "$RemoteHome/prism-backup.sh; echo pushed"
$chk = (AnchorRemote "ls -l '$AnchorBackupDir/$Node.bin' 2>/dev/null || echo MISSING").Trim()
if ($chk -match 'MISSING') { throw "backup file did not appear on the anchor" }

Write-Host "== backup OK ==" -ForegroundColor Green
Write-Host "  $chk"
Write-Host "  target : ${AnchorUser}@${AnchorServer}:$AnchorBackupDir/$Node.bin  (every $BackupEvery)"
Write-Host ("  restore: scp -i `"{0}`" {1}@{2}:{3}/{4}.bin ." -f $Key, $AnchorUser, $AnchorServer, $AnchorBackupDir, $Node) -ForegroundColor DarkCyan
