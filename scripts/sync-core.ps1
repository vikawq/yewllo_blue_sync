$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ConfigPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'sync.config.ps1'
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Missing config file: $ConfigPath"
}
. $ConfigPath

$Script:LocalDir = $SyncSettings.LocalDir
$Script:Branch = $SyncSettings.Branch
$Script:RemoteName = $SyncSettings.RemoteName
$Script:BlueRemoteUrl = $SyncSettings.BlueRemoteUrl
$Script:YellowRemoteUrl = $SyncSettings.YellowRemoteUrl

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "OK  $Message" -ForegroundColor Green
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $Script:LocalDir,
        [switch]$AllowFailure
    )

    Push-Location -LiteralPath $WorkingDirectory
    try {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & git @Arguments
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }

    if ($code -ne 0 -and -not $AllowFailure) {
        throw "Git command failed: git $($Arguments -join ' ')"
    }

    return $code
}

function Get-GitOutput {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $Script:LocalDir,
        [switch]$AllowFailure
    )

    $stdoutFile = New-TemporaryFile
    $stderrFile = New-TemporaryFile

    Push-Location -LiteralPath $WorkingDirectory
    try {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & git @Arguments > $stdoutFile 2> $stderrFile
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }

    $stdout = Get-Content -LiteralPath $stdoutFile -Raw -ErrorAction SilentlyContinue
    $stderr = Get-Content -LiteralPath $stderrFile -Raw -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutFile, $stderrFile -Force -ErrorAction SilentlyContinue

    $output = (($stdout, $stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"

    if ($code -ne 0 -and -not $AllowFailure) {
        throw "Git command failed: git $($Arguments -join ' ')`n$output"
    }

    return @{
        Code = $code
        Text = $output
    }
}

function Assert-GitInstalled {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is not installed or not in PATH. Install Git for Windows first: https://git-scm.com/download/win"
    }
}

function Assert-BlueIdentity {
    $email = (Get-GitOutput -Arguments @('config', '--global', 'user.email') -WorkingDirectory $env:TEMP -AllowFailure).Text.Trim()
    $name = (Get-GitOutput -Arguments @('config', '--global', 'user.name') -WorkingDirectory $env:TEMP -AllowFailure).Text.Trim()

    if ([string]::IsNullOrWhiteSpace($email) -or [string]::IsNullOrWhiteSpace($name)) {
        throw "Blue upload needs Git user.name and user.email. Run: git config --global user.name ""Your Name"" ; git config --global user.email ""you@example.com"""
    }
}

function Test-DirectoryEmpty {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $true
    }

    $item = Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop | Select-Object -First 1
    return $null -eq $item
}

function Ensure-LocalDirectory {
    if (-not (Test-Path -LiteralPath $Script:LocalDir)) {
        New-Item -ItemType Directory -Path $Script:LocalDir | Out-Null
    }
}

function Test-LocalGitRepo {
    return Test-Path -LiteralPath (Join-Path $Script:LocalDir '.git')
}

function Set-RemoteUrl {
    param(
        [Parameter(Mandatory = $true)][string]$FetchUrl,
        [string]$PushUrl
    )

    $remoteCheck = Get-GitOutput -Arguments @('remote', 'get-url', $Script:RemoteName) -AllowFailure
    if ($remoteCheck.Code -eq 0) {
        Invoke-Git -Arguments @('remote', 'set-url', $Script:RemoteName, $FetchUrl) | Out-Null
    }
    else {
        Invoke-Git -Arguments @('remote', 'add', $Script:RemoteName, $FetchUrl) | Out-Null
    }

    if (-not [string]::IsNullOrWhiteSpace($PushUrl)) {
        Invoke-Git -Arguments @('remote', 'set-url', '--push', $Script:RemoteName, $PushUrl) | Out-Null
    }
}

function Initialize-BlueRepo {
    Ensure-LocalDirectory

    if (-not (Test-LocalGitRepo)) {
        Write-Step "Initializing Git repository in $Script:LocalDir"
        Invoke-Git -Arguments @('init') | Out-Null
        Invoke-Git -Arguments @('branch', '-M', $Script:Branch) | Out-Null
    }

    Invoke-Git -Arguments @('config', 'core.quotepath', 'false') | Out-Null
    Set-RemoteUrl -FetchUrl $Script:BlueRemoteUrl -PushUrl $Script:BlueRemoteUrl
}

function Initialize-YellowRepo {
    if (-not (Test-Path -LiteralPath $Script:LocalDir)) {
        Write-Step "Cloning repository to $Script:LocalDir"
        Invoke-Git -Arguments @('clone', '--branch', $Script:Branch, $Script:YellowRemoteUrl, $Script:LocalDir) -WorkingDirectory $env:TEMP | Out-Null
    }
    elseif (-not (Test-LocalGitRepo)) {
        if (Test-DirectoryEmpty -Path $Script:LocalDir) {
            Write-Step "Cloning repository to empty folder $Script:LocalDir"
            Remove-Item -LiteralPath $Script:LocalDir -Force
            Invoke-Git -Arguments @('clone', '--branch', $Script:Branch, $Script:YellowRemoteUrl, $Script:LocalDir) -WorkingDirectory $env:TEMP | Out-Null
        }
        else {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            $backup = "${Script:LocalDir}_before_yellow_sync_$stamp"
            Write-Step "Moving existing non-Git folder to $backup"
            Move-Item -LiteralPath $Script:LocalDir -Destination $backup
            Invoke-Git -Arguments @('clone', '--branch', $Script:Branch, $Script:YellowRemoteUrl, $Script:LocalDir) -WorkingDirectory $env:TEMP | Out-Null
        }
    }

    Invoke-Git -Arguments @('config', 'core.quotepath', 'false') | Out-Null
    Set-RemoteUrl -FetchUrl $Script:YellowRemoteUrl -PushUrl 'DISABLED_YELLOW_DOWNLOAD_ONLY'
}

function Test-RemoteBranch {
    $result = Get-GitOutput -Arguments @('ls-remote', '--heads', $Script:RemoteName, $Script:Branch) -AllowFailure
    return ($result.Code -eq 0 -and -not [string]::IsNullOrWhiteSpace($result.Text))
}

function Test-LocalHasCommit {
    $result = Get-GitOutput -Arguments @('rev-parse', '--verify', 'HEAD') -AllowFailure
    return ($result.Code -eq 0)
}

function Get-StatusText {
    return (Get-GitOutput -Arguments @('status', '--porcelain')).Text.Trim()
}

function Clear-DirectoryExceptGit {
    if (-not (Test-LocalGitRepo)) {
        throw "Refusing to clear $Script:LocalDir because it is not a Git repository."
    }

    $root = (Resolve-Path -LiteralPath $Script:LocalDir).Path.TrimEnd('\')
    if ([string]::IsNullOrWhiteSpace($root) -or $root -match '^[A-Za-z]:$') {
        throw "Refusing to clear unsafe path: $root"
    }

    Get-ChildItem -LiteralPath $root -Force | Where-Object { $_.Name -ne '.git' } | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

function Invoke-BlueSync {
    Assert-GitInstalled
    Assert-BlueIdentity
    Initialize-BlueRepo

    Write-Step "Collecting local changes from $Script:LocalDir"
    Invoke-Git -Arguments @('add', '-A') | Out-Null
    $status = Get-StatusText
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        $message = "Blue sync $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Invoke-Git -Arguments @('commit', '-m', $message) | Out-Null
        Write-Ok "Committed local changes"
    }
    else {
        Write-Ok "No local changes to commit"
    }

    $hasRemoteBranch = Test-RemoteBranch
    if ($hasRemoteBranch) {
        Write-Step "Downloading latest GitHub changes"
        Invoke-Git -Arguments @('fetch', $Script:RemoteName, $Script:Branch) | Out-Null

        if (Test-LocalHasCommit) {
            Invoke-Git -Arguments @('pull', '--rebase', $Script:RemoteName, $Script:Branch) | Out-Null
        }
        else {
            Invoke-Git -Arguments @('checkout', '-B', $Script:Branch, "$Script:RemoteName/$Script:Branch") | Out-Null
        }
        Write-Ok "Downloaded latest changes"
    }
    else {
        Write-Ok "Remote branch does not exist yet; this looks like the first upload"
    }

    if (Test-LocalHasCommit) {
        Write-Step "Uploading to GitHub"
        Invoke-Git -Arguments @('push', '-u', $Script:RemoteName, $Script:Branch) | Out-Null
        Write-Ok "Uploaded to GitHub"
    }
    else {
        Write-Ok "Nothing to upload because there are no files yet"
    }

    Write-Step "Finished"
    Write-Host "Blue sync completed for $Script:LocalDir"
}

function Invoke-YellowDownload {
    Assert-GitInstalled
    Initialize-YellowRepo

    Write-Step "Checking GitHub for latest files"
    Invoke-Git -Arguments @('fetch', $Script:RemoteName, $Script:Branch) | Out-Null

    $hasRemoteBranch = Test-RemoteBranch
    if (-not $hasRemoteBranch) {
        throw "Remote branch '$Script:Branch' does not exist yet. Run blue upload once first."
    }

    Write-Step "Deleting local files before download"
    Clear-DirectoryExceptGit
    Write-Ok "Deleted local files under $Script:LocalDir"

    Write-Step "Downloading latest files to $Script:LocalDir"
    Invoke-Git -Arguments @('checkout', '-B', $Script:Branch, "$Script:RemoteName/$Script:Branch") | Out-Null
    Invoke-Git -Arguments @('reset', '--hard', "$Script:RemoteName/$Script:Branch") | Out-Null
    Invoke-Git -Arguments @('clean', '-fd') | Out-Null
    Write-Ok "Downloaded latest GitHub files"

    Write-Step "Finished"
    Write-Host "Yellow download completed for $Script:LocalDir"
}
