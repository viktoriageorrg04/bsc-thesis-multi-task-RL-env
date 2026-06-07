param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RangeFromEnvText {
    param(
        [string]$Text,
        [string]$Key
    )

    $pattern = [regex]::Escape($Key) + ":\s*!!python/tuple\s*[\r\n]+\s*-\s*([-+]?\d*\.?\d+)\s*[\r\n]+\s*-\s*([-+]?\d*\.?\d+)"
    $match = [regex]::Match($Text, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($match.Success) {
        return @{
            min = [double]$match.Groups[1].Value
            max = [double]$match.Groups[2].Value
        }
    }
    return $null
}

function Resolve-RunDir {
    param([string]$InputPath)

    $resolved = Resolve-Path -LiteralPath $InputPath
    $item = Get-Item -LiteralPath $resolved

    if ($item.PSIsContainer) {
        if (Test-Path -LiteralPath (Join-Path $item.FullName "params")) {
            return $item.FullName
        }
        throw "Directory does not look like a run directory (missing 'params'): $($item.FullName)"
    }

    if ($item.Name -match "^model_\d+\.pt$") {
        return $item.DirectoryName
    }

    throw "Input must be either a checkpoint file 'model_XXXX.pt' or a run directory."
}

$runDir = Resolve-RunDir -InputPath $Path
$paramsDir = Join-Path $runDir "params"
$envPath = Join-Path $paramsDir "env.yaml"
$agentPath = Join-Path $paramsDir "agent.yaml"
$samplingPath = Join-Path $paramsDir "sampling_profile.json"

if (-not (Test-Path -LiteralPath $envPath)) {
    throw "Missing env config: $envPath"
}

$envText = Get-Content -LiteralPath $envPath -Raw
$agentText = if (Test-Path -LiteralPath $agentPath) { Get-Content -LiteralPath $agentPath -Raw } else { "" }

$experimentDir = Split-Path -Parent $runDir
$family = Split-Path -Leaf $experimentDir
$runName = Split-Path -Leaf $runDir

$linVelY = Get-RangeFromEnvText -Text $envText -Key "lin_vel_y"
$angVelZ = Get-RangeFromEnvText -Text $envText -Key "ang_vel_z"

$taskGuess = "Unknown"
$confidence = "low"

if ($envText -match "envs\.families\.multi_task\.go2_mtl_env_cfg" -or (Test-Path -LiteralPath $samplingPath)) {
    $taskGuess = "MTL-Unified-Unitree-Go2-AllTerrains-v0"
    $confidence = "high"
}
elseif ($envText -match "envs\.families\.agility_terrain\.go2_fam_c_env_cfg" -or $envText -match "gap_terrain") {
    $taskGuess = "MTL-Custom-Gap-Unitree-Go2-C2-v0"
    $confidence = "high"
}
elseif ($envText -match "pyramid_stairs_inv") {
    $taskGuess = "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0"
    $confidence = "high"
}
elseif ($family -eq "unitree_go2_flat") {
    if ($linVelY -and $angVelZ) {
        $isForwardOnly = (
            [Math]::Abs($linVelY.min) -lt 1e-9 -and
            [Math]::Abs($linVelY.max) -lt 1e-9 -and
            [Math]::Abs($angVelZ.min) -lt 1e-9 -and
            [Math]::Abs($angVelZ.max) -lt 1e-9
        )
        if ($isForwardOnly) {
            $taskGuess = "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0"
            $confidence = "medium"
        }
        else {
            $taskGuess = "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0"
            $confidence = "medium"
        }
    }
    else {
        $taskGuess = "Flat family run (A1/A2), exact variant unclear"
        $confidence = "low"
    }
}
elseif ($family -eq "unitree_go2_rough") {
    $taskGuess = "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0 (likely)"
    $confidence = "medium"
}

$stdType = $null
$stdMatch = [regex]::Match($agentText, "std_type:\s*([A-Za-z_]+)")
if ($stdMatch.Success) {
    $stdType = $stdMatch.Groups[1].Value
}

Write-Host "=== Checkpoint/Run Task Info ==="
Write-Host "Run dir : $runDir"
Write-Host "Experiment family : $family"
Write-Host "Run name : $runName"
Write-Host "Task guess : $taskGuess"
Write-Host "Confidence : $confidence"
Write-Host "env.yaml : $envPath"
if (Test-Path -LiteralPath $agentPath) {
    Write-Host "agent.yaml : $agentPath"
}
if (Test-Path -LiteralPath $samplingPath) {
    Write-Host "sampling_profile : $samplingPath"
}
if ($linVelY) {
    Write-Host ("lin_vel_y range : [{0}, {1}]" -f $linVelY.min, $linVelY.max)
}
if ($angVelZ) {
    Write-Host ("ang_vel_z range : [{0}, {1}]" -f $angVelZ.min, $angVelZ.max)
}
if ($stdType) {
    Write-Host "std_type : $stdType"
}
