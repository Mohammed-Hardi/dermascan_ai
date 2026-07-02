param(
    [int]$WorkerCount = 24,
    [int]$MaxRetries = 4
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..\..\..\..")).Path
$manifestPath = Join-Path $scriptDir "download_manifest.csv"
$knownUnavailableSourcePaths = @(
    "dataset/images/-2243186711511406658.png"
)

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Manifest not found: $manifestPath"
}

$jobs = for ($workerId = 0; $workerId -lt $WorkerCount; $workerId++) {
    Start-Job -ArgumentList $manifestPath, $projectRoot, $knownUnavailableSourcePaths, $workerId, $WorkerCount, $MaxRetries -ScriptBlock {
        param($ManifestPath, $ProjectRoot, $KnownUnavailableSourcePaths, $WorkerId, $WorkerCount, $MaxRetries)

        Add-Type -AssemblyName System.Net.Http
        $rows = Import-Csv -LiteralPath $ManifestPath
        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromMinutes(5)
        $downloaded = 0
        $failed = [System.Collections.Generic.List[string]]::new()

        try {
            for ($index = $WorkerId; $index -lt $rows.Count; $index += $WorkerCount) {
                $row = $rows[$index]
                $destination = Join-Path $ProjectRoot $row.destination

                if ($KnownUnavailableSourcePaths -contains $row.source_path) {
                    continue
                }

                if ((Test-Path -LiteralPath $destination) -and
                    (Get-Item -LiteralPath $destination).Length -gt 0) {
                    continue
                }

                $parent = Split-Path -Parent $destination
                [System.IO.Directory]::CreateDirectory($parent) | Out-Null
                $succeeded = $false

                for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
                    try {
                        $bytes = $client.GetByteArrayAsync($row.url).GetAwaiter().GetResult()
                        [System.IO.File]::WriteAllBytes($destination, $bytes)
                        $downloaded++
                        $succeeded = $true
                        break
                    }
                    catch {
                        if ($attempt -lt $MaxRetries) {
                            Start-Sleep -Seconds ([Math]::Min(2 * $attempt, 8))
                        }
                    }
                }

                if (-not $succeeded) {
                    $failed.Add($row.destination)
                }
            }
        }
        finally {
            $client.Dispose()
        }

        [pscustomobject]@{
            Worker = $WorkerId
            Downloaded = $downloaded
            Failed = $failed.Count
            FailedPaths = ($failed -join "|")
        }
    }
}

$results = $jobs | Receive-Job -Wait
$jobs | Remove-Job

$results | Sort-Object Worker | Format-Table Worker, Downloaded, Failed -AutoSize
"Downloaded this run: $((($results | Measure-Object Downloaded -Sum).Sum))"
"Failed this run: $((($results | Measure-Object Failed -Sum).Sum))"

$manifest = Import-Csv -LiteralPath $manifestPath
$missing = @($manifest | Where-Object {
    $destination = Join-Path $projectRoot $_.destination
    $_.source_path -notin $knownUnavailableSourcePaths -and (
        -not (Test-Path -LiteralPath $destination) -or
        (Get-Item -LiteralPath $destination).Length -eq 0
    )
})

"Remaining missing files: $($missing.Count)"
if ($missing.Count -gt 0) {
    exit 1
}
