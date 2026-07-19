$ErrorActionPreference = 'Stop'
$installDir = Join-Path $env:ProgramFiles 'Helix'

Uninstall-ChocolateyPath "$installDir\bin" 'Machine'

if (Test-Path $installDir) {
    Remove-Item $installDir -Recurse -Force
    Write-Host "Helix uninstalled from $installDir"
}
