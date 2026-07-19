$ErrorActionPreference = 'Stop'

$packageName = 'helix'
$version     = '1.0.0'
$installDir  = Join-Path $env:ProgramFiles 'Helix'
$arch        = if ([System.Environment]::Is64BitOperatingSystem -and
                   [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq 'Arm64') {
                   'windows-arm64'
               } else {
                   'windows-x86_64'
               }

$url         = "https://github.com/anemoi-ai/helix/releases/download/v$version/libhelix-$version-$arch-cpu.zip"
$checksum    = 'PLACEHOLDER_CHECKSUM_UPDATED_AT_RELEASE'

$packageArgs = @{
    packageName    = $packageName
    unzipLocation  = $installDir
    url64          = $url
    checksum64     = $checksum
    checksumType64 = 'sha256'
}

Install-ChocolateyZipPackage @packageArgs

# Add to PATH for current machine
Install-ChocolateyPath "$installDir\bin" 'Machine'

Write-Host "Helix $version installed to $installDir"
Write-Host "Run 'helix-doctor' to verify the installation."
