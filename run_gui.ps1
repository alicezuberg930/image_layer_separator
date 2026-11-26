param(
    [switch]$Reload
)

Set-Location $PSScriptRoot
$runGuiPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $runGuiPython -PathType Leaf)) {
    throw 'The Python environment is missing. Run install.ps1 first.'
}

$runGuiArguments = @('-u', (Join-Path $PSScriptRoot 'demo.py'))
if ($Reload) {
    $runGuiArguments += '--reload'
}

& $runGuiPython @runGuiArguments @args
exit $LASTEXITCODE
