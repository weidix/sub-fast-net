$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PADDLE_PDX_CACHE_HOME = Join-Path $scriptDir ".paddlex_cache"
& (Join-Path $scriptDir ".venv\Scripts\Activate.ps1")
