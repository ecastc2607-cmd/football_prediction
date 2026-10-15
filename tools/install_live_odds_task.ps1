# Registra (o reemplaza) la tarea de Windows que captura cuotas en vivo cada
# 5 minutos: python -m src.capture_live_odds, sin abrir ventana (pythonw).
# Corre solo con la sesión del usuario iniciada y el PC encendido.
#
# Uso:    powershell -ExecutionPolicy Bypass -File tools\install_live_odds_task.ps1
# Quitar: Unregister-ScheduledTask -TaskName "FootballAnalytics-CuotasEnVivo" -Confirm:$false

$ErrorActionPreference = "Stop"
$TaskName = "FootballAnalytics-CuotasEnVivo"
$ProjectDir = Split-Path -Parent $PSScriptRoot

# El mismo intérprete con el que se instalaron las dependencias del proyecto.
$Python = (& python -c "import sys; print(sys.executable)").Trim()
$PythonW = Join-Path (Split-Path -Parent $Python) "pythonw.exe"
if (-not (Test-Path $PythonW)) { $PythonW = $Python }

$Action = New-ScheduledTaskAction -Execute $PythonW -Argument "-m src.capture_live_odds" -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Captura cuotas 365scores (prepartido y en vivo) para football-analytics" -Force | Out-Null

Write-Output "Tarea '$TaskName' registrada: $PythonW -m src.capture_live_odds (cada 5 min) en $ProjectDir"
