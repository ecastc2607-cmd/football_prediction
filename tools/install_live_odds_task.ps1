# Registra (o reemplaza) la tarea de Windows de captura de cuotas en vivo.
#
# Disparador: AL INICIAR SESIÓN (y se arranca ya mismo al instalar). Windows
# solo levanta el proceso; el propio código decide cuándo trabajar
# (python -m src.capture_live_odds --servicio): duerme hasta 3 h antes del
# próximo partido y, dentro de la ventana, captura cada 5 min. Si el proceso
# se cae, Windows lo reinicia (3 intentos, cada 5 min). Sin ventana (pythonw).
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

$Action = New-ScheduledTaskAction -Execute $PythonW -Argument "-m src.capture_live_odds --servicio" -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
# Sin límite de duración (corre mientras dure la sesión); una sola instancia.
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -StartWhenAvailable

# Si ya corría la versión anterior (o el servicio), se detiene antes de reemplazarla.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Captura cuotas 365scores (prepartido y en vivo) para football-analytics; el código decide cuándo trabajar" -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Output "Tarea '$TaskName' registrada (al iniciar sesión) y arrancada: $PythonW -m src.capture_live_odds --servicio"
