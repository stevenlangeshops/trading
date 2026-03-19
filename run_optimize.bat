@echo off
REM ============================================================
REM  trading_v2 - Optuna Optimierung lokal starten
REM  Voraussetzung: setup_local_windows.bat wurde ausgefuehrt
REM ============================================================

cd /d "%~dp0"

REM Virtuelle Umgebung aktivieren
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo  [FEHLER] Virtuelle Umgebung nicht gefunden!
    echo  Bitte zuerst setup_local_windows.bat ausfuehren.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo  === trading_v2 Optuna Optimierung ===
echo.
echo  Multi-Asset, 30 Trials, Zeitreihe: 1d
echo  Logs werden in Echtzeit angezeigt.
echo  Abbrechen mit: Strg+C
echo.
echo  ============================================================
echo.

python main.py optimize --multi --timeframe 1d --trials 30

echo.
echo  ============================================================
echo   Optimierung abgeschlossen!
echo  ============================================================
echo.
echo  Ergebnisse in: logs\optuna_AAPL_1d.json
echo.
pause
