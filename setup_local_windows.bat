@echo off
REM ============================================================
REM  trading_v2 - Lokales Setup fuer Windows (Python 3.13 fix)
REM  Ausfuehren mit: Doppelklick
REM ============================================================

echo.
echo  === trading_v2 Lokales Windows Setup ===
echo.

REM Pruefen ob Python vorhanden
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  [FEHLER] Python nicht gefunden!
    echo  Bitte installiere Python 3.11 von: https://www.python.org/downloads/
    echo  Wichtig: Haken bei "Add Python to PATH" setzen!
    pause
    exit /b 1
)

echo  [OK] Python gefunden:
python --version

REM Python-Version pruefen
for /f "tokens=2 delims= " %%v in ('python --version') do set PYVER=%%v
echo  [INFO] Version: %PYVER%

REM In Projektverzeichnis wechseln
cd /d "%~dp0"
echo  [OK] Verzeichnis: %CD%

REM Virtuelle Umgebung erstellen (falls noch nicht vorhanden)
IF NOT EXIST "venv\" (
    echo.
    echo  Erstelle virtuelle Umgebung...
    python -m venv venv
    echo  [OK] venv erstellt
) ELSE (
    echo  [OK] venv bereits vorhanden
)

REM Aktivieren
call venv\Scripts\activate.bat

REM pip aktualisieren
echo.
echo  Aktualisiere pip...
python -m pip install --upgrade pip --quiet

REM requirements_local.txt bevorzugen (Python 3.13 kompatibel)
IF EXIST "requirements_local.txt" (
    echo.
    echo  Installiere Pakete aus requirements_local.txt
    echo  ^(Python 3.13 kompatible Versionen^)
    echo  ^(kann 5-10 Minuten dauern^)...
    echo.
    pip install -r requirements_local.txt
) ELSE (
    echo.
    echo  Installiere Pakete aus requirements.txt
    echo  ^(kann 5-10 Minuten dauern^)...
    echo.
    pip install -r requirements.txt
)

IF ERRORLEVEL 1 (
    echo.
    echo  [FEHLER] Installation fehlgeschlagen!
    echo.
    echo  Moegliche Ursache: Pakete noch nicht fuer Python 3.13 verfuegbar.
    echo  Empfehlung: Installiere Python 3.11 zusaetzlich von:
    echo    https://www.python.org/downloads/release/python-3119/
    echo.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Setup abgeschlossen!
echo  ============================================================
echo.
echo  Naechster Schritt: run_optimize.bat starten
echo.
pause
