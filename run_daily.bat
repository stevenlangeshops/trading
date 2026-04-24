@echo off
cd /d C:\steven\trading_v5\trading
echo ==============================
echo  Trading Bot - Taeglicher Lauf
echo  %date% %time%
echo ==============================

if not exist logs mkdir logs

python update_ic.py       >> logs\daily_run.log 2>&1
python live_inference.py  >> logs\daily_run.log 2>&1

echo Lauf beendet: %date% %time% >> logs\daily_run.log
echo. >> logs\daily_run.log
