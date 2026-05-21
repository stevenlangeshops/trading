#!/usr/bin/env bash
# ==============================================================================
# run_daily_scan.sh
# Taeglich per Cron um 09:00 Uhr (Berlin) ausgefuehrt.
#
# Ablauf:
#   1. .env laden (Telegram-Credentials)
#   2. Virtual Environment aktivieren
#   3. daily_scan_report.py ausfuehren (Daten-Update + Scan + Telegram)
#   4. Log schreiben
#
# Manuell testen:
#   bash /opt/trading/deploy/run_daily_scan.sh
# ==============================================================================

set -euo pipefail

TRADING_DIR="/opt/trading"
VENV_DIR="${TRADING_DIR}/.venv"
LOG_DIR="${TRADING_DIR}/logs"
LOG_FILE="${LOG_DIR}/scan_$(date +%Y%m%d).log"

mkdir -p "${LOG_DIR}"

# .env laden
if [[ -f "${TRADING_DIR}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${TRADING_DIR}/.env"
    set +a
fi

{
    echo "======================================================"
    echo "  START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "======================================================"

    cd "${TRADING_DIR}"
    "${VENV_DIR}/bin/python" daily_scan_report.py
    EXIT_CODE=$?

    echo ""
    echo "  END: $(date '+%Y-%m-%d %H:%M:%S %Z')  (exit=${EXIT_CODE})"
    echo "======================================================"
} >> "${LOG_FILE}" 2>&1

exit ${EXIT_CODE:-0}
