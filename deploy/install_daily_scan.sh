#!/usr/bin/env bash
# ==============================================================================
# install_daily_scan.sh
# Richtet den taeglichen VCP-Scanner-Cron-Job auf dem Hetzner-Server ein.
#
# Ausfuehren auf dem Server (als root):
#   chmod +x /opt/trading/deploy/install_daily_scan.sh
#   /opt/trading/deploy/install_daily_scan.sh
#
# Was dieses Skript tut:
#   1. Prueft ob .env mit Telegram-Credentials existiert
#   2. Installiert python-dotenv (falls fehlend)
#   3. Macht run_daily_scan.sh ausfuehrbar
#   4. Setzt Cron-Job: Mo-Fr 09:00 Uhr Berliner Zeit
#   5. Fuehrt einmaligen Test-Lauf durch (--dry-run)
# ==============================================================================

set -euo pipefail

TRADING_DIR="/opt/trading"
VENV_DIR="${TRADING_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
RUN_SCRIPT="${TRADING_DIR}/deploy/run_daily_scan.sh"
LOG_DIR="${TRADING_DIR}/logs"
CRON_FILE="/etc/cron.d/trading-daily-scan"

# Farben
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "  install_daily_scan.sh  |  VCP Daily Scan Setup"
echo "============================================================"

# Root-Check
[[ $EUID -ne 0 ]] && error "Bitte als root ausfuehren: sudo bash install_daily_scan.sh"

# Verzeichnis pruefen
[[ -d "${TRADING_DIR}" ]] || error "Projektverzeichnis nicht gefunden: ${TRADING_DIR}"
[[ -d "${VENV_DIR}"    ]] || error "Virtual Environment nicht gefunden: ${VENV_DIR}. Erst setup_hetzner.sh ausfuehren."

# ── Schritt 1: .env pruefen ───────────────────────────────────────────────────
echo ""
echo "[ 1/5 ]  .env pruefen ..."
ENV_FILE="${TRADING_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    error ".env nicht gefunden: ${ENV_FILE}\nBitte erstellen:\n  cp ${TRADING_DIR}/.env.example ${ENV_FILE}\n  nano ${ENV_FILE}"
fi

TELEGRAM_TOKEN=$(grep -E "^TELEGRAM_TOKEN=" "${ENV_FILE}" | cut -d= -f2- | tr -d '"' | tr -d "'" | xargs 2>/dev/null || echo "")
TELEGRAM_CHAT_ID=$(grep -E "^TELEGRAM_CHAT_ID=" "${ENV_FILE}" | cut -d= -f2- | tr -d '"' | tr -d "'" | xargs 2>/dev/null || echo "")

if [[ -z "${TELEGRAM_TOKEN}" || "${TELEGRAM_TOKEN}" == "DEIN_TOKEN_HIER" ]]; then
    warn "TELEGRAM_TOKEN ist leer oder nicht gesetzt!"
    warn "Bitte in ${ENV_FILE} eintragen:"
    warn "  TELEGRAM_TOKEN=123456789:ABCDEF..."
    warn "Fahre trotzdem fort – Nachrichten werden nicht gesendet bis Token gesetzt ist."
else
    info "TELEGRAM_TOKEN gefunden (${TELEGRAM_TOKEN:0:15}...)"
fi

if [[ -z "${TELEGRAM_CHAT_ID}" || "${TELEGRAM_CHAT_ID}" == "DEINE_CHAT_ID" ]]; then
    warn "TELEGRAM_CHAT_ID ist leer – bitte in .env eintragen."
else
    info "TELEGRAM_CHAT_ID gefunden: ${TELEGRAM_CHAT_ID}"
fi

# ── Schritt 2: python-dotenv installieren ────────────────────────────────────
echo ""
echo "[ 2/5 ]  Python-Abhaengigkeiten pruefen ..."
"${PIP}" install python-dotenv requests --quiet
info "python-dotenv + requests OK"

# ── Schritt 3: Run-Skript ausfuehrbar machen ──────────────────────────────────
echo ""
echo "[ 3/5 ]  run_daily_scan.sh vorbereiten ..."
[[ -f "${RUN_SCRIPT}" ]] || error "run_daily_scan.sh nicht gefunden: ${RUN_SCRIPT}"
chmod +x "${RUN_SCRIPT}"
mkdir -p "${LOG_DIR}"
info "Skript ausfuehrbar: ${RUN_SCRIPT}"

# ── Schritt 4: Zeitzone sicherstellen ─────────────────────────────────────────
echo ""
echo "[ 4/5 ]  Cron-Job einrichten (09:00 Uhr Berlin, Mo-Fr) ..."
timedatectl set-timezone Europe/Berlin 2>/dev/null || true
CURRENT_TZ=$(timedatectl show --property=Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || echo "unbekannt")
info "Zeitzone: ${CURRENT_TZ}"

# Cron-Job schreiben
# 09:00 Berliner Zeit (CET/CEST), Mo-Fr
# Cron laeuft in UTC: 07:00 UTC (Winter/CET) oder 07:00 UTC (Sommer/CEST)
# Wir setzen TZ=Europe/Berlin direkt im Cron-Eintrag – das ist der sicherste Weg
cat > "${CRON_FILE}" << EOF
# VCP Daily Scan: taeglich Mo-Fr um 09:00 Uhr Berliner Zeit
# Sendet Telegram-Nachricht mit heutigen VCP-Kaufkandidaten
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
TZ=Europe/Berlin

0 9 * * 1-5 root ${RUN_SCRIPT} >> ${LOG_DIR}/cron_scan.log 2>&1
EOF

chmod 644 "${CRON_FILE}"
service cron restart 2>/dev/null || systemctl restart cron 2>/dev/null || true
info "Cron-Job installiert: ${CRON_FILE}"
info "Naechste Ausfuehrung: naechster Werktag um 09:00 Uhr Berlin"

# ── Schritt 5: Test-Lauf ──────────────────────────────────────────────────────
echo ""
echo "[ 5/5 ]  Test-Lauf (--dry-run, kein Telegram) ..."
echo ""
cd "${TRADING_DIR}"
"${PYTHON}" daily_scan_report.py --dry-run --no-update 2>&1 | head -60
echo ""

# ── Zusammenfassung ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
info "Setup abgeschlossen!"
echo ""
echo "  Cron-Job:   ${CRON_FILE}"
echo "  Laeuf:      Mo-Fr um 09:00 Uhr Berliner Zeit"
echo "  Logs:       ${LOG_DIR}/scan_YYYYMMDD.log"
echo ""
echo "  Manueller Test (mit Telegram):"
echo "    bash ${RUN_SCRIPT}"
echo ""
echo "  Test ohne Telegram:"
echo "    cd ${TRADING_DIR} && ${PYTHON} daily_scan_report.py --dry-run"
echo ""
echo "  Logs pruefen:"
echo "    tail -f ${LOG_DIR}/scan_\$(date +%Y%m%d).log"
echo ""
echo "  Cron-Job entfernen:"
echo "    rm ${CRON_FILE} && service cron restart"
echo "============================================================"
