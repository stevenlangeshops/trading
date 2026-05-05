#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup_hetzner.sh
# Server-Setup für das Trading-System auf einem Hetzner-VPS.
#
# Voraussetzungen:
#   - Ubuntu 22.04 oder 24.04 LTS (frische Instanz)
#   - SSH-Zugang als root
#   - Das Projektverzeichnis wurde bereits per upload_to_hetzner.sh übertragen
#
# Ausführung (auf dem Server als root):
#   chmod +x setup_hetzner.sh
#   ./setup_hetzner.sh
#
# Was dieses Skript macht:
#   1. System-Pakete aktualisieren
#   2. Python 3.11 installieren (via deadsnakes PPA)
#   3. Projektverzeichnis unter /opt/trading anlegen
#   4. Python Virtual Environment erstellen
#   5. PyTorch (CPU) + alle Abhängigkeiten installieren
#   6. .env-Template erstellen
#   7. Cron-Job einrichten (täglich 16:10 Uhr CET → 30 Min nach Marktöffnung)
#   8. Log-Rotation konfigurieren
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Konfiguration ─────────────────────────────────────────────────────────────
TRADING_DIR="/opt/trading"
VENV_DIR="${TRADING_DIR}/.venv"
LOG_DIR="${TRADING_DIR}/logs"
PYTHON_VERSION="3.12"
CRON_TIME="45 15 * * 1-5"   # 15:45 CEST (Serverzeit Europe/Berlin) = 13:45 UTC = 09:45 ET
                               # Mo–Fr, 15 Minuten nach Marktöffnung (09:30 ET)
                               # Bewusst NICHT direkt bei Marktöffnung: Opening-Volatilität
                               # und breite Spreads in den ersten Minuten vermeiden.

# Farben für Output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}══════════════════════════════════════════${NC}"; \
            echo -e "${GREEN}  $*${NC}"; \
            echo -e "${GREEN}══════════════════════════════════════════${NC}"; }

# ── Root-Check ────────────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && error "Bitte als root ausführen: sudo ./setup_hetzner.sh"

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1: System-Pakete
# ══════════════════════════════════════════════════════════════════════════════
section "1/7  System-Pakete"
apt-get update -qq
apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    git \
    rsync \
    cron \
    logrotate \
    tzdata \
    libhdf5-dev \
    libatlas-base-dev \
    gfortran
info "System-Pakete installiert."

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2: Python 3.11
# ══════════════════════════════════════════════════════════════════════════════
section "2/7  Python ${PYTHON_VERSION}"

# Prüfe ob Python 3.12 bereits verfügbar ist (auf Ubuntu 24.04 ist es das)
if ! command -v python${PYTHON_VERSION} &>/dev/null; then
    info "Python ${PYTHON_VERSION} nicht gefunden – installiere..."
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
else
    info "Python ${PYTHON_VERSION} bereits vorhanden: $(python${PYTHON_VERSION} --version)"
fi

# python3-venv sicherstellen
apt-get install -y python${PYTHON_VERSION}-venv --quiet

# pip sicherstellen
if ! python${PYTHON_VERSION} -m pip --version &>/dev/null; then
    apt-get install -y python3-pip --quiet
fi

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3: Verzeichnisstruktur
# ══════════════════════════════════════════════════════════════════════════════
section "3/7  Verzeichnisstruktur"

mkdir -p "${TRADING_DIR}"/{logs,checkpoints/production,data/raw,features}
info "Verzeichnisse erstellt unter ${TRADING_DIR}"

# Prüfe ob der Code bereits hochgeladen wurde
if [[ ! -f "${TRADING_DIR}/live_inference.py" ]]; then
    warn "live_inference.py nicht gefunden in ${TRADING_DIR}!"
    warn "Bitte zuerst upload_to_hetzner.sh lokal ausführen, dann dieses Skript."
    warn "Fortfahre trotzdem mit der Umgebungs-Installation..."
fi

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4: Virtual Environment + Pakete
# ══════════════════════════════════════════════════════════════════════════════
section "4/7  Python Virtual Environment"

if [[ ! -d "${VENV_DIR}" ]]; then
    python${PYTHON_VERSION} -m venv "${VENV_DIR}"
    info "Neues venv erstellt: ${VENV_DIR}"
else
    info "Vorhandenes venv wird verwendet: ${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

"${PIP}" install --upgrade pip --quiet

# ── PyTorch CPU (muss zuerst, da großes Paket) ────────────────────────────────
info "Installiere PyTorch 2.10.0 (CPU-only)..."
"${PIP}" install \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --quiet

# ── Alle weiteren Abhängigkeiten ──────────────────────────────────────────────
info "Installiere Projekt-Abhängigkeiten..."
"${PIP}" install -r "${TRADING_DIR}/deploy/requirements_server.txt" --quiet

info "Paket-Installation abgeschlossen."
"${PYTHON}" -c "import torch, pandas, numpy, alpaca; print('  torch:', torch.__version__, '| pandas:', pandas.__version__, '| numpy:', numpy.__version__)"

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5: .env-Template
# ══════════════════════════════════════════════════════════════════════════════
section "5/7  Konfiguration (.env)"

ENV_FILE="${TRADING_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    cat > "${ENV_FILE}" << 'EOF'
# ══════════════════════════════════════════════════════════════════════════════
# Trading-Bot Konfiguration
# Ausfüllen und dann: chmod 600 /opt/trading/.env
# ══════════════════════════════════════════════════════════════════════════════

# ── Alpaca Paper/Live Trading ─────────────────────────────────────────────────
APCA_API_KEY_ID=DEIN_KEY_ID_HIER
APCA_API_SECRET_KEY=DEIN_SECRET_KEY_HIER
# APCA_PAPER=true          # true = Paper-Trading (Standard), false = Live

# ── Telegram-Benachrichtigungen (optional) ────────────────────────────────────
# TELEGRAM_TOKEN=
# TELEGRAM_CHAT_ID=
EOF
    chmod 600 "${ENV_FILE}"
    warn ".env-Template erstellt: ${ENV_FILE}"
    warn ">>> JETZT AUSFÜLLEN: nano ${ENV_FILE} <<<"
else
    info ".env bereits vorhanden – nicht überschrieben."
fi

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6: Run-Skript
# ══════════════════════════════════════════════════════════════════════════════
section "6/7  Run-Skript"

RUN_SCRIPT="${TRADING_DIR}/run_inference.sh"
cat > "${RUN_SCRIPT}" << SCRIPT
#!/usr/bin/env bash
# Täglich ausgeführt vom Cron-Job.
# Lädt .env, aktiviert venv, führt live_inference.py aus.

set -euo pipefail

TRADING_DIR="/opt/trading"
VENV_DIR="\${TRADING_DIR}/.venv"
LOG_DIR="\${TRADING_DIR}/logs"
LOG_FILE="\${LOG_DIR}/inference_\$(date +%Y%m%d).log"

# .env laden
if [[ -f "\${TRADING_DIR}/.env" ]]; then
    set -a
    source "\${TRADING_DIR}/.env"
    set +a
fi

echo "======================================================" >> "\${LOG_FILE}"
echo "  START: \$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "\${LOG_FILE}"
echo "======================================================" >> "\${LOG_FILE}"

cd "\${TRADING_DIR}"
"\${VENV_DIR}/bin/python" live_inference.py --execute >> "\${LOG_FILE}" 2>&1
EXIT_CODE=\$?

echo "" >> "\${LOG_FILE}"
echo "  END:  \$(date '+%Y-%m-%d %H:%M:%S %Z')  (exit=\${EXIT_CODE})" >> "\${LOG_FILE}"
echo "======================================================" >> "\${LOG_FILE}"

exit \${EXIT_CODE}
SCRIPT

chmod +x "${RUN_SCRIPT}"
info "Run-Skript erstellt: ${RUN_SCRIPT}"

# ── Dry-Run-Skript (zum Testen ohne echte Orders) ─────────────────────────────
DRYRUN_SCRIPT="${TRADING_DIR}/run_dryrun.sh"
cat > "${DRYRUN_SCRIPT}" << SCRIPT
#!/usr/bin/env bash
# Testlauf: Simulation ohne echte Orders (--dry-run)
set -euo pipefail
TRADING_DIR="/opt/trading"
if [[ -f "\${TRADING_DIR}/.env" ]]; then set -a; source "\${TRADING_DIR}/.env"; set +a; fi
cd "\${TRADING_DIR}"
"\${TRADING_DIR}/.venv/bin/python" live_inference.py --dry-run
SCRIPT
chmod +x "${DRYRUN_SCRIPT}"
info "Dry-Run-Skript: ${DRYRUN_SCRIPT}"

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 7: Cron-Job + Log-Rotation
# ══════════════════════════════════════════════════════════════════════════════
section "7/7  Cron-Job & Log-Rotation"

# Zeitzone sicherstellen
timedatectl set-timezone Europe/Berlin 2>/dev/null || true

# Cron-Job einrichten (Mo–Fr, 16:10 Uhr CET)
CRON_ENTRY="${CRON_TIME} root TZ=Europe/Berlin ${RUN_SCRIPT} >> ${LOG_DIR}/cron.log 2>&1"
CRON_FILE="/etc/cron.d/trading-inference"

cat > "${CRON_FILE}" << EOF
# Trading-Inference: täglich Mo–Fr um 16:10 Uhr CET (30 Min nach Marktöffnung)
# Format: Minute Stunde Tag Monat Wochentag Benutzer Befehl
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
${CRON_ENTRY}
EOF
chmod 644 "${CRON_FILE}"
service cron restart 2>/dev/null || systemctl restart cron 2>/dev/null || true
info "Cron-Job aktiv: ${CRON_TIME} Berliner Zeit (Mo-Fr, = 13:45 UTC = 09:45 ET)"

# Log-Rotation
cat > "/etc/logrotate.d/trading" << EOF
${LOG_DIR}/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    dateext
    dateformat -%Y%m%d
}
EOF
info "Log-Rotation konfiguriert (30 Tage, täglich komprimiert)."

# ══════════════════════════════════════════════════════════════════════════════
# FERTIG
# ══════════════════════════════════════════════════════════════════════════════
section "Setup abgeschlossen!"
echo ""
echo "  Projektpfad:  ${TRADING_DIR}"
echo "  Virtual Env:  ${VENV_DIR}"
echo "  Logs:         ${LOG_DIR}/"
echo "  Cron-Job:     ${CRON_FILE}"
echo ""
echo "  Nächste Schritte:"
echo "  1. .env ausfüllen:       nano ${ENV_FILE}"
echo "  2. Testlauf:             ${DRYRUN_SCRIPT}"
echo "  3. Echter Paper-Trade:   ${RUN_SCRIPT}"
echo ""
