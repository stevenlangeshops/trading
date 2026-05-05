#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# upload_to_hetzner.sh
# Lokales Script: überträgt das Trading-Projekt per rsync auf den Hetzner-Server.
#
# LOKAL ausführen (Windows: Git-Bash oder WSL):
#   chmod +x deploy/upload_to_hetzner.sh
#   SERVER_IP=1.2.3.4 ./deploy/upload_to_hetzner.sh
#
# Oder mit allen Optionen:
#   SERVER_IP=1.2.3.4 SERVER_USER=root SSH_KEY=~/.ssh/hetzner_trading \
#     ./deploy/upload_to_hetzner.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Konfiguration (via Umgebungsvariablen überschreibbar) ─────────────────────
SERVER_IP="${SERVER_IP:-}"
SERVER_USER="${SERVER_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/hetzner_trading}"
REMOTE_DIR="/opt/trading"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"    # Repo-Root (ein Level über deploy/)

# Farben
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Eingabe-Prüfung ───────────────────────────────────────────────────────────
if [[ -z "${SERVER_IP}" ]]; then
    echo -n "Hetzner Server-IP eingeben: "
    read -r SERVER_IP
fi
[[ -z "${SERVER_IP}" ]] && error "SERVER_IP darf nicht leer sein."

if [[ ! -f "${SSH_KEY}" ]]; then
    warn "SSH-Key nicht gefunden: ${SSH_KEY}"
    warn "Verwende Standard-SSH-Schlüssel (id_rsa / id_ed25519)."
    SSH_OPTS="-o StrictHostKeyChecking=accept-new"
else
    SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"
fi

SSH_TARGET="${SERVER_USER}@${SERVER_IP}"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Upload: ${LOCAL_DIR}"
echo "  Ziel:   ${SSH_TARGET}:${REMOTE_DIR}"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Remote-Verzeichnis anlegen ────────────────────────────────────────────────
info "Erstelle Remote-Verzeichnis ${REMOTE_DIR} ..."
ssh ${SSH_OPTS} "${SSH_TARGET}" "mkdir -p ${REMOTE_DIR}"

# ── rsync: Projektdateien übertragen ─────────────────────────────────────────
# Ausgeschlossen:
#   - .git/               (Git-Verlauf)
#   - data/raw/           (große Parquet-Daten – separat verwalten)
#   - checkpoints/*.pt    (Modelle – separat übertragen)
#   - __pycache__/        (Byte-Code)
#   - kaggle_kernel_runs/ (Kaggle-Logs, nicht benötigt)
#   - .pytest_cache/      (Test-Cache)
#   - *.egg-info/         (Python-Build-Artefakte)
#   - node_modules/       (falls vorhanden)
#   - .env                (Credentials – niemals per rsync übertragen!)

info "Übertrage Python-Code und Konfiguration ..."
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='.env' \
    --exclude='data/raw/' \
    --exclude='checkpoints/*.pt' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='kaggle_kernel_runs/' \
    --exclude='.kaggle_kernel_upload/' \
    --exclude='.pytest_cache/' \
    --exclude='*.egg-info/' \
    --exclude='node_modules/' \
    --exclude='extracting/' \
    --exclude='transferring/' \
    --exclude='reading/' \
    --exclude='resolve/' \
    --exclude='sensitivity_results.csv' \
    --exclude='archive*.zip' \
    -e "ssh ${SSH_OPTS}" \
    "${LOCAL_DIR}/" \
    "${SSH_TARGET}:${REMOTE_DIR}/"

echo ""
info "Code-Transfer abgeschlossen."

# ── Ensemble-Modelle übertragen ───────────────────────────────────────────────
PROD_CKPT_LOCAL="${LOCAL_DIR}/checkpoints/production"
PROD_CKPT_REMOTE="${REMOTE_DIR}/checkpoints/production"

if ls "${PROD_CKPT_LOCAL}"/prod_model_seed*.pt &>/dev/null 2>&1; then
    MODEL_COUNT=$(ls "${PROD_CKPT_LOCAL}"/prod_model_seed*.pt | wc -l)
    info "Übertrage ${MODEL_COUNT} Ensemble-Modelle ..."
    ssh ${SSH_OPTS} "${SSH_TARGET}" "mkdir -p ${PROD_CKPT_REMOTE}"
    rsync -avz --progress \
        -e "ssh ${SSH_OPTS}" \
        "${PROD_CKPT_LOCAL}"/prod_model_seed*.pt \
        "${SSH_TARGET}:${PROD_CKPT_REMOTE}/"
    info "Ensemble-Modelle übertragen."
else
    warn "Keine prod_model_seed*.pt Dateien gefunden unter ${PROD_CKPT_LOCAL}"
    warn "Ensemble-Modelle müssen manuell übertragen werden."
fi

# ── Walk-Forward Checkpoints übertragen (Fallback) ───────────────────────────
WF_LOCAL="${LOCAL_DIR}/checkpoints/v2_7d"
WF_REMOTE="${REMOTE_DIR}/checkpoints/v2_7d"

if [[ -d "${WF_LOCAL}" ]]; then
    info "Übertrage Walk-Forward Checkpoints (Fallback-Modelle) ..."
    ssh ${SSH_OPTS} "${SSH_TARGET}" "mkdir -p ${WF_REMOTE}"
    rsync -avz --progress \
        -e "ssh ${SSH_OPTS}" \
        "${WF_LOCAL}/" \
        "${SSH_TARGET}:${WF_REMOTE}/"
    info "Walk-Forward Checkpoints übertragen."
fi

# ── Berechtigungen setzen ─────────────────────────────────────────────────────
info "Setze Berechtigungen ..."
ssh ${SSH_OPTS} "${SSH_TARGET}" "chmod +x ${REMOTE_DIR}/deploy/setup_hetzner.sh ${REMOTE_DIR}/deploy/upload_to_hetzner.sh 2>/dev/null || true"

# ── Fertig ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Upload abgeschlossen!"
echo ""
echo "  Nächste Schritte:"
echo "  1. SSH:    ssh ${SSH_OPTS} ${SSH_TARGET}"
echo "  2. Setup:  cd ${REMOTE_DIR} && ./deploy/setup_hetzner.sh"
echo "══════════════════════════════════════════════════════"
echo ""
