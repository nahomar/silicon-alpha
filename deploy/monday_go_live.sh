#!/usr/bin/env bash
# deploy/monday_go_live.sh — Monday 4/20/2026 production launcher.
#
# Assumes you are SSH'd into the head node of a 24x H100 SXM cluster with
# Mellanox ConnectX-7 NICs. Runs the whole ritual:
#   1. Sanity checks (GPU count, NIC, driver versions)
#   2. Sources NCCL env
#   3. Builds the Hopper-native kernels in-place
#   4. Syncs broker margin table
#   5. Refits HybridBinTokenizer on the pre-market tape
#   6. Runs the preflight checker (fails fast if any go/no-go gate fails)
#   7. Launches the live paper runner inside a headless screen session
#
# Usage:
#   export DATABENTO_API_KEY=...
#   export FMP_API_KEY=...
#   export BROKER_MARGIN_URL=https://broker.example.com/intraday_margin.yml
#   ./deploy/monday_go_live.sh

set -euo pipefail

ROOT="${ROOT:-/root/silicon-alpha}"
LOG_DIR="$ROOT/reports/monday_live_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[go-live] root=$ROOT   log=$LOG_DIR"

# -----------------------------------------------------------------------
# 1. Preflight sanity
# -----------------------------------------------------------------------
echo "[1/7] Host sanity"
if ! command -v nvidia-smi >/dev/null; then
    echo "  FATAL: nvidia-smi not on PATH — are we on the H100 box?" >&2
    exit 2
fi
N_GPU=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1 || echo 0)
echo "  GPUs detected: $N_GPU"
[[ "${N_GPU:-0}" -ge 8 ]] || { echo "  FATAL: expected ≥8 GPUs, got $N_GPU" >&2; exit 2; }
if ! ip link show ens1 >/dev/null 2>&1; then
    echo "  WARN: ens1 interface missing; NCCL will fall back. Edit nccl_env.sh." >&2
fi
if ! lsmod | grep -q nv_peer_memory; then
    echo "  WARN: nv_peer_memory kernel module not loaded — GPUDirect RDMA disabled" >&2
fi

# -----------------------------------------------------------------------
# 2. NCCL env
# -----------------------------------------------------------------------
echo "[2/7] Sourcing NCCL env"
# shellcheck disable=SC1091
source "$ROOT/deploy/nccl_env.sh"

# -----------------------------------------------------------------------
# 3. Build Hopper-native kernels (Phase 3)
# -----------------------------------------------------------------------
echo "[3/7] Building odte_kernels_cu (H100 SM 90a)"
cd "$ROOT/odte/kernels"
if [[ "${ODTE_BUILD_RDMA:-0}" == "1" ]]; then
    echo "  building WITH GPUDirect RDMA (requires rdma-core headers)"
fi
python setup.py build_ext --inplace 2>&1 | tee "$LOG_DIR/kernel_build.log"
python -c "import odte_kernels_cu; print('[kernels] ok:', odte_kernels_cu)" \
    || { echo "FATAL: kernel import failed" >&2; exit 3; }
cd "$ROOT"

# -----------------------------------------------------------------------
# 4. Sync broker margin table
# -----------------------------------------------------------------------
echo "[4/7] Syncing broker margin table"
BROKER_YML="$ROOT/configs/broker_margin_live.yml"
if [[ -n "${BROKER_MARGIN_URL:-}" ]]; then
    curl -fsSL "$BROKER_MARGIN_URL" -o "$BROKER_YML" \
        && echo "  fetched $BROKER_MARGIN_URL → $BROKER_YML"
else
    # fall back to the Monday-crisis template
    if [[ ! -f "$BROKER_YML" ]]; then
        cp "$ROOT/configs/broker_margin_monday_crisis.yml" "$BROKER_YML"
        echo "  no BROKER_MARGIN_URL set; using Monday-crisis template"
    fi
fi
# Start a background watcher that refreshes the table every 5 minutes
if [[ -n "${BROKER_MARGIN_URL:-}" ]]; then
    (while true; do
        sleep 300
        curl -fsSL "$BROKER_MARGIN_URL" -o "$BROKER_YML.new" \
            && mv "$BROKER_YML.new" "$BROKER_YML"
    done) &
    echo "  intraday refresh loop PID=$! (every 5 min)"
fi

# -----------------------------------------------------------------------
# 5. Monday-morning quantile refit on pre-market tape
# -----------------------------------------------------------------------
echo "[5/7] Refitting HybridBinTokenizer on Monday pre-market"
python "$ROOT/deploy/monday_quantile_refit.py" \
    --lookback-min 30 \
    --underlying SPX \
    --out "$ROOT/checkpoints/hybrid_tokenizer_monday.json" \
    2>&1 | tee "$LOG_DIR/tokenizer_refit.log"

# -----------------------------------------------------------------------
# 6. Preflight go/no-go
# -----------------------------------------------------------------------
echo "[6/7] Preflight go/no-go"
if ! python "$ROOT/deploy/preflight.py" --full 2>&1 | tee "$LOG_DIR/preflight.log"; then
    echo "PREFLIGHT FAIL — aborting launch" >&2
    exit 4
fi

# -----------------------------------------------------------------------
# 7. Launch headless screen session
# -----------------------------------------------------------------------
echo "[7/7] Launching live paper runner in screen session 'odte_live'"
SESSION="odte_live_$(date -u +%Y%m%d)"
if screen -list | grep -q "$SESSION"; then
    echo "  session $SESSION already running — attach with 'screen -r $SESSION'"
    exit 0
fi

screen -dmS "$SESSION" bash -c "
    cd $ROOT
    source .venv/bin/activate
    source deploy/nccl_env.sh
    python odte_live_paper.py \
        --feed databento_opra \
        --underlying SPX \
        --device cuda \
        --dml-ckpt checkpoints/dml_pricer.pt \
        --tradefm-ckpt checkpoints/tradefm_524m.pt \
        --tokenizer checkpoints/hybrid_tokenizer_monday.json \
        --margin-table configs/broker_margin_live.yml \
        --equity 500000 \
        --gross-budget 0.3 \
        --max-ticks 5000000 \
        --log-every 5000 \
        --post-trade 2>&1 | tee $LOG_DIR/live.log
"
echo "  screen session: $SESSION  (attach: screen -r $SESSION)"
echo "  log tail:       tail -f $LOG_DIR/live.log"
echo
echo "DONE. Good luck."
