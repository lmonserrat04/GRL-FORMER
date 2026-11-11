#!/usr/bin/env bash
# Pipeline completo del experimento de alineación por diagnóstico entre sitios.
#
# Uso (dentro del contenedor `grl`, workdir /app):
#   bash scripts/pipelines/run_dx_site_experiment.sh
#
# Variables opcionales:
#   PRETRAIN_DIR   checkpoints de pretrain (default: baseline del paper)
#   SEED           semilla (default: 42)
#   BASELINE_AUC   AUC fused de referencia (default: 0.7030)
#   CONTRASTIVE_NAME / FINETUNE_NAME  nombres de los runs
set -euo pipefail

PRETRAIN_DIR="${PRETRAIN_DIR:-experiments/20260928_160330_transformer_baseline/checkpoints}"
SEED="${SEED:-42}"
BASELINE_AUC="${BASELINE_AUC:-0.7030}"
CONTRASTIVE_NAME="${CONTRASTIVE_NAME:-dx_site_contrastive_seed${SEED}}"
FINETUNE_NAME="${FINETUNE_NAME:-dx_site_contrastive_loso_seed${SEED}}"

echo "=============================================================="
echo " 1/5  Contrastive diagnosis_site"
echo "=============================================================="
python -u scripts/train/run_contrastive_only.py \
  --ckpt-dir "$PRETRAIN_DIR" \
  --seed "$SEED" \
  --run-name "$CONTRASTIVE_NAME"

C_RUN=$(ls -dt experiments_contrastive_only/*_"$CONTRASTIVE_NAME" | head -1)
echo "Contrastive run: $C_RUN"

echo "=============================================================="
echo " 2/5  Finetune LOSO (solo CE)"
echo "=============================================================="
python -u scripts/train/run_finetune_only.py \
  --ckpt-dir "$C_RUN/checkpoints" \
  --pretrain-ckpt-dir "$PRETRAIN_DIR" \
  --contrastive-ckpt "$C_RUN/checkpoints/contrastive_global.pt" \
  --seed "$SEED" \
  --run-name "$FINETUNE_NAME"

FT_RUN=$(ls -dt experiments_finetune_only/*_"$FINETUNE_NAME" | head -1)
echo "Finetune run: $FT_RUN"

echo "=============================================================="
echo " 3/5  Linear probes (fused, h_ts, h_fc)"
echo "=============================================================="
python -u scripts/analysis/linear_probe_h_ts.py \
  --run-dir "$FT_RUN" \
  --eval-protocol loso \
  --representations fused h_ts_raw h_ts_proj h_fc_raw h_fc_proj

echo "=============================================================="
echo " 4/5  AUC + ECE por sitio"
echo "=============================================================="
python -u scripts/analysis/fused_calibration_report.py \
  --run-dir "$FT_RUN" \
  --baseline-auc "$BASELINE_AUC"

echo "=============================================================="
echo " 5/5  Site-invariance sobre fused"
echo "=============================================================="
python -u scripts/analysis/site_invariance_probe.py \
  --run-dir "$FT_RUN" \
  --representation fused

echo
echo "=============================================================="
echo " COMPLETADO"
echo "=============================================================="
echo " contrastive : $C_RUN"
echo " finetune    : $FT_RUN"
echo " probes      : $FT_RUN/linear_probe_h_ts/"
echo " calibración : $FT_RUN/calibration/"
echo " site-invar. : $FT_RUN/site_invariance/"
