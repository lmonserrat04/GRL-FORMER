# Experimento: alineación contrastiva por diagnóstico entre sitios

Branch: `exp/dx-site-contrastive`

## Cambio respecto al paper

El pipeline mantiene las tres etapas (pretrain enmascarado → contrastive →
finetune solo CE). **Solo cambia la regla de emparejamiento de la etapa
contrastiva**:

| | Paper | Este experimento |
|---|---|---|
| Positivos | mismo sujeto en las dos ramas | mismo diagnóstico, **sitio distinto** |
| Negativos | resto del batch | diagnóstico opuesto, cualquier sitio (incluido el mismo) |
| Excluidos | — | mismo diagnóstico y mismo sitio (incluido el propio sujeto) |

El par del mismo sujeto deja de ser positivo: queda excluido por ser mismo
diagnóstico y mismo sitio.

## Batch sampler

`DiagnosisSiteBalancedSampler` (`data/loaders/samplers.py`):

- 16 ASD + 16 TD por batch.
- Al menos 4 sitios distintos por diagnóstico.
- Reintenta el batch si no se cumple la cobertura (con fallback para clases
  con menos de 4 sitios).
- Los sujetos de una celda (diagnóstico, sitio) pequeña pueden aparecer, pero
  un batch no repite sujetos mientras haya suficientes.

## Loss

Para cada ancla `i` de la rama temporal mirando a la de conectividad:

```
L_i = -1/|P(i)| Σ_{p∈P(i)} log exp(z_i·z_p/τ) / Σ_{a∈P(i)∪N(i)} exp(z_i·z_a/τ)
```

- `P(i)`: mismo diagnóstico, sitio distinto.
- `N(i)`: diagnóstico opuesto, cualquier sitio.
- Excluidos: mismo diagnóstico y mismo sitio (no entran en numerador ni denominador).
- Anclas sin positivos aportan 0; la media se hace sobre todos los anclajes.
- Implementación: `DiagnosisSiteInfoNCELoss` en `training/tasks/contrastive.py`.
- `τ = 0.07`, proyecciones MLP 2 capas → 128, batch 32, 50 épocas.

## Configuración

`config/config.yaml`:

```yaml
T_CONTRASTIVE:
  OBJECTIVE: diagnosis_site
  MIN_SITES_PER_CLASS: 4
  N_EPOCHS: 50
  BATCH_SIZE: 32
  TEMPERATURE: 0.07

AUX_TS_HEAD:
  ENABLED: false   # finetune solo CE, como el paper
```

## Ejecución (GPU remota)

Pipeline completo en un comando (desde la raíz del repo):

```bash
bash scripts/pipelines/run_dx_site_experiment.sh
```

Pasos individuales:

```bash
# 1. Contrastive (reutiliza pretrain)
python scripts/train/run_contrastive_only.py \
  --ckpt-dir experiments/20260928_160330_transformer_baseline/checkpoints \
  --seed 42 \
  --run-name dx_site_contrastive_seed42

# 2. Finetune LOSO
python scripts/train/run_finetune_only.py \
  --ckpt-dir experiments_contrastive_only/<RUN>/checkpoints \
  --pretrain-ckpt-dir experiments/20260928_160330_transformer_baseline/checkpoints \
  --contrastive-ckpt experiments_contrastive_only/<RUN>/checkpoints/contrastive_global.pt \
  --seed 42 \
  --run-name dx_site_contrastive_loso

# 3. Probes lineales (fused, h_ts, h_fc)
python scripts/analysis/linear_probe_h_ts.py \
  --run-dir experiments_finetune_only/<FT_RUN> \
  --eval-protocol loso \
  --representations fused h_ts_raw h_ts_proj h_fc_raw h_fc_proj

# 4. AUC + ECE por sitio
python scripts/analysis/fused_calibration_report.py \
  --run-dir experiments_finetune_only/<FT_RUN> \
  --baseline-auc 0.7030

# 5. Site-invariance sobre fused congelado
python scripts/analysis/site_invariance_probe.py \
  --run-dir experiments_finetune_only/<FT_RUN> \
  --representation fused
```

## Métricas de evaluación

- **AUC fused macro por sitio** (cada sitio pesa igual) contra el baseline
  `0.7030`.
- **ECE por sitio** (10 bins de igual ancho, confianza `max(p, 1-p)`), con
  media macro e IC bootstrap sobre sitios.
- **Probes lineales** entrenados con train+val de los 18 sitios y evaluados
  en el sitio held-out: `fused`, `h_ts_raw`, `h_ts_proj`, `h_fc_raw`,
  `h_fc_proj`.
- **Site-invariance**: clasificador logístico multinomial de sitio sobre
  `fused` congelado, validación cruzada estratificada de 5 folds; accuracy /
  balanced accuracy / macro-F1 contra chance `1/19 ≈ 0.053`.
