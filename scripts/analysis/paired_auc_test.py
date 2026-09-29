"""
Test pareado por sitio entre dos configuraciones LOSO.

Uso:
    python paired_auc_test.py <dir_no_grl> <dir_grl>

Ejemplo:
    python paired_auc_test.py \
        experiments_sweep_dropout_loso/loso_no_grl_dp0.3/checkpoints \
        experiments_sweep_minimal/loso_grl_g3_w1/checkpoints
"""
# --- sys.path bootstrap (scripts movidos a subcarpetas) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_auc_per_fold(ckpt_dir: Path) -> dict:
    """Devuelve {fold_idx: (auc, n, sens_at_05, spec_at_05)}."""
    npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
    results = {}
    for f in npz_files:
        fold_idx = int(f.stem.split("_")[-1])
        d = np.load(f, allow_pickle=True)
        labels = d["labels"]
        probs = d["probs"]

        if len(np.unique(labels)) < 2:
            continue
        auc = roc_auc_score(labels, probs)

        # Sens/Spec a 0.5
        preds_05 = (probs >= 0.5).astype(int)
        tp = ((preds_05 == 1) & (labels == 1)).sum()
        fn = ((preds_05 == 0) & (labels == 1)).sum()
        tn = ((preds_05 == 0) & (labels == 0)).sum()
        fp = ((preds_05 == 1) & (labels == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[fold_idx] = {"auc": auc, "n": int(len(labels)),
                              "sens": float(sens), "spec": float(spec)}
    return results


def main(dir_a, dir_b, label_a, label_b):
    a = load_auc_per_fold(Path(dir_a))
    b = load_auc_per_fold(Path(dir_b))

    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        print("[ERROR] no hay folds comunes")
        sys.exit(1)

    print(f"\nFolds comunes: {len(common)}")
    print(f"  {label_a}: {len(a)} folds")
    print(f"  {label_b}: {len(b)} folds")

    # Filtrar por n mínimo (opcional)
    MIN_N = 30
    valid = [f for f in common if a[f]["n"] >= MIN_N and b[f]["n"] >= MIN_N]
    print(f"  Válidos con n>={MIN_N}: {len(valid)}")

    # ─── Tabla detallada ─────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"{'fold':<8s}  {'n':<6s}  "
          f"{label_a+' AUC':<14s}  {label_b+' AUC':<14s}  "
          f"{'ΔAUC':<10s}  {'Sens_a':<10s}  {'Sens_b':<10s}")
    print("-"*100)
    for f in valid:
        da, db = a[f], b[f]
        delta = db["auc"] - da["auc"]
        mark = "*" if delta > 0 else " "
        print(f"{f:<8d}  {da['n']:<6d}  "
              f"{da['auc']:<14.4f}  {db['auc']:<14.4f}  "
              f"{delta:+.4f}{mark}  "
              f"{da['sens']:.4f}     {db['sens']:.4f}")

    # ─── Test pareado ────────────────────────────────────────────
    auc_a = np.array([a[f]["auc"] for f in valid])
    auc_b = np.array([b[f]["auc"] for f in valid])
    diff = auc_b - auc_a

    print(f"\n{'='*100}")
    print("TEST PAREADO POR SITIO")
    print(f"{'='*100}")
    print(f"  n sitios: {len(valid)}")
    print(f"  AUC {label_a}: {auc_a.mean():.4f} ± {auc_a.std():.4f}")
    print(f"  AUC {label_b}: {auc_b.mean():.4f} ± {auc_b.std():.4f}")
    print(f"  Δ AUC medio: {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}")

    # t-test pareado
    t_stat, p_t = stats.ttest_rel(auc_b, auc_a)
    print(f"\n  t-test pareado: t={t_stat:.4f}  p={p_t:.4f}")

    # Wilcoxon (no paramétrico)
    if len(valid) >= 6:
        w_stat, p_w = stats.wilcoxon(auc_b, auc_a)
        print(f"  Wilcoxon signed-rank: W={w_stat:.4f}  p={p_w:.4f}")

    # Cohen's d (efecto pareado)
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    print(f"  Cohen's d (pareado): {d:.4f}")

    # Consistency
    improved = (diff > 0).sum()
    worsened = (diff < 0).sum()
    ties = (diff == 0).sum()
    print(f"\n  Mejoran con {label_b}: {improved}/{len(valid)}")
    print(f"  Empeoran: {worsened}/{len(valid)}")
    print(f"  Empate: {ties}/{len(valid)}")

    # ─── Interpretación ─────────────────────────────────────────
    print(f"\n{'='*100}")
    print("INTERPRETACIÓN")
    print(f"{'='*100}")

    def interp_p(p):
        if p < 0.01:   return "muy significativo (**)"
        if p < 0.05:   return "significativo (*)"
        if p < 0.10:   return "marginal (~)"
        return "no significativo (ns)"

    def interp_d(d):
        d = abs(d)
        if d < 0.2:  return "trivial"
        if d < 0.5:  return "pequeño"
        if d < 0.8:  return "mediano"
        return "grande"

    print(f"  p-value (t-test): {p_t:.4f}  →  {interp_p(p_t)}")
    print(f"  Cohen's d:        {d:+.4f}  →  efecto {interp_d(d)}")

    if p_t < 0.05 and abs(d) >= 0.5:
        verdict = "EFECTO REAL Y RELEVANTE"
    elif p_t < 0.05 and abs(d) >= 0.2:
        verdict = "EFECTO REAL PERO PEQUEÑO"
    elif p_t < 0.10:
        verdict = "TENDENCIA, REQUIERE MÁS DATOS"
    else:
        verdict = "NO HAY EVIDENCIA DE MEJORA"

    print(f"\n  → {verdict}")

    if improved >= len(valid) * 0.75 and p_t < 0.05:
        print(f"  → Consistencia alta ({improved}/{len(valid)}) refuerza la señal")
    elif improved < len(valid) * 0.6:
        print(f"  → Consistencia baja ({improved}/{len(valid)}), cuidado con sobreinterpretar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_a", help="checkpoints dir config A (baseline)")
    parser.add_argument("dir_b", help="checkpoints dir config B (tratamiento)")
    parser.add_argument("--label_a", default="no_grl")
    parser.add_argument("--label_b", default="grl")
    args = parser.parse_args()
    main(args.dir_a, args.dir_b, args.label_a, args.label_b)
