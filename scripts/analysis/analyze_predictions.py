"""
Analiza predicciones crudas guardadas por finetune_fold.

Parámetro MIN_N: descarta folds con menos de MIN_N sujetos en test
(solo afecta a los agregados; el per-fold siempre se imprime completo).
"""
# --- sys.path bootstrap (scripts movidos a subcarpetas) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from utils.metrics import bootstrap_confidence_interval


MIN_N = 30   # mínimo de muestras de test por fold para entrar en agregados


def sens_spec_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0, 0.0
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens), float(spec)


def pooled_optimal_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[j.argmax()])


def analyze_dir(ckpt_dir: Path, name: str, min_n: int):
    npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
    if not npz_files:
        return None

    per_fold_all = []

    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        labels = d["labels"]
        probs = d["probs"]
        optimal_thr = float(d["optimal_thr"][0])

        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        sens_05, spec_05 = sens_spec_at(labels, probs, 0.5)
        sens_opt, spec_opt = sens_spec_at(labels, probs, optimal_thr)

        per_fold_all.append({
            "fold": f.stem,
            "n": int(len(labels)),
            "auc": float(auc),
            "sens_0.5": sens_05, "spec_0.5": spec_05,
            "sens_opt": sens_opt, "spec_opt": spec_opt,
            "optimal_thr": optimal_thr,
        })

    # Filtro
    per_fold = [f for f in per_fold_all if f["n"] >= min_n]
    excluded = [f["fold"] for f in per_fold_all if f["n"] < min_n]

    # Pooled solo con los folds válidos
    pooled_labels, pooled_probs = [], []
    for f in npz_files:
        name_ = f.stem
        if name_ in [x["fold"] for x in per_fold]:
            d = np.load(f, allow_pickle=True)
            pooled_labels.append(d["labels"])
            pooled_probs.append(d["probs"])

    if not per_fold:
        return {"name": name, "error": f"ningún fold tiene n>={min_n}"}

    pooled_labels = np.concatenate(pooled_labels)
    pooled_probs = np.concatenate(pooled_probs)
    pooled_thr = pooled_optimal_threshold(pooled_labels, pooled_probs)

    pooled_auc = roc_auc_score(pooled_labels, pooled_probs) if len(np.unique(pooled_labels)) > 1 else 0.0
    pooled_sens_05, pooled_spec_05 = sens_spec_at(pooled_labels, pooled_probs, 0.5)
    pooled_sens_opt, pooled_spec_opt = sens_spec_at(pooled_labels, pooled_probs, pooled_thr)

    def summarize(key):
        vals = [f[key] for f in per_fold]
        mean_v, std_v, lo, hi = bootstrap_confidence_interval(
            vals, n_bootstrap=1000, ci=0.95, seed=42)
        return {"mean": mean_v, "std": std_v, "ci95": [lo, hi],
                "min": float(min(vals)), "max": float(max(vals))}

    return {
        "name": name,
        "min_n": min_n,
        "excluded_folds": excluded,
        "n_valid_folds": len(per_fold),
        "per_fold_all": per_fold_all,
        "per_fold_filtered": per_fold,
        "pooled": {
            "n": int(len(pooled_labels)),
            "auc": float(pooled_auc),
            "sens_0.5": pooled_sens_05, "spec_0.5": pooled_spec_05,
            "sens_opt": pooled_sens_opt, "spec_opt": pooled_spec_opt,
            "optimal_thr": pooled_thr,
        },
        "aggregated": {
            "auc":      summarize("auc"),
            "sens_0.5": summarize("sens_0.5"),
            "spec_0.5": summarize("spec_0.5"),
            "sens_opt": summarize("sens_opt"),
            "spec_opt": summarize("spec_opt"),
        },
    }


def print_report(res):
    if "error" in res:
        print(f"\n[ERROR] {res['name']}: {res['error']}")
        return

    print(f"\n{'='*90}")
    print(f"ANALYSIS: {res['name']}   (min_n={res['min_n']}, "
          f"folds válidos={res['n_valid_folds']})")
    print(f"{'='*90}")

    if res["excluded_folds"]:
        print(f"\n[EXCLUIDOS por n<{res['min_n']}]: {res['excluded_folds']}")

    print("\n--- Per fold (TODOS) ---")
    print(f"{'fold':<22s} {'n':>4s}  {'AUC':<8s} "
          f"{'Sens@.5':<10s} {'Spec@.5':<10s} "
          f"{'Sens@opt':<10s} {'Spec@opt':<10s} {'thr':<8s}")
    for f in res["per_fold_all"]:
        marker = " " if f["n"] >= res["min_n"] else "*"
        print(f"{marker}{f['fold']:<21s} {f['n']:>4d}  {f['auc']:.4f}   "
              f"{f['sens_0.5']:.4f}     {f['spec_0.5']:.4f}     "
              f"{f['sens_opt']:.4f}     {f['spec_opt']:.4f}     "
              f"{f['optimal_thr']:.3f}")

    print("\n--- Pooled (solo folds válidos) ---")
    p = res["pooled"]
    print(f"  n = {p['n']}")
    print(f"  AUC:       {p['auc']:.4f}")
    print(f"  @ 0.5    : Sens={p['sens_0.5']:.4f}  Spec={p['spec_0.5']:.4f}")
    print(f"  @ opt    : Sens={p['sens_opt']:.4f}  Spec={p['spec_opt']:.4f}  "
          f"(thr={p['optimal_thr']:.3f})")

    print("\n--- Agregado (solo folds válidos) ---")
    for name, s in res["aggregated"].items():
        lo, hi = s["ci95"]
        print(f"  {name:<10s}: {s['mean']:.4f} ± {s['std']:.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  |  min={s['min']:.4f}  max={s['max']:.4f}")


def main(root_dir, min_n=MIN_N):
    root = Path(root_dir)
    if not root.exists():
        print(f"[ERROR] no existe {root}")
        sys.exit(1)
    all_results = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        ckpt_dir = subdir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        res = analyze_dir(ckpt_dir, subdir.name, min_n)
        if res is None:
            print(f"[SKIP] {subdir.name}: sin preds_fold_*.npz")
            continue
        all_results[subdir.name] = res
        print_report(res)
    out = root / f"analysis_report_min{min_n}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] Guardado en {out}")


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments_sweep_minimal"
    min_n = int(sys.argv[2]) if len(sys.argv) > 2 else MIN_N
    main(root_dir, min_n)
