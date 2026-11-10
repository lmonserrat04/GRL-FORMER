# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Linear probe sobre las features de un run LOSO.

Para cada fold N (sitio held out):
  1. Train: concatenar `features_fold_M[rep]` para todos M != N
     (18 sitios, ~790 sujetos).
  2. Test: `features_fold_N[rep]` (sitio held out, ~n sujetos).
  3. Estandarizar por dimensión (fit en train, apply a test).
  4. LogisticRegression(C=C).
  5. AUC sobre test.
  6. Recalcular AUC del baseline (probs guardadas) para el mismo fold.

Al final:
  - Media ± std de AUC del probe.
  - Bootstrap IC 95% pareado por sujeto del AUC medio del probe.
  - Test pareado (t-test, Wilcoxon, Cohen's d) probe vs baseline seed 42.
  - Guarda CSV, JSON y MD.

Uso:
    python scripts/analysis/linear_probe.py \\
        --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds \\
        --representation fused \\
        --C 1.0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


VALID_REPS = ("fused", "h_ts_proj", "h_fc_proj", "h_ts_raw", "h_fc_raw")


def load_all_folds(feat_dir: Path, representation: str):
    """
    Carga features de los 19 folds.

    Devuelve:
        dict {fold_idx: {"X": (n, d), "y": (n,), "sub_ids": (n,)}}
    """
    files = sorted(feat_dir.glob("features_fold_*.npz"))
    if not files:
        raise SystemExit(f"No hay features_fold_*.npz en {feat_dir}")

    out = {}
    for f in files:
        fold_idx = int(f.stem.split("_")[-1])
        d = np.load(f, allow_pickle=True)
        if representation not in d.files:
            raise SystemExit(
                f"Representación {representation!r} no encontrada en {f.name}. "
                f"Disponibles: {list(d.files)}"
            )
        out[fold_idx] = {
            "X": d[representation].astype(np.float32),
            "y": d["labels"].astype(np.int64),
            "sub_ids": d["sub_ids"].astype(np.int64),
        }
    return out


def load_baseline_probs(ckpt_dir: Path):
    """Carga probs y labels del baseline (preds_fold_N.npz)."""
    out = {}
    for f in sorted(ckpt_dir.glob("preds_fold_*.npz")):
        fold_idx = int(f.stem.split("_")[-1])
        d = np.load(f, allow_pickle=True)
        out[fold_idx] = {
            "labels": d["labels"].astype(np.int64),
            "probs":  d["probs"].astype(np.float64),
        }
    return out


def fit_linear_probe(X_train, y_train, X_test, C=1.0, max_iter=2000):
    """Estandarización + regresión logística."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]
    return probs


def bootstrap_ci_paired_sites(
    labels_per_site: list,
    probs_per_site: list,
    n_bootstrap: int = 2000,
    seed: int = 42,
):
    """
    Bootstrap sobre sujetos POR sitio, para IC del AUC medio entre sitios.

    Cada remuestreo:
      - Para cada sitio, remuestrea sus sujetos con reemplazo.
      - Recalcula AUC por sitio.
      - Promedia los 19 AUCs.
    Descarta remuestreos donde algún sitio tiene una sola clase.
    """
    rng = np.random.default_rng(seed)
    n_sites = len(labels_per_site)
    means = []
    attempts = 0
    max_attempts = n_bootstrap * 20

    while len(means) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        site_aucs = []
        valid = True
        for labels, probs in zip(labels_per_site, probs_per_site):
            n = len(labels)
            idx = rng.integers(0, n, size=n)
            y_b = labels[idx]
            p_b = probs[idx]
            if len(np.unique(y_b)) < 2:
                valid = False
                break
            site_aucs.append(roc_auc_score(y_b, p_b))
        if not valid:
            continue
        means.append(float(np.mean(site_aucs)))

    return np.array(means)


def paired_stats(aucs_probe: np.ndarray, aucs_baseline: np.ndarray):
    """t-test + Wilcoxon + Cohen's d pareado. Robusto a n=1."""
    assert aucs_probe.shape == aucs_baseline.shape
    d = aucs_probe - aucs_baseline
    n = len(d)

    # t-test (requiere n>=2 y sd>0)
    if n >= 2 and np.std(d, ddof=1) > 1e-12:
        t, p_t = ttest_rel(aucs_probe, aucs_baseline)
    else:
        t, p_t = float("nan"), float("nan")

    # Wilcoxon
    try:
        if n >= 2 and np.any(d != 0):
            _, p_w = wilcoxon(aucs_probe, aucs_baseline)
        else:
            p_w = float("nan")
    except Exception:
        p_w = float("nan")

    # Cohen's d pareado
    sd = np.std(d, ddof=1) if n >= 2 else 0.0
    cohen = float(np.mean(d) / sd) if sd > 1e-12 else float("nan")

    return {
        "n_sites": n,
        "delta_mean": float(np.mean(d)),
        "delta_std": float(np.std(d, ddof=1)) if n >= 2 else 0.0,
        "delta_min": float(np.min(d)),
        "delta_max": float(np.max(d)),
        "t_stat": float(t) if not np.isnan(t) else None,
        "p_ttest": float(p_t) if not np.isnan(p_t) else None,
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else None,
        "cohen_d": cohen if not np.isnan(cohen) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run con features/ y checkpoints/preds_fold_*.npz")
    parser.add_argument("--representation", default="fused",
                        choices=VALID_REPS)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Regularización L2 inversa (default 1.0)")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subconjunto de folds (default: todos)")
    args = parser.parse_args()

    run_dir = args.run_dir
    feat_dir = run_dir / "features"
    ckpt_dir = run_dir / "checkpoints"

    if not feat_dir.exists():
        raise SystemExit(f"No existe {feat_dir}")
    if not ckpt_dir.exists():
        raise SystemExit(f"No existe {ckpt_dir}")

    out_dir = (args.out_dir if args.out_dir else
               run_dir / f"linear_probe_{args.representation}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run:            {run_dir}")
    print(f"Representación: {args.representation}")
    print(f"C:              {args.C}")
    print(f"Out dir:        {out_dir}\n")

    folds = load_all_folds(feat_dir, args.representation)
    baseline = load_baseline_probs(ckpt_dir)

    fold_ids = sorted(folds.keys())
    if args.folds is not None:
        fold_ids = [f for f in fold_ids if f in args.folds]

    print(f"Folds: {fold_ids}\n")

    rows = []
    labels_per_site = []
    probs_per_site_probe = []
    aucs_probe = []
    aucs_baseline = []

    for test_fold in fold_ids:
        # Train: concatenar features de todos los folds != test_fold
        train_X = []
        train_y = []
        for k, v in folds.items():
            if k == test_fold:
                continue
            train_X.append(v["X"])
            train_y.append(v["y"])
        X_train = np.concatenate(train_X, axis=0)
        y_train = np.concatenate(train_y, axis=0)

        # Test: fold held-out
        X_test = folds[test_fold]["X"]
        y_test = folds[test_fold]["y"]

        if len(np.unique(y_test)) < 2:
            print(f"  [SKIP] fold {test_fold}: una sola clase en test")
            continue
        if len(np.unique(y_train)) < 2:
            print(f"  [SKIP] fold {test_fold}: una sola clase en train")
            continue

        # Linear probe
        probs_test = fit_linear_probe(
            X_train, y_train, X_test,
            C=args.C, max_iter=args.max_iter,
        )
        auc_probe = float(roc_auc_score(y_test, probs_test))

        # Baseline AUC (misma fold, seed 42)
        bl = baseline[test_fold]
        if not np.array_equal(bl["labels"], y_test):
            raise SystemExit(
                f"Desalineamiento en fold {test_fold}: labels probe vs baseline"
            )
        auc_base = float(roc_auc_score(bl["labels"], bl["probs"]))

        delta = auc_probe - auc_base

        n = len(y_test)
        n_td  = int((y_test == 0).sum())
        n_asd = int((y_test == 1).sum())

        rows.append({
            "fold": test_fold,
            "n": n,
            "n_td": n_td,
            "n_asd": n_asd,
            "auc_probe": auc_probe,
            "auc_baseline": auc_base,
            "delta": delta,
        })

        labels_per_site.append(y_test)
        probs_per_site_probe.append(probs_test)
        aucs_probe.append(auc_probe)
        aucs_baseline.append(auc_base)

        print(f"  fold {test_fold:2d}: n={n:>3d}  "
              f"probe={auc_probe:.4f}  baseline={auc_base:.4f}  "
              f"Δ={delta:+.4f}")

    if not rows:
        raise SystemExit("Sin folds válidos")

    df = pd.DataFrame(rows)
    aucs_probe = np.array(aucs_probe)
    aucs_baseline = np.array(aucs_baseline)

    # Guardar CSV
    csv_out = out_dir / "linear_probe_by_site.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nGuardado: {csv_out}")

    # Global
    probe_mean = float(aucs_probe.mean())
    probe_std  = float(aucs_probe.std(ddof=1)) if len(aucs_probe) >= 2 else 0.0
    base_mean  = float(aucs_baseline.mean())
    base_std   = float(aucs_baseline.std(ddof=1)) if len(aucs_baseline) >= 2 else 0.0

    # Bootstrap IC pareado por sujeto (sobre AUC medio entre sitios)
    print(f"\nBootstrap IC 95% (B={args.n_bootstrap})...")
    boot_means = bootstrap_ci_paired_sites(
        labels_per_site, probs_per_site_probe,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    if len(boot_means) >= 2:
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    # Test pareado
    paired = paired_stats(aucs_probe, aucs_baseline)

    print("\n" + "="*70)
    print(f"RESUMEN — Linear probe ({args.representation}, C={args.C})")
    print("="*70)
    print(f"  Probe AUC:      {probe_mean:.4f} ± {probe_std:.4f}  "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Baseline AUC:   {base_mean:.4f} ± {base_std:.4f}  (seed 42)")
    print(f"  Δ medio:        {paired['delta_mean']:+.4f} "
          f"± {paired['delta_std']:.4f}  "
          f"[{paired['delta_min']:+.4f}, {paired['delta_max']:+.4f}]")
    t_str = f"{paired['t_stat']:.3f}" if paired['t_stat'] is not None else "—"
    pt_str = f"{paired['p_ttest']:.4f}" if paired['p_ttest'] is not None else "—"
    pw_str = f"{paired['p_wilcoxon']:.4f}" if paired['p_wilcoxon'] is not None else "—"
    cd_str = f"{paired['cohen_d']:.3f}" if paired['cohen_d'] is not None else "—"
    print(f"  t-test pareado: t={t_str}  p={pt_str}")
    print(f"  Wilcoxon:       p={pw_str}")
    print(f"  Cohen's d:      {cd_str}")

    # JSON
    summary = {
        "run_dir": str(run_dir),
        "representation": args.representation,
        "C": args.C,
        "n_folds": len(df),
        "probe": {
            "auc_mean": probe_mean,
            "auc_std": probe_std,
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
            "n_bootstrap": int(len(boot_means)),
        },
        "baseline_seed42": {
            "auc_mean": base_mean,
            "auc_std": base_std,
        },
        "paired_test": paired,
        "per_site": df.to_dict(orient="records"),
    }

    json_out = out_dir / "linear_probe_global.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGuardado: {json_out}")

    # Markdown
    md = ["# Linear probe — TwoTST (LOSO)", ""]
    md.append(f"**Run:** `{run_dir}`")
    md.append(f"**Representación:** `{args.representation}`")
    md.append(f"**C:** {args.C}")
    md.append(f"**Folds:** {len(df)}")
    md.append("")
    md.append("## Metodología")
    md.append("")
    md.append("Para cada fold N (sitio held out):")
    md.append("1. Train: features de los 18 sitios restantes (concatenadas).")
    md.append("2. Test:  features del sitio held out.")
    md.append("3. Estandarización (fit en train, apply a test).")
    md.append(f"4. LogisticRegression(C={args.C}, L2, lbfgs).")
    md.append("5. AUC sobre test.")
    md.append("")
    md.append("**Sin fuga:** el linear probe solo ve features de train durante el fit.")
    md.append("")
    md.append("## Tabla por sitio")
    md.append("")
    md.append("| Fold | Site | n | n_td | n_asd | AUC probe | AUC baseline (seed 42) | Δ |")
    md.append("|---|---|---|---|---|---|---|---|")
    site_lookup = {}
    try:
        import pandas as pd_csv
        df_csv = pd_csv.read_csv("data/csv/data_train.csv")
        # No podemos asociar site a fold sin el splitter, así que usamos fold como identificador
    except Exception:
        pass
    for _, r in df.iterrows():
        md.append(f"| {int(r['fold'])} | — | {int(r['n'])} | {int(r['n_td'])} | "
                  f"{int(r['n_asd'])} | {r['auc_probe']:.4f} | "
                  f"{r['auc_baseline']:.4f} | {r['delta']:+.4f} |")
    md.append("")
    md.append("## Resumen global")
    md.append("")
    md.append(f"- **Probe AUC:** {probe_mean:.4f} ± {probe_std:.4f}  "
              f"[{ci_lo:.4f}, {ci_hi:.4f}] (IC bootstrap pareado por sujeto)")
    md.append(f"- **Baseline AUC (seed 42):** {base_mean:.4f} ± {base_std:.4f}")
    md.append(f"- **Δ medio:** {paired['delta_mean']:+.4f} "
              f"± {paired['delta_std']:.4f}")
    md.append("")
    md.append("## Test pareado (probe vs baseline seed 42)")
    md.append("")
    md.append(f"- t-test: t={t_str}, p={pt_str}")
    md.append(f"- Wilcoxon: p={pw_str}")
    md.append(f"- Cohen's d: {cd_str}")
    md.append("")
    md.append("## Interpretación")
    md.append("")
    md.append("- Si **probe AUC ≥ baseline AUC** → las features contienen señal lineal")
    md.append("  comparable a la del clasificador entrenado. El clasificador no")
    md.append("  desperdicia la representación.")
    md.append("- Si **probe AUC < baseline AUC** → el clasificador del modelo")
    md.append("  (MLP no lineal) extrae información que un linear probe no")
    md.append("  captura. La representación es no linealmente separable.")
    md.append("- Si **probe AUC >> baseline AUC** → el clasificador del modelo")
    md.append("  está desaprovechando la señal lineal de `fused`.")

    md_out = out_dir / "linear_probe_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
