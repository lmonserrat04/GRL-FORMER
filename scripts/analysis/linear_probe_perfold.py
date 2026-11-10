# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Linear probe per-fold sobre features extraídas del mismo modelo N.

Para cada fold N (sitio held out):
  1. Reconstruir el modelo N (build_experiment + best_finetune_fold_N.pt).
  2. Extraer features de train_N y test_N con ese mismo modelo N.
  3. Estandarizar (fit en train_N, apply a test_N).
  4. LogisticRegression(C).
  5. AUC sobre test_N.

Diferencias con `linear_probe.py`:
  - `linear_probe.py` mezcla features de 19 modelos distintos → contaminación.
  - `linear_probe_perfold.py` usa, para cada fold, las features del mismo
    modelo N que generó el finetune. Sin mezcla de espacios latentes.

Salidas:
  <out>/features/fold_N_{train,test}.npz  (si --save-features)
  <out>/linear_probe_perfold_by_site.csv
  <out>/linear_probe_perfold_global.json
  <out>/linear_probe_perfold_report.md

Uso:
    python scripts/analysis/linear_probe_perfold.py \\
        --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds \\
        --representation fused \\
        --C 1.0 \\
        --save-features
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from training.setup import build_experiment
from data.loaders.dataloader import get_finetune_loaders


VALID_REPS = ("fused", "h_ts_proj", "h_fc_proj", "h_ts_raw", "h_fc_raw")


def extract_features_for_probe(model, loader, device):
    """
    Extrae features de un loader dado, a través del modelo (encoders + proj + fusion).

    Devuelve dict:
        fused:     (n, 128)
        h_ts_proj: (n, 128)
        h_fc_proj: (n, 128)
        h_ts_raw:  (n, 512)
        h_fc_raw:  (n, 256)
        labels:    (n,)
        sub_ids:   (n,)  (si el loader los tiene)
    """
    model.eval()

    h_ts_raw_all = []
    h_ts_proj_all = []
    h_fc_raw_all = []
    h_fc_proj_all = []
    fused_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)

            h_ts_raw = model.transformer_ts(ts, mode="finetune")
            h_fc_raw = model.transformer_fc(pcc, mode="finetune")

            h_ts_proj = (model.proj_head_1(h_ts_raw)
                         if model.proj_head_1 is not None else h_ts_raw)
            h_fc_proj = (model.proj_head_2(h_fc_raw)
                         if model.proj_head_2 is not None else h_fc_raw)

            fused = model.fusion(h_ts_proj, h_fc_proj)

            h_ts_raw_all.append(h_ts_raw.cpu().numpy())
            h_ts_proj_all.append(h_ts_proj.cpu().numpy())
            h_fc_raw_all.append(h_fc_raw.cpu().numpy())
            h_fc_proj_all.append(h_fc_proj.cpu().numpy())
            fused_all.append(fused.cpu().numpy())
            labels_all.append(y.cpu().numpy())

    out = {
        "h_ts_raw":   np.concatenate(h_ts_raw_all,  axis=0).astype(np.float32),
        "h_ts_proj":  np.concatenate(h_ts_proj_all, axis=0).astype(np.float32),
        "h_fc_raw":   np.concatenate(h_fc_raw_all,  axis=0).astype(np.float32),
        "h_fc_proj":  np.concatenate(h_fc_proj_all, axis=0).astype(np.float32),
        "fused":      np.concatenate(fused_all,     axis=0).astype(np.float32),
        "labels":     np.concatenate(labels_all,    axis=0).astype(np.int64),
    }

    # sub_ids: intentar desde dataset si expone site_ids o desde el sampler
    ds = loader.dataset
    if hasattr(ds, "sub_ids"):
        out["sub_ids"] = ds.sub_ids.astype(np.int64)
    return out


def fit_linear(X_train, y_train, X_test, C=1.0, max_iter=2000):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_test_s)[:, 1]


def bootstrap_ci_paired_sites(labels_per_site, probs_per_site,
                              n_bootstrap=2000, seed=42):
    """Bootstrap sobre sujetos por sitio. Descarta remuestreos de una clase."""
    rng = np.random.default_rng(seed)
    means = []
    attempts = 0
    max_attempts = n_bootstrap * 20
    while len(means) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        site_aucs = []
        ok = True
        for labels, probs in zip(labels_per_site, probs_per_site):
            n = len(labels)
            idx = rng.integers(0, n, size=n)
            y_b = labels[idx]
            p_b = probs[idx]
            if len(np.unique(y_b)) < 2:
                ok = False
                break
            site_aucs.append(roc_auc_score(y_b, p_b))
        if ok:
            means.append(float(np.mean(site_aucs)))
    return np.array(means)


def paired_stats(a, b):
    d = a - b
    n = len(d)
    if n >= 2 and np.std(d, ddof=1) > 1e-12:
        t, p_t = ttest_rel(a, b)
    else:
        t, p_t = float("nan"), float("nan")
    try:
        if n >= 2 and np.any(d != 0):
            _, p_w = wilcoxon(a, b)
        else:
            p_w = float("nan")
    except Exception:
        p_w = float("nan")
    sd = np.std(d, ddof=1) if n >= 2 else 0.0
    cohen = float(np.mean(d) / sd) if sd > 1e-12 else float("nan")
    return {
        "n_sites": n,
        "delta_mean": float(np.mean(d)),
        "delta_std": sd,
        "delta_min": float(np.min(d)),
        "delta_max": float(np.max(d)),
        "t_stat": float(t) if not np.isnan(t) else None,
        "p_ttest": float(p_t) if not np.isnan(p_t) else None,
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else None,
        "cohen_d": cohen if not np.isnan(cohen) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--representation", default="fused", choices=VALID_REPS)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--save-features", action="store_true",
                        help="Guardar features por fold a disco")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override FINETUNING.BATCH_SIZE para la extracción")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = run_dir / "config.yaml"
    ckpt_dir = run_dir / "checkpoints"
    if not config_path.exists():
        raise SystemExit(f"No existe {config_path}")
    if not ckpt_dir.exists():
        raise SystemExit(f"No existe {ckpt_dir}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["EXPERIMENT_TYPE"] = "finetune"
    config["DEVICE"] = args.device
    config["NUM_WORKERS"] = 0

    out_dir = (args.out_dir if args.out_dir else
               run_dir / f"linear_probe_perfold_{args.representation}")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_out_dir = out_dir / "features"
    if args.save_features:
        feat_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run:              {run_dir}")
    print(f"Representación:   {args.representation}")
    print(f"C:                {args.C}")
    print(f"Device:           {args.device}")
    print(f"Save features:    {args.save_features}")
    print(f"Out:              {out_dir}\n")

    ckpts = sorted(ckpt_dir.glob("best_finetune_fold_*.pt"))
    if not ckpts:
        raise SystemExit(f"No hay best_finetune_fold_*.pt en {ckpt_dir}")
    fold_ids = sorted([int(c.stem.split("_")[-1]) for c in ckpts])
    if args.folds is not None:
        fold_ids = [f for f in fold_ids if f in args.folds]

    bs = args.batch_size or config["FINETUNING"]["BATCH_SIZE"]

    rows = []
    labels_per_site = []
    probs_per_site = []
    aucs_probe = []
    aucs_base = []

    for fold_idx in fold_ids:
        print(f"=== Fold {fold_idx} ===")

        # 1) Reconstruir modelo y cargar checkpoint
        exp = build_experiment(
            config, fold_idx=fold_idx,
            ckpt_contrastive=config.get("CKPT_CONTRASTIVE"),
        )
        model = exp.model
        ckpt = torch.load(ckpt_dir / f"best_finetune_fold_{fold_idx}.pt",
                          map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.to(args.device).eval()

        # 2) Loaders (mismo split que el finetune original)
        train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
            config, batch_size=bs, num_workers=0,
            fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5),
            seed=config.get("SEED", 42),
            eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
        )

        # 3) Extraer features
        feat_train = extract_features_for_probe(model, train_loader, args.device)
        feat_test  = extract_features_for_probe(model, test_loader,  args.device)

        X_train = feat_train[args.representation]
        y_train = feat_train["labels"]
        X_test  = feat_test[args.representation]
        y_test  = feat_test["labels"]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  [SKIP] una sola clase en train o test")
            continue

        # 4) Linear probe
        probs_test = fit_linear(X_train, y_train, X_test, C=args.C,
                                max_iter=args.max_iter)
        auc_probe = float(roc_auc_score(y_test, probs_test))

        # 5) Baseline AUC (mismo fold, seed 42)
        bl = np.load(ckpt_dir / f"preds_fold_{fold_idx}.npz", allow_pickle=True)
        # Verificar alineamiento
        if not np.array_equal(bl["labels"], y_test):
            raise SystemExit(
                f"Desalineamiento en fold {fold_idx}: "
                f"labels baseline vs features"
            )
        auc_base = float(roc_auc_score(bl["labels"], bl["probs"]))

        delta = auc_probe - auc_base

        n = len(y_test)
        n_td  = int((y_test == 0).sum())
        n_asd = int((y_test == 1).sum())

        rows.append({
            "fold": fold_idx,
            "n": n, "n_td": n_td, "n_asd": n_asd,
            "auc_probe": auc_probe,
            "auc_baseline": auc_base,
            "delta": delta,
        })
        labels_per_site.append(y_test)
        probs_per_site.append(probs_test)
        aucs_probe.append(auc_probe)
        aucs_base.append(auc_base)

        print(f"  n={n:>3d} (TD={n_td}, ASD={n_asd})  "
              f"probe={auc_probe:.4f}  baseline={auc_base:.4f}  "
              f"Δ={delta:+.4f}")

        # 6) Guardar features
        if args.save_features:
            np.savez(feat_out_dir / f"fold_{fold_idx}_train.npz",
                     fused=feat_train["fused"],
                     h_ts_proj=feat_train["h_ts_proj"],
                     h_fc_proj=feat_train["h_fc_proj"],
                     labels=feat_train["labels"])
            np.savez(feat_out_dir / f"fold_{fold_idx}_test.npz",
                     fused=feat_test["fused"],
                     h_ts_proj=feat_test["h_ts_proj"],
                     h_fc_proj=feat_test["h_fc_proj"],
                     labels=feat_test["labels"])

        # liberar GPU
        del exp, model
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("Sin folds válidos")

    df = pd.DataFrame(rows)
    aucs_probe = np.array(aucs_probe)
    aucs_base = np.array(aucs_base)

    csv_out = out_dir / "linear_probe_perfold_by_site.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nGuardado: {csv_out}")

    probe_mean = float(aucs_probe.mean())
    probe_std  = float(aucs_probe.std(ddof=1)) if len(aucs_probe) >= 2 else 0.0
    base_mean  = float(aucs_base.mean())
    base_std   = float(aucs_base.std(ddof=1)) if len(aucs_base) >= 2 else 0.0

    print(f"\nBootstrap IC 95% (B={args.n_bootstrap})...")
    boot_means = bootstrap_ci_paired_sites(
        labels_per_site, probs_per_site,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    if len(boot_means) >= 2:
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    paired = paired_stats(aucs_probe, aucs_base)

    t_str  = f"{paired['t_stat']:.3f}"   if paired['t_stat']   is not None else "—"
    pt_str = f"{paired['p_ttest']:.4f}"  if paired['p_ttest']  is not None else "—"
    pw_str = f"{paired['p_wilcoxon']:.4f}" if paired['p_wilcoxon'] is not None else "—"
    cd_str = f"{paired['cohen_d']:.3f}"  if paired['cohen_d']  is not None else "—"

    print("\n" + "="*70)
    print(f"RESUMEN — Linear probe per-fold ({args.representation}, C={args.C})")
    print("="*70)
    print(f"  Probe AUC:      {probe_mean:.4f} ± {probe_std:.4f}  "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Baseline AUC:   {base_mean:.4f} ± {base_std:.4f}  (seed 42)")
    print(f"  Δ medio:        {paired['delta_mean']:+.4f} "
          f"± {paired['delta_std']:.4f}")
    print(f"  t-test pareado: t={t_str}  p={pt_str}")
    print(f"  Wilcoxon:       p={pw_str}")
    print(f"  Cohen's d:      {cd_str}")

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
    json_out = out_dir / "linear_probe_perfold_global.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGuardado: {json_out}")

    # Markdown
    md = ["# Linear probe per-fold — TwoTST (LOSO)", ""]
    md.append(f"**Run:** `{run_dir}`")
    md.append(f"**Representación:** `{args.representation}`")
    md.append(f"**C:** {args.C}")
    md.append(f"**Folds:** {len(df)}")
    md.append("")
    md.append("## Metodología")
    md.append("")
    md.append("Para cada fold N (sitio held out):")
    md.append("1. Reconstruir el modelo N y cargar `best_finetune_fold_N.pt`.")
    md.append("2. Extraer features de `train_N` y `test_N` con ese modelo N.")
    md.append("3. Estandarización (fit en train, apply a test).")
    md.append(f"4. `LogisticRegression(C={args.C})` sobre train, predecir test.")
    md.append("5. AUC sobre test.")
    md.append("")
    md.append("**Sin mezcla de espacios latentes:** cada fold usa solo "
              "las features del modelo N, que es el mismo que generó las "
              "predicciones baseline del fold N.")
    md.append("")
    md.append("## Tabla por sitio")
    md.append("")
    md.append("| Fold | n | n_td | n_asd | AUC probe | AUC baseline (seed 42) | Δ |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(f"| {int(r['fold'])} | {int(r['n'])} | {int(r['n_td'])} | "
                  f"{int(r['n_asd'])} | {r['auc_probe']:.4f} | "
                  f"{r['auc_baseline']:.4f} | {r['delta']:+.4f} |")
    md.append("")
    md.append("## Resumen global")
    md.append("")
    md.append(f"- **Probe AUC:** {probe_mean:.4f} ± {probe_std:.4f}  "
              f"[{ci_lo:.4f}, {ci_hi:.4f}] (IC bootstrap pareado por sujeto)")
    md.append(f"- **Baseline AUC (seed 42):** {base_mean:.4f} ± {base_std:.4f}")
    md.append(f"- **Δ medio:** {paired['delta_mean']:+.4f} ± "
              f"{paired['delta_std']:.4f}")
    md.append("")
    md.append("## Test pareado")
    md.append("")
    md.append(f"- t-test: t={t_str}, p={pt_str}")
    md.append(f"- Wilcoxon: p={pw_str}")
    md.append(f"- Cohen's d: {cd_str}")
    md.append("")
    md.append("## Interpretación")
    md.append("")
    md.append("- **Probe AUC ≥ baseline** → la representación de `fused` es "
              "linealmente separable por sí sola. El clasificador entrenado no "
              "aprovecha más que una regresión logística.")
    md.append("- **Probe AUC < baseline** → el clasificador MLP del modelo "
              "extrae información no lineal que el linear probe no captura.")
    md.append("- **Probe AUC ≈ 0.5** → `fused` no contiene señal lineal "
              "cross-site para ASD/TD en ese sitio.")

    md_out = out_dir / "linear_probe_perfold_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
