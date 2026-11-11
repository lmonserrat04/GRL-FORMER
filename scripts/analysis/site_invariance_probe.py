# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Site-invariance: ¿cuánta información de sitio queda en la representación fused?

Protocolo
---------
Para cada modelo LOSO (un fold = un sitio held-out):
  1. Se congelan encoder, projections y fusión.
  2. Se extrae ``fused`` para TODOS los sujetos (no solo el sitio held-out).
  3. Se entrena un clasificador logístico multinomial para predecir el
     sitio con validación cruzada estratificada de 5 folds sobre el pool.
  4. Se reportan accuracy, balanced accuracy y macro-F1.

Un clasificador de sitio cercano al azar (1/19 ≈ 0.053) indica que la
representación fused es site-invariante. Valores altos indican que la
identidad del sitio sigue siendo linealmente decodificable.

Uso:
    python scripts/analysis/site_invariance_probe.py \
        --run-dir experiments_finetune_only/<run> \
        --representation fused

Salidas (por defecto en <run-dir>/site_invariance/):
    site_invariance_by_fold.csv
    site_invariance_global.json
    site_invariance_report.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.loaders.dataloader import TwoTSTDataset, load_raw_data
from training.setup import build_experiment


VALID_REPS = ("fused", "h_ts_raw", "h_ts_proj", "h_fc_raw", "h_fc_proj")


def _extract_all(model, data: dict, representation: str, device: str,
                 batch_size: int) -> np.ndarray:
    ds = TwoTSTDataset(
        data["timeseries"], data["pcc_vectors"], data["labels"],
        sites=data["site_ids"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            ts = batch["timeseries"].to(device, non_blocking=True)
            pcc = batch["pcc_vector"].to(device, non_blocking=True)

            h_ts = model.transformer_ts(ts, mode="finetune")
            h_fc = model.transformer_fc(pcc, mode="finetune")
            h_ts_p = (model.proj_head_1(h_ts)
                      if model.proj_head_1 is not None else h_ts)
            h_fc_p = (model.proj_head_2(h_fc)
                      if model.proj_head_2 is not None else h_fc)

            if representation == "fused":
                feat = model.fusion(h_ts_p, h_fc_p)
            elif representation == "h_ts_raw":
                feat = h_ts
            elif representation == "h_ts_proj":
                feat = h_ts_p
            elif representation == "h_fc_raw":
                feat = h_fc
            elif representation == "h_fc_proj":
                feat = h_fc_p
            else:
                raise ValueError(f"Representación no soportada: {representation}")
            chunks.append(feat.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _site_cv(features: np.ndarray, sites: np.ndarray, n_splits: int,
             C: float, max_iter: int, seed: int,
             shuffle_labels: bool = False) -> dict:
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        sites = rng.permutation(sites)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in skf.split(features, sites):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", n_jobs=-1,
        )
        clf.fit(X_train, sites[train_idx])
        pred = clf.predict(X_test)
        y_true_all.extend(sites[test_idx])
        y_pred_all.extend(pred)

    y_true = np.asarray(y_true_all)
    y_pred = np.asarray(y_pred_all)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _summary(values: list[float], n_bootstrap: int, seed: int) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    if n_bootstrap <= 0 or arr.size < 2:
        return {"mean": mean, "std": std, "ci95_lower": None, "ci95_upper": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": mean, "std": std,
        "ci95_lower": float(lo), "ci95_upper": float(hi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--representation", default="fused", choices=list(VALID_REPS))
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subconjunto de folds (para smoke tests).")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <run-dir>/site_invariance/")
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

    if not config.get("CKPT_CONTRASTIVE"):
        raise SystemExit(
            "El config del run no define CKPT_CONTRASTIVE; no se pueden "
            "reconstruir las projection heads."
        )

    data = load_raw_data(config)
    sites = np.asarray(data["site_ids"])
    n_sites = len(np.unique(sites))
    counts = pd.Series(sites).value_counts()
    majority = float(counts.iloc[0] / counts.sum())

    out_dir = args.out_dir if args.out_dir else run_dir / "site_invariance"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(
        ckpt_dir.glob("best_finetune_fold_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    fold_ids = [int(p.stem.split("_")[-1]) for p in ckpts]
    if args.folds is not None:
        fold_ids = [f for f in fold_ids if f in args.folds]
    if not fold_ids:
        raise SystemExit("No hay checkpoints de finetune para evaluar.")

    print(f"Run:             {run_dir}")
    print(f"Representación:  {args.representation}")
    print(f"Sujetos:         {len(sites)}  |  sitios: {n_sites}")
    print(f"Chance accuracy: {1.0 / n_sites:.4f}  |  majority: {majority:.4f}")
    print(f"Folds:           {len(fold_ids)}")
    print(f"Device:          {args.device}\n")

    rows = []
    for fold_idx in fold_ids:
        exp = build_experiment(
            config, fold_idx=fold_idx,
            ckpt_contrastive=config["CKPT_CONTRASTIVE"],
        )
        model = exp.model
        ckpt = torch.load(
            ckpt_dir / f"best_finetune_fold_{fold_idx}.pt",
            map_location="cpu", weights_only=False,
        )
        state = (ckpt["model_state_dict"]
                 if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                 else ckpt)
        model.load_state_dict(state, strict=True)
        model.to(args.device).eval()

        features = _extract_all(
            model, data, args.representation, args.device, args.batch_size
        )
        m = _site_cv(
            features, sites, n_splits=args.n_splits, C=args.C,
            max_iter=args.max_iter, seed=args.seed,
        )
        m["fold"] = fold_idx
        m["n_subjects"] = int(len(sites))
        m["n_sites"] = int(n_sites)
        rows.append(m)
        print(f"  fold {fold_idx:2d}: acc={m['accuracy']:.4f}  "
              f"bal_acc={m['balanced_accuracy']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}")

        del exp, model, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)[
        ["fold", "n_subjects", "n_sites", "accuracy",
         "balanced_accuracy", "macro_f1"]
    ].sort_values("fold").reset_index(drop=True)
    csv_out = out_dir / "site_invariance_by_fold.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nGuardado: {csv_out}")

    summary = {
        "run_dir": str(run_dir),
        "representation": args.representation,
        "protocol": (
            f"StratifiedKFold({args.n_splits}) sobre el pool completo de "
            "sujetos, con features congeladas de cada modelo LOSO"
        ),
        "n_subjects": int(len(sites)),
        "n_sites": int(n_sites),
        "chance_accuracy": float(1.0 / n_sites),
        "majority_baseline": majority,
        "classifier": f"LogisticRegression(C={args.C}, max_iter={args.max_iter})",
        "metrics": {
            metric: _summary(df[metric].tolist(), args.n_bootstrap, args.seed)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        },
        "per_fold": df.to_dict(orient="records"),
    }
    json_out = out_dir / "site_invariance_global.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Guardado: {json_out}")

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    md = [
        "# Site-invariance sobre la representación fused",
        "",
        f"**Run:** `{run_dir}`",
        f"**Representación:** `{args.representation}`",
        f"**Sujetos:** {len(sites)}  |  **Sitios:** {n_sites}",
        f"**Chance accuracy:** {1.0 / n_sites:.4f}  |  "
        f"**Mayoría:** {majority:.4f}",
        "",
        "## Metodología",
        "",
        "Para cada modelo LOSO (encoder, projections y fusión congelados) se "
        "extrae la representación de todos los sujetos y se entrena un "
        "clasificador logístico multinomial para predecir el sitio con "
        f"validación cruzada estratificada de {args.n_splits} folds.",
        "",
        "## Resumen global",
        "",
        "| Métrica | Media | std | IC 95% |",
        "|---|---|---|---|",
    ]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        s = summary["metrics"][metric]
        ci = (f"[{fmt(s['ci95_lower'])}, {fmt(s['ci95_upper'])}]"
              if s["ci95_lower"] is not None else "—")
        md.append(f"| {metric} | {fmt(s['mean'])} | {fmt(s['std'])} | {ci} |")
    md += [
        "",
        "## Resultados por fold",
        "",
        "| Fold | Sujetos | Sitios | Accuracy | Balanced acc | Macro F1 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {int(r['fold'])} | {int(r['n_subjects'])} | {int(r['n_sites'])} | "
            f"{fmt(r['accuracy'])} | {fmt(r['balanced_accuracy'])} | "
            f"{fmt(r['macro_f1'])} |"
        )
    md += [
        "",
        "## Lectura",
        "",
        "- Una accuracy cercana a `1/n_sites` indica que `fused` no permite "
        "decodificar linealmente el sitio: representación site-invariante.",
        "- Una accuracy muy superior al chance indica que la identidad del "
        "sitio persiste en la representación, aunque el clasificador de "
        "diagnóstico no la use directamente.",
        "",
    ]
    md_out = out_dir / "site_invariance_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")

    print("\n" + "=" * 60)
    print("RESUMEN SITE-INVARIANCE (fused)")
    print("=" * 60)
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        s = summary["metrics"][metric]
        print(f"  {metric:<18s}: {fmt(s['mean'])} ± {fmt(s['std'])}")
    print(f"  chance            : {1.0 / n_sites:.4f}")
    print(f"  majority baseline : {majority:.4f}")


if __name__ == "__main__":
    main()
