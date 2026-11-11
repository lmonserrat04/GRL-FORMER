# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Evaluación del clasificador fused por sitio: AUC + calibración.

Protocolo LOSO:
  - Cada ``preds_fold_N.npz`` contiene las predicciones del sitio held-out N.
  - Se calcula AUC, ECE (10 bins), Brier y NLL por sitio.
  - La media global es macro (cada sitio pesa lo mismo), que es la
    "media por sitio con base n normal" pedida para comparar con 0.7030.
  - IC 95% bootstrap remuestreando sitios (no sujetos).

Uso:
    python scripts/analysis/fused_calibration_report.py \
        --run-dir experiments_finetune_only/<run> \
        --baseline-auc 0.7030

Salidas (por defecto en <run-dir>/calibration/):
    fused_calibration_by_site.csv
    fused_calibration_global.json
    fused_calibration_report.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from utils.metrics import brier_score, expected_calibration_error


def _subject_level(npz: dict) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (labels, probs) a nivel de sujeto.

    Si el npz ya está agregado (un registro por sujeto), se usa tal cual.
    Si hay varias ventanas por sujeto, se promedia la probabilidad.
    """
    labels = np.asarray(npz["labels"]).reshape(-1)
    probs = np.asarray(npz["probs"], dtype=float).reshape(-1)

    if "test_subject_ids" in npz:
        sub_ids = np.asarray(npz["test_subject_ids"]).reshape(-1)
    elif "subject_indices" in npz and "test_idx" in npz:
        sub_ids = np.asarray(npz["subject_indices"])[np.asarray(npz["test_idx"])]
    else:
        return labels, probs

    unique = np.unique(sub_ids)
    if len(unique) == len(sub_ids):
        return labels, probs

    y_out, p_out = [], []
    for s in unique:
        mask = sub_ids == s
        y_out.append(int(labels[mask][0]))
        p_out.append(float(probs[mask].mean()))
    return np.asarray(y_out), np.asarray(p_out)


def _metrics(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict:
    out = {
        "n": int(len(y)),
        "n_td": int((y == 0).sum()),
        "n_asd": int((y == 1).sum()),
        "auc": None,
        "ece": None,
        "brier": None,
        "nll": None,
        "accuracy": None,
    }
    if len(np.unique(y)) < 2:
        return out

    p = np.clip(p, 1e-7, 1 - 1e-7)
    out["auc"] = float(roc_auc_score(y, p))
    out["ece"] = float(expected_calibration_error(y, p, n_bins=n_bins))
    out["brier"] = float(brier_score(y, p))
    out["nll"] = float(log_loss(y, p, labels=[0, 1]))
    out["accuracy"] = float(((p >= 0.5).astype(int) == y).mean())
    return out


def _bootstrap_macro(values: list[float], n_bootstrap: int, seed: int):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return None, None, None
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size >= 2 else 0.0
    if n_bootstrap <= 0 or values.size < 2:
        return mean, std, (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    boots = values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, std, (float(lo), float(hi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-auc", type=float, default=0.7030)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <run-dir>/calibration/")
    args = parser.parse_args()

    ckpt_dir = args.run_dir / "checkpoints"
    files = sorted(
        ckpt_dir.glob("preds_fold_*.npz"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        raise SystemExit(f"No hay preds_fold_*.npz en {ckpt_dir}")

    out_dir = args.out_dir if args.out_dir else args.run_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_y, all_p = [], []
    for f in files:
        fold = int(f.stem.split("_")[-1])
        npz = np.load(f, allow_pickle=True)
        y, p = _subject_level({k: npz[k] for k in npz.files})
        m = _metrics(y, p, args.n_bins)
        m["fold"] = fold
        rows.append(m)
        all_y.append(y)
        all_p.append(p)

    df = pd.DataFrame(rows)[
        ["fold", "n", "n_td", "n_asd", "auc", "ece", "brier", "nll", "accuracy"]
    ].sort_values("fold").reset_index(drop=True)
    pooled = _metrics(
        np.concatenate(all_y), np.concatenate(all_p), args.n_bins
    )

    csv_out = out_dir / "fused_calibration_by_site.csv"
    df.to_csv(csv_out, index=False)
    print(f"Guardado: {csv_out}")

    summary = {}
    for metric in ("auc", "ece", "brier", "nll", "accuracy"):
        values = df[metric].tolist()
        mean, std, (lo, hi) = _bootstrap_macro(
            values, args.n_bootstrap, args.seed
        )
        summary[metric] = {
            "mean": mean,
            "std": std,
            "ci95_lower": lo,
            "ci95_upper": hi,
            "n_sites": int(df[metric].notna().sum()),
        }

    delta = None
    if summary["auc"]["mean"] is not None:
        delta = float(summary["auc"]["mean"] - args.baseline_auc)

    payload = {
        "run_dir": str(args.run_dir),
        "protocol": "loso",
        "aggregation": "macro (equal weight per site)",
        "n_bins_ece": int(args.n_bins),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "baseline_auc": float(args.baseline_auc),
        "delta_vs_baseline": delta,
        "metrics": summary,
        "pooled_metrics": pooled,
        "per_site": df.to_dict(orient="records"),
    }
    json_out = out_dir / "fused_calibration_global.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Guardado: {json_out}")

    def fmt(v):
        return "—" if v is None else f"{v:.4f}"

    md = [
        "# Evaluación fused por sitio — AUC y calibración",
        "",
        f"**Run:** `{args.run_dir}`",
        "**Protocolo:** LOSO (19 sitios)",
        "**Agregación global:** media macro por sitio (cada sitio pesa igual)",
        f"**ECE:** bins de igual ancho, {args.n_bins} bins",
        "",
        "## Resumen global",
        "",
        "| Métrica | Media | std | IC 95% | Sitios |",
        "|---|---|---|---|---|",
    ]
    for metric in ("auc", "ece", "brier", "nll", "accuracy"):
        s = summary[metric]
        ci = (f"[{fmt(s['ci95_lower'])}, {fmt(s['ci95_upper'])}]"
              if s["ci95_lower"] is not None else "—")
        md.append(
            f"| {metric} | {fmt(s['mean'])} | {fmt(s['std'])} | {ci} | "
            f"{s['n_sites']} |"
        )

    md += [
        "",
        f"**AUC baseline de referencia:** {args.baseline_auc:.4f}",
        f"**Δ AUC vs baseline:** {fmt(delta)}",
        "",
        "## Métricas pooled (todos los sujetos, referencia)",
        "",
        "| AUC | ECE | Brier | NLL | ACC | n |",
        "|---|---|---|---|---|---|",
        f"| {fmt(pooled['auc'])} | {fmt(pooled['ece'])} | "
        f"{fmt(pooled['brier'])} | {fmt(pooled['nll'])} | "
        f"{fmt(pooled['accuracy'])} | {pooled['n']} |",
        "",
        "## Resultados por sitio",
        "",
        "| Fold | n | TD | ASD | AUC | ECE | Brier | NLL | ACC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {int(r['fold'])} | {int(r['n'])} | {int(r['n_td'])} | "
            f"{int(r['n_asd'])} | {fmt(r['auc'])} | {fmt(r['ece'])} | "
            f"{fmt(r['brier'])} | {fmt(r['nll'])} | {fmt(r['accuracy'])} |"
        )
    md += [
        "",
        "## Lectura",
        "",
        "- **AUC macro**: promedio simple de las AUC de los 19 sitios, para no "
        "dejar que los sitios grandes dominen la comparación.",
        "- **ECE**: diferencia media entre confianza y accuracy. Más bajo es "
        "mejor; un modelo perfectamente calibrado da 0.",
        "- **Brier/NLL**: métricas propias (proper scoring rules) que combinan "
        "discriminación y calibración.",
        "",
    ]
    md_out = out_dir / "fused_calibration_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")

    print("\n" + "=" * 60)
    print("RESUMEN FUSED (macro por sitio)")
    print("=" * 60)
    for metric in ("auc", "ece", "brier", "nll", "accuracy"):
        s = summary[metric]
        print(f"  {metric:<8s}: {fmt(s['mean'])} ± {fmt(s['std'])}")
    print(f"  AUC pooled: {fmt(pooled['auc'])}")
    print(f"  Δ AUC vs baseline {args.baseline_auc:.4f}: {fmt(delta)}")


if __name__ == "__main__":
    main()
