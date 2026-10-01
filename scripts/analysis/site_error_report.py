"""
Análisis de errores por sitio a partir de los preds_fold_*.npz.

Uso:
    python scripts/analysis/site_error_report.py <exp_dir>

<exp_dir> es el directorio del experimento (contiene checkpoints/preds_fold_*.npz).

Salida:
    <exp_dir>/site_analysis.json          resumen por sitio
    <exp_dir>/misclassified_subjects.csv  lista de sujetos mal clasificados
    <exp_dir>/site_error_summary.csv      tabla compacta por sitio
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _site_for_subject(sub_id: int, df_sites: dict) -> str:
    return df_sites.get(int(sub_id), "UNKNOWN")


def main(exp_dir: Path, csv_path: Path, threshold_mode: str = "both"):
    """
    threshold_mode: "0.5" | "youden" | "both"
    """
    ckpt_dir = exp_dir / "checkpoints"
    npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
    if not npz_files:
        raise SystemExit(f"No hay preds_fold_*.npz en {ckpt_dir}")

    # Cargar CSV para mapear SUB_ID -> SITE_ID
    df = pd.read_csv(csv_path)
    df_sites = dict(zip(df["SUB_ID"].astype(int), df["SITE_ID"].astype(str)))

    # Acumular predicciones de todos los folds, agregando a nivel de sujeto
    # Cuando hay sliding window (más ventanas que sujetos), se agrega con
    # majority_vote. Cuando no (n_ventanas == n_sujetos), es 1:1.
    subject_records = {}   # sub_id -> dict(label, prob, pred_05, pred_youden, site, fold)

    for f in npz_files:
        npz = np.load(f, allow_pickle=True)
        labels   = npz["labels"]
        probs    = npz["probs"]
        preds    = npz["preds"]
        thr      = float(npz["optimal_thr"][0])

        # Soporta .npz con y sin test_subject_ids (compatibilidad entre ramas)
        if "test_subject_ids" in npz.files:
            sub_ids = npz["test_subject_ids"]
        else:
            test_idx = npz["test_idx"]
            subject_indices = npz["subject_indices"]
            sub_ids = subject_indices[test_idx]

        unique_subs = np.unique(sub_ids)
        for s in unique_subs:
            mask = sub_ids == s
            n_windows = int(mask.sum())

            y_true = int(labels[mask][0])           # misma etiqueta en todas las ventanas
            p_mean = float(probs[mask].mean())

            # Predicción con umbral 0.5
            if n_windows > 1:
                p05 = int(round(preds[mask].mean()))
            else:
                p05 = int(preds[mask][0])

            # Predicción con umbral de Youden
            p_youden = int(p_mean >= thr)

            subject_records[int(s)] = {
                "sub_id":    int(s),
                "site":      _site_for_subject(s, df_sites),
                "label":     y_true,
                "prob":      p_mean,
                "pred_05":   p05,
                "pred_youden": p_youden,
                "thr_youden": thr,
                "n_windows": n_windows,
                "fold":      f.stem,
            }

    # ─── Resumen por sitio ────────────────────────────────────────────
    by_site = defaultdict(list)
    for rec in subject_records.values():
        by_site[rec["site"]].append(rec)

    site_summary = []
    for site, recs in sorted(by_site.items()):
        n = len(recs)
        n_correct_05 = sum(1 for r in recs if r["pred_05"] == r["label"])
        n_correct_yj = sum(1 for r in recs if r["pred_youden"] == r["label"])

        # Desglose por clase
        asd_recs = [r for r in recs if r["label"] == 1]
        td_recs  = [r for r in recs if r["label"] == 0]

        # @0.5
        asd_ok_05 = sum(1 for r in asd_recs if r["pred_05"] == 1)
        asd_err_05 = len(asd_recs) - asd_ok_05    # ASD predichos como TD
        td_ok_05  = sum(1 for r in td_recs  if r["pred_05"] == 0)
        td_err_05 = len(td_recs) - td_ok_05        # TD predichos como ASD

        # @Youden
        asd_ok_yj = sum(1 for r in asd_recs if r["pred_youden"] == 1)
        asd_err_yj = len(asd_recs) - asd_ok_yj
        td_ok_yj  = sum(1 for r in td_recs  if r["pred_youden"] == 0)
        td_err_yj = len(td_recs) - td_ok_yj

        site_summary.append({
            "site":          site,
            "n_subjects":    n,
            "n_asd":         len(asd_recs),
            "n_td":          len(td_recs),

            # @0.5
            "asd_ok_05":     asd_ok_05,
            "asd_err_05":    asd_err_05,
            "td_ok_05":      td_ok_05,
            "td_err_05":     td_err_05,
            "asd_err_rate_05": round(asd_err_05 / len(asd_recs), 4) if asd_recs else 0.0,
            "td_err_rate_05":  round(td_err_05 / len(td_recs), 4) if td_recs else 0.0,
            "sens_05":       round(asd_ok_05 / len(asd_recs), 4) if asd_recs else 0.0,
            "spec_05":       round(td_ok_05 / len(td_recs), 4) if td_recs else 0.0,
            "correct_05":    n_correct_05,
            "error_05":      round(1 - n_correct_05 / n, 4),

            # @Youden
            "asd_ok_yj":     asd_ok_yj,
            "asd_err_yj":    asd_err_yj,
            "td_ok_yj":      td_ok_yj,
            "td_err_yj":     td_err_yj,
            "asd_err_rate_yj": round(asd_err_yj / len(asd_recs), 4) if asd_recs else 0.0,
            "td_err_rate_yj":  round(td_err_yj / len(td_recs), 4) if td_recs else 0.0,
            "sens_yj":       round(asd_ok_yj / len(asd_recs), 4) if asd_recs else 0.0,
            "spec_yj":       round(td_ok_yj / len(td_recs), 4) if td_recs else 0.0,
            "correct_youden": n_correct_yj,
            "error_youden":  round(1 - n_correct_yj / n, 4),
        })

    # ─── Sujetos mal clasificados ─────────────────────────────────────
    misclassified = [
        {
            "sub_id":   r["sub_id"],
            "site":     r["site"],
            "label":    r["label"],
            "prob":     round(r["prob"], 4),
            "pred_05":  r["pred_05"],
            "pred_youden": r["pred_youden"],
            "error_05": (r["pred_05"] != r["label"]),
            "error_youden": (r["pred_youden"] != r["label"]),
            "fold":     r["fold"],
        }
        for r in subject_records.values()
    ]

    # ─── Global ───────────────────────────────────────────────────────
    total = len(subject_records)
    correct_05 = sum(1 for r in subject_records.values() if r["pred_05"] == r["label"])
    correct_yj = sum(1 for r in subject_records.values() if r["pred_youden"] == r["label"])

    global_summary = {
        "experiment":       str(exp_dir),
        "n_subjects":       total,
        "n_asd":            sum(1 for r in subject_records.values() if r["label"] == 1),
        "n_td":             sum(1 for r in subject_records.values() if r["label"] == 0),
        "global_accuracy_05":      round(correct_05 / total, 4),
        "global_accuracy_youden":  round(correct_yj / total, 4),
        "global_error_05":         round(1 - correct_05 / total, 4),
        "global_error_youden":     round(1 - correct_yj / total, 4),
    }

    # ─── Guardar ──────────────────────────────────────────────────────
    out_json = exp_dir / "site_analysis.json"
    out_mis  = exp_dir / "misclassified_subjects.csv"
    out_sum  = exp_dir / "site_error_summary.csv"

    with open(out_json, "w") as f:
        json.dump({
            "global": global_summary,
            "per_site": site_summary,
        }, f, indent=2)

    pd.DataFrame(misclassified).to_csv(out_mis, index=False)
    pd.DataFrame(site_summary).to_csv(out_sum, index=False)

    # ─── Imprimir resumen ─────────────────────────────────────────────
    print(f"=== Resumen global ===")
    print(f"  experiment:      {exp_dir}")
    print(f"  n_subjects:      {total}  (ASD={global_summary['n_asd']}, TD={global_summary['n_td']})")
    print(f"  accuracy @0.5:   {global_summary['global_accuracy_05']:.4f}")
    print(f"  accuracy @Youden:{global_summary['global_accuracy_youden']:.4f}")
    print()
    print(f"=== Por sitio (@0.5) ===")
    print(f"  ASD = Autism, TD = Typical Development")
    print(f"  asd_ok = ASD bien clasificados, asd_err = ASD predichos como TD")
    print(f"  td_ok  = TD bien clasificados,  td_err  = TD predichos como ASD")
    print()
    header = (f"{'site':<12s} {'n':>4s}  "
              f"{'ASD':>4s} {'ok':>4s} {'err':>4s}  "
              f"{'TD':>4s} {'ok':>4s} {'err':>4s}  "
              f"{'err_ASD':>8s} {'err_TD':>7s}  "
              f"{'err_tot':>8s}")
    print(header)
    print("-" * len(header))
    for s in sorted(site_summary, key=lambda x: x["site"]):
        print(f"{s['site']:<12s} {s['n_subjects']:>4d}  "
              f"{s['n_asd']:>4d} {s['asd_ok_05']:>4d} {s['asd_err_05']:>4d}  "
              f"{s['n_td']:>4d} {s['td_ok_05']:>4d} {s['td_err_05']:>4d}  "
              f"{s['asd_err_rate_05']:>8.3f} {s['td_err_rate_05']:>7.3f}  "
              f"{s['error_05']:>8.3f}")
    print()
    print(f"=== Por sitio (@Youden) ===")
    header_yj = (f"{'site':<12s} {'n':>4s}  "
                 f"{'ASD':>4s} {'ok':>4s} {'err':>4s}  "
                 f"{'TD':>4s} {'ok':>4s} {'err':>4s}  "
                 f"{'err_ASD':>8s} {'err_TD':>7s}  "
                 f"{'err_tot':>8s}")
    print(header_yj)
    print("-" * len(header_yj))
    for s in sorted(site_summary, key=lambda x: x["site"]):
        print(f"{s['site']:<12s} {s['n_subjects']:>4d}  "
              f"{s['n_asd']:>4d} {s['asd_ok_yj']:>4d} {s['asd_err_yj']:>4d}  "
              f"{s['n_td']:>4d} {s['td_ok_yj']:>4d} {s['td_err_yj']:>4d}  "
              f"{s['asd_err_rate_yj']:>8.3f} {s['td_err_rate_yj']:>7.3f}  "
              f"{s['error_youden']:>8.3f}")
    print()
    print(f"Guardado:")
    print(f"  {out_json}")
    print(f"  {out_mis}")
    print(f"  {out_sum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de errores por sitio")
    parser.add_argument("exp_dir", type=Path, help="Directorio del experimento")
    parser.add_argument("--csv", type=Path, default=Path("data/csv/data_train.csv"),
                        help="CSV con SUB_ID y SITE_ID")
    parser.add_argument("--threshold", choices=["0.5", "youden", "both"], default="both")
    args = parser.parse_args()

    main(args.exp_dir, args.csv, args.threshold)
