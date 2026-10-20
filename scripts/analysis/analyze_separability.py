# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Analiza la separabilidad ASD/TD por sitio en las representaciones internas.

Para cada sitio y cada nivel de representación (h_ts_raw, h_ts_proj,
h_fc_raw, h_fc_proj, fused), calcula:

  - Fisher ratio         (distancia de centroides / varianza intra)
  - Silhouette score     (separación geométrica)
  - LOO-KNN AUC          (separabilidad con clasificador trivial)
  - AUC del clasificador real (probs guardadas)

Objetivo:
  - Si silhouette es alta pero AUC real es baja → el clasificador
    DESAPROVECHA la separación existente (problema de calibración).
  - Si silhouette es baja → las features no contienen señal
    (problema de representación).

Uso:
    python scripts/analysis/analyze_separability.py \
        --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds

Salida:
    separability_by_site.csv    tabla por sitio × representación
    separability_by_site.json   resumen estructurado
    separability_report.md      informe
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict


REPRESENTATIONS = ["h_ts_raw", "h_ts_proj", "h_fc_raw", "h_fc_proj", "fused"]


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Fisher ratio (d'^2) entre dos clases."""
    X0 = X[y == 0]
    X1 = X[y == 1]
    if len(X0) < 1 or len(X1) < 1:
        return float("nan")
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    # varianza pooled
    var0 = X0.var(axis=0, ddof=1) if len(X0) > 1 else np.zeros_like(mu0)
    var1 = X1.var(axis=0, ddof=1) if len(X1) > 1 else np.zeros_like(mu1)
    var_pool = (var0 * (len(X0) - 1) + var1 * (len(X1) - 1)) / max(len(X0) + len(X1) - 2, 1)
    var_pool = var_pool + 1e-8
    d_squared = ((mu1 - mu0) ** 2 / var_pool).mean()
    return float(d_squared)


def safe_silhouette(X: np.ndarray, y: np.ndarray) -> float:
    """Silhouette score, con guardas para n pequeño o clase única."""
    if len(np.unique(y)) < 2:
        return float("nan")
    n = len(y)
    # silhouette necesita al menos 2 muestras por clase
    if (y == 0).sum() < 2 or (y == 1).sum() < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, y, metric="euclidean"))
    except Exception:
        return float("nan")


def loo_knn_auc(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """AUC de LOO-KNN con k vecinos (k adaptado a n)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    n = len(y)
    if n < 4:
        return float("nan")
    k_eff = min(k, max(1, min((y == 0).sum(), (y == 1).sum()) - 1))
    if k_eff < 1:
        return float("nan")
    try:
        clf = KNeighborsClassifier(n_neighbors=k_eff)
        loo = LeaveOneOut()
        proba = cross_val_predict(clf, X, y, cv=loo, method="predict_proba")
        return float(roc_auc_score(y, proba[:, 1]))
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run con subcarpeta features/")
    parser.add_argument("--csv", type=Path,
                        default=Path("data/csv/data_train.csv"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    feat_dir = run_dir / "features"
    if not feat_dir.exists():
        raise SystemExit(f"No existe {feat_dir}")

    out_dir = args.out_dir if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV mapping
    df_csv = pd.read_csv(args.csv)
    sub_to_site = dict(zip(df_csv["SUB_ID"].astype(int),
                            df_csv["SITE_ID"].astype(str)))

    # Cargar features de todos los folds
    feat_files = sorted(feat_dir.glob("features_fold_*.npz"))
    if not feat_files:
        raise SystemExit(f"No hay features_fold_*.npz en {feat_dir}")

    print(f"Cargando {len(feat_files)} folds...\n")

    rows = []

    for ff in feat_files:
        d = np.load(ff, allow_pickle=True)
        fold = ff.stem.split("_")[-1]

        labels = d["labels"]
        sub_ids = d["sub_ids"]
        probs = d["probs"]

        # sitio del fold: el SITE_ID más común entre los sujetos
        sites = [sub_to_site.get(int(s), "UNKNOWN") for s in sub_ids]
        site = max(set(sites), key=sites.count) if sites else "UNKNOWN"

        # AUC real del clasificador
        if len(np.unique(labels)) > 1:
            auc_real = float(roc_auc_score(labels, probs))
        else:
            auc_real = float("nan")

        n = len(labels)
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())

        for rep in REPRESENTATIONS:
            X = d[rep]
            y = labels

            fisher = fisher_ratio(X, y)
            silh   = safe_silhouette(X, y)
            knn    = loo_knn_auc(X, y, k=5)

            rows.append({
                "fold": fold,
                "site": site,
                "n": n,
                "n_td": n0,
                "n_asd": n1,
                "representation": rep,
                "fisher": fisher,
                "silhouette": silh,
                "knn_loo_auc": knn,
                "auc_real": auc_real,
            })

    df = pd.DataFrame(rows)

    # ─── Guardar CSV ─────────────────────────────────────────────────
    csv_out = out_dir / "separability_by_site.csv"
    df.to_csv(csv_out, index=False)
    print(f"Guardado: {csv_out}")

    # ─── Print tabla ─────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("Separabilidad por sitio y representación")
    print("=" * 120)

    # Pivot por representación (una fila por sitio+rep)
    for rep in REPRESENTATIONS:
        sub = df[df["representation"] == rep].sort_values("auc_real", ascending=False)
        print(f"\n── {rep} ──")
        print(f"{'site':<12s} {'n':>4s} {'n_td':>5s} {'n_asd':>6s}  "
              f"{'fisher':>8s} {'silh':>7s} {'knn_auc':>9s} {'auc_real':>9s}")
        print("-" * 80)
        for _, r in sub.iterrows():
            fisher_s = f"{r['fisher']:>8.2f}" if not pd.isna(r['fisher']) else f"{'—':>8s}"
            silh_s = f"{r['silhouette']:>7.3f}" if not pd.isna(r['silhouette']) else f"{'—':>7s}"
            knn_s  = f"{r['knn_loo_auc']:>9.3f}" if not pd.isna(r['knn_loo_auc']) else f"{'—':>9s}"
            auc_s  = f"{r['auc_real']:>9.3f}" if not pd.isna(r['auc_real']) else f"{'—':>9s}"
            print(f"{r['site']:<12s} {int(r['n']):>4d} {int(r['n_td']):>5d} {int(r['n_asd']):>6d}  "
                  f"{fisher_s} {silh_s} {knn_s} {auc_s}")

    # ─── Resumen por representación ─────────────────────────────────
    print("\n" + "=" * 80)
    print("Resumen por representación (media entre sitios válidos)")
    print("=" * 80)
    print(f"{'rep':<14s} {'fisher':>8s} {'silh':>8s} {'knn_auc':>9s} {'auc_real':>10s}")
    print("-" * 60)
    for rep in REPRESENTATIONS:
        sub = df[df["representation"] == rep]
        m_f  = sub["fisher"].mean()
        m_s  = sub["silhouette"].mean()
        m_k  = sub["knn_loo_auc"].mean()
        m_a  = sub["auc_real"].mean()
        print(f"{rep:<14s} {m_f:>8.2f} {m_s:>8.3f} {m_k:>9.3f} {m_a:>10.3f}")

    # ─── Guardar JSON ───────────────────────────────────────────────
    json_out = out_dir / "separability_by_site.json"
    with open(json_out, "w") as f:
        json.dump({
            "run_dir": str(run_dir),
            "n_folds": len(feat_files),
            "per_site": df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nGuardado: {json_out}")

    # ─── Informe Markdown ──────────────────────────────────────────
    md = ["# Análisis de separabilidad por sitio", ""]
    md.append(f"**Run:** `{run_dir}`")
    md.append(f"**Folds:** {len(feat_files)}")
    md.append("")
    md.append("## Resumen por representación (media entre sitios)")
    md.append("")
    md.append("| Representación | Fisher | Silhouette | KNN-LOO AUC | AUC real |")
    md.append("|---|---|---|---|---|")
    for rep in REPRESENTATIONS:
        sub = df[df["representation"] == rep]
        md.append(f"| {rep} | {sub['fisher'].mean():.2f} | "
                  f"{sub['silhouette'].mean():.3f} | "
                  f"{sub['knn_loo_auc'].mean():.3f} | "
                  f"{sub['auc_real'].mean():.3f} |")
    md.append("")
    md.append("## Detalle por sitio (fused)")
    md.append("")
    sub = df[df["representation"] == "fused"].sort_values("auc_real", ascending=False)
    md.append("| Sitio | n | n_TD | n_ASD | Fisher | Silhouette | KNN-LOO AUC | AUC real |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        md.append(f"| {r['site']} | {int(r['n'])} | {int(r['n_td'])} | {int(r['n_asd'])} | "
                  f"{r['fisher']:.2f} | {r['silhouette']:.3f} | "
                  f"{r['knn_loo_auc']:.3f} | {r['auc_real']:.3f} |")
    md.append("")
    md.append("## Interpretación")
    md.append("")
    md.append("- **Fisher**: separación de centroides relativa a la varianza intra-clase.")
    md.append("- **Silhouette**: separación geométrica (rango -1 a 1; >0.1 indica cierta separación).")
    md.append("- **KNN-LOO AUC**: AUC de un 5-NN leave-one-out. Mide si la clase es predecible por geometría local.")
    md.append("- **AUC real**: AUC del clasificador del modelo (tag_classifier).")
    md.append("")
    md.append("### Diagnóstico")
    md.append("")
    md.append("- Si **KNN-LOO AUC es alto pero AUC real es bajo** → "
              "las features separan, pero el clasificador no aprovecha.")
    md.append("- Si **KNN-LOO AUC es bajo** → las features no contienen señal separable.")
    md.append("- Si **silhouette es alto pero KNN-LOO AUC es bajo** → separación frágil.")

    md_out = out_dir / "separability_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
