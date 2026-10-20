# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Análisis descriptivo por sujeto en sitios seleccionados.

No entrena modelos. No recalcula métricas existentes (AUC, accuracy,
sensitivity, specificity). Solo:

  1. Carga preds_fold_N.npz del run original.
  2. Verifica alineamiento: sub_ids ↔ CSV, labels ↔ CSV,
     probs ↔ softmax(logits) (de features_fold_N.npz).
  3. Genera tablas por sujeto (SUB_ID, label, prob, pred@0.5, correcto).
  4. Genera distribuciones resumidas por clase.
  5. Describe los errores: conteos, probs en errores vs aciertos,
     errores cerca del umbral vs lejos.
  6. Referencia los AUC de ramas ya calculados en branch_ablation.

Uso:
    python scripts/analysis/site_subject_analysis.py \
        --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds

Salida:
    <run-dir>/site_subject_analysis.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SITES = ["MAX_MUN", "SBL", "NYU", "USM", "UM_2"]


def softmax_col1(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable sobre axis=1, devuelve la prob de clase 1."""
    logits = logits.astype(np.float64)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def find_fold_for_site(ckpt_dir: Path, site: str,
                        df_csv: pd.DataFrame, n_folds: int = 19):
    """
    En LOSO cada sitio = un fold. Identificar por coincidencia de sub_ids.

    No exige igualdad exacta: el pipeline puede excluir sujetos por QC
    (ROIs constantes, min_timesteps). Se acepta el fold donde el sitio
    mayoritario de sus sujetos sea el buscado.

    Criterio: se acepta si >50% de los sujetos del fold pertenecen al
    sitio Y el sitio mayoritario es exactamente ese sitio.
    """
    sub_to_site = dict(zip(df_csv["SUB_ID"].astype(int),
                            df_csv["SITE_ID"].astype(str)))
    site_sub_ids = set(
        df_csv[df_csv["SITE_ID"] == site]["SUB_ID"].astype(int).tolist()
    )

    for fold_idx in range(n_folds):
        npz_path = ckpt_dir / f"preds_fold_{fold_idx}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        if "test_subject_ids" in npz.files:
            test_ids = [int(x) for x in npz["test_subject_ids"]]
        else:
            test_ids = [int(x) for x in npz["subject_indices"][npz["test_idx"]]]

        if len(test_ids) == 0:
            continue

        # Sitio mayoritario del fold
        sites_here = [sub_to_site.get(sid, "UNKNOWN") for sid in test_ids]
        majority_site = max(set(sites_here), key=sites_here.count)

        if majority_site != site:
            continue

        overlap = len(set(test_ids) & site_sub_ids)
        if overlap > len(test_ids) * 0.5:
            return fold_idx, npz_path

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("data/csv/data_train.csv"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    ckpt_dir = run_dir / "checkpoints"
    feat_dir = run_dir / "features"

    out_path = args.out if args.out else (run_dir / "site_subject_analysis.md")

    df_csv = pd.read_csv(args.csv)
    label_col = "DX_GROUP"
    sub_to_site  = dict(zip(df_csv["SUB_ID"].astype(int), df_csv["SITE_ID"].astype(str)))
    sub_to_label = dict(zip(df_csv["SUB_ID"].astype(int), df_csv[label_col].astype(int)))

    # ── Localizar fold por sitio ─────────────────────────────────────
    site_data = {}
    for site in SITES:
        fold_idx, npz_path = find_fold_for_site(ckpt_dir, site, df_csv)
        if fold_idx is None:
            print(f"[WARN] no se encontró fold para {site}")
            continue
        site_data[site] = (fold_idx, npz_path)

    # ── Cargar AUC de ramas existentes ────────────────────────────────
    branch_csv = run_dir / "branch_ablation" / "branch_ablation_by_site.csv"
    branch_df = pd.read_csv(branch_csv) if branch_csv.exists() else None

    md = []
    md.append("# Análisis por sujeto — TwoTST LOSO (seed 42)")
    md.append("")
    md.append(f"**Run:** `{run_dir}`")
    md.append(f"**Sitios analizados:** {', '.join(SITES)}")
    md.append("")
    md.append("Este informe **no reentrena modelos** ni recalcula métricas "
              "existentes (AUC, accuracy, sensitivity, specificity). "
              "Solo extrae y describe las predicciones ya guardadas.")
    md.append("")
    md.append("---")
    md.append("")

    # ── Verificación de alineamiento ─────────────────────────────────
    md.append("## Verificación de alineamiento")
    md.append("")
    all_ok = True
    for site in SITES:
        if site not in site_data:
            md.append(f"- **{site}**: fold no encontrado ✗")
            all_ok = False
            continue
        fold_idx, npz_path = site_data[site]
        npz = np.load(npz_path, allow_pickle=True)
        labels = npz["labels"]
        probs  = npz["probs"]
        if "test_subject_ids" in npz.files:
            test_ids = npz["test_subject_ids"]
        else:
            test_ids = npz["subject_indices"][npz["test_idx"]]

        # Check 1: labels match CSV
        csv_labels = np.array([sub_to_label[int(s)] for s in test_ids])
        label_ok = bool(np.array_equal(labels, csv_labels))

        # Check 2: probs match softmax(logits)[:,1] si features_fold existe
        feat_path = feat_dir / f"features_fold_{fold_idx}.npz"
        if feat_path.exists():
            feat = np.load(feat_path, allow_pickle=True)
            # Verificar que features y preds tienen los mismos sujetos (mismo orden)
            same_sub_ids = bool(np.array_equal(
                np.array([int(s) for s in feat["sub_ids"]]),
                np.array([int(s) for s in test_ids]),
            ))
            probs_from_logits = softmax_col1(feat["logits"])
            probs_ok = bool(np.allclose(probs, probs_from_logits, atol=1e-5))
            labels_match_feat = bool(np.array_equal(labels, feat["labels"]))
        else:
            same_sub_ids = probs_ok = labels_match_feat = None

        # Check 3: sitio correcto
        sites_here = [sub_to_site[int(s)] for s in test_ids]
        majority_site = max(set(sites_here), key=sites_here.count)
        site_ok = (majority_site == site)
        n_other_sites = sum(1 for s in sites_here if s != site)

        msg = (f"- **{site}** (fold {fold_idx}, n={len(labels)}): "
               f"labels↔CSV: {'✓' if label_ok else '✗'}, "
               f"probs↔softmax(logits): {'✓' if probs_ok else '✗' if probs_ok is not None else 'N/A'}, "
               f"labels↔features: {'✓' if labels_match_feat else '✗' if labels_match_feat is not None else 'N/A'}, "
               f"sub_ids↔features: {'✓' if same_sub_ids else '✗' if same_sub_ids is not None else 'N/A'}, "
               f"sitio mayoritario: {majority_site} {'✓' if site_ok else '✗'}"
               + (f" ({n_other_sites} sujetos de otros sitios)" if n_other_sites > 0 else ""))
        md.append(msg)
        if not label_ok or not site_ok or (probs_ok is False) or (labels_match_feat is False) or (same_sub_ids is False):
            all_ok = False

    md.append("")
    if all_ok:
        md.append("**Resultado:** todas las verificaciones pasaron. "
                  "Las probabilidades corresponden a las predicciones guardadas "
                  "y las etiquetas están correctamente alineadas con los sujetos del CSV.")
    else:
        md.append("**Resultado:** alguna verificación falló. Interpretar con cautela.")
    md.append("")
    md.append("**Nota:** las probabilidades por sujeto de las ramas temporal y de "
              "conectividad no están guardadas en disco. Solo se conservan sus AUC "
              "agregados por sitio en `branch_ablation/branch_ablation_by_site.csv`.")
    md.append("")
    md.append("---")
    md.append("")

    # ── Análisis por sitio ───────────────────────────────────────────
    site_stats = []
    for site in SITES:
        if site not in site_data:
            continue
        fold_idx, npz_path = site_data[site]
        npz = np.load(npz_path, allow_pickle=True)
        labels = npz["labels"]
        probs  = npz["probs"]
        if "test_subject_ids" in npz.files:
            test_ids = npz["test_subject_ids"]
        else:
            test_ids = npz["subject_indices"][npz["test_idx"]]

        n = len(labels)
        n_asd = int((labels == 1).sum())
        n_td  = int((labels == 0).sum())

        md.append(f"## {site} (fold {fold_idx}, n={n}, ASD={n_asd}, TD={n_td})")
        md.append("")

        # Tabla por sujeto (ordenada por prob descendente)
        md.append("### Predicciones por sujeto")
        md.append("")
        md.append("| SUB_ID | Label | Prob(ASD) | Pred @0.5 | Correcto |")
        md.append("|---|---|---|---|---|")
        order = np.argsort(-probs)
        for i in order:
            lbl = "ASD" if labels[i] == 1 else "TD"
            pred = 1 if probs[i] >= 0.5 else 0
            pred_str = "ASD" if pred == 1 else "TD"
            ok = "✓" if pred == labels[i] else "✗"
            md.append(f"| {int(test_ids[i])} | {lbl} | {probs[i]:.4f} | {pred_str} | {ok} |")
        md.append("")

        # Distribuciones resumidas por clase
        md.append("### Distribuciones resumidas por clase")
        md.append("")
        md.append("| Clase | n | mean | std | min | Q1 | median | Q3 | max |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for cls, name in [(1, "ASD"), (0, "TD")]:
            p = probs[labels == cls]
            if len(p) == 0:
                continue
            md.append(f"| {name} | {len(p)} | {p.mean():.4f} | {p.std():.4f} | "
                      f"{p.min():.4f} | {np.percentile(p, 25):.4f} | "
                      f"{np.median(p):.4f} | {np.percentile(p, 75):.4f} | "
                      f"{p.max():.4f} |")
        md.append("")

        # Descripción de errores a 0.5
        preds_05 = (probs >= 0.5).astype(int)
        errors = (preds_05 != labels)
        n_err = int(errors.sum())
        asd_err = int(((labels == 1) & (preds_05 == 0)).sum())
        td_err  = int(((labels == 0) & (preds_05 == 1)).sum())

        md.append("### Descripción de errores (umbral 0.5)")
        md.append("")
        md.append(f"- Total errores: **{n_err}/{n}** ({n_err/n*100:.1f}%)")
        md.append(f"- ASD→TD: **{asd_err}/{n_asd}** "
                  f"({asd_err/n_asd*100:.1f}% de ASD)" if n_asd > 0 else "- ASD→TD: 0/0")
        md.append(f"- TD→ASD: **{td_err}/{n_td}** "
                  f"({td_err/n_td*100:.1f}% de TD)" if n_td > 0 else "- TD→ASD: 0/0")
        md.append("")

        # Distribución de probs en errores vs aciertos
        correct_probs = probs[~errors]
        error_probs = probs[errors]
        md.append("**Distribución de probs en aciertos vs errores:**")
        md.append("")
        if len(correct_probs) > 0:
            md.append(f"- Aciertos (n={len(correct_probs)}): "
                      f"mean={correct_probs.mean():.4f}, "
                      f"std={correct_probs.std():.4f}, "
                      f"rango=[{correct_probs.min():.4f}, {correct_probs.max():.4f}]")
        if len(error_probs) > 0:
            md.append(f"- Errores (n={len(error_probs)}): "
                      f"mean={error_probs.mean():.4f}, "
                      f"std={error_probs.std():.4f}, "
                      f"rango=[{error_probs.min():.4f}, {error_probs.max():.4f}]")
            near = error_probs[(error_probs > 0.3) & (error_probs < 0.7)]
            far  = error_probs[(error_probs <= 0.3) | (error_probs >= 0.7)]
            md.append(f"  - Errores con prob ∈ (0.3, 0.7) [«indecisos»]: {len(near)}")
            md.append(f"  - Errores con prob ≤0.3 o ≥0.7 [«confiados pero equivocados»]: {len(far)}")
        md.append("")

        # Referencia agregada de ramas
        if branch_df is not None:
            row = branch_df[branch_df["site"] == site]
            if len(row) > 0:
                r = row.iloc[0]
                md.append("### AUC por representación (referencia — ya calculado previamente)")
                md.append("")
                md.append("| Representación | AUC |")
                md.append("|---|---|")
                md.append(f"| Temporal (h_ts_proj) | {r['auc_temporal']:.4f} |")
                md.append(f"| Connectivity (h_fc_proj) | {r['auc_connectivity']:.4f} |")
                md.append(f"| Fused frozen | {r['auc_fused_frozen']:.4f} |")
                md.append(f"| Full original | {r['auc_full_original']:.4f} |")
                md.append("")

        # Acumular stats
        site_stats.append({
            "site": site,
            "n": n,
            "n_asd": n_asd,
            "n_td": n_td,
            "asd_mean": float(probs[labels == 1].mean()) if n_asd > 0 else float("nan"),
            "td_mean":  float(probs[labels == 0].mean()) if n_td > 0 else float("nan"),
            "gap":      float(probs[labels == 1].mean() - probs[labels == 0].mean())
                        if n_asd > 0 and n_td > 0 else float("nan"),
            "asd_err":  asd_err,
            "td_err":   td_err,
            "n_err":    n_err,
        })
        md.append("---")
        md.append("")

    # ── Observaciones cruzadas ───────────────────────────────────────
    md.append("## Observaciones cruzadas (resultados observados)")
    md.append("")
    md.append("| Sitio | n | n_asd | n_td | mean prob ASD | mean prob TD | gap (ASD−TD) | err ASD→TD | err TD→ASD |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in site_stats:
        md.append(f"| {s['site']} | {s['n']} | {s['n_asd']} | {s['n_td']} | "
                  f"{s['asd_mean']:.4f} | {s['td_mean']:.4f} | {s['gap']:+.4f} | "
                  f"{s['asd_err']} | {s['td_err']} |")
    md.append("")

    md.append("**Patrones observables (sin interpretación causal):**")
    md.append("")

    # 1. Gap ASD-TD
    max_gap = max(site_stats, key=lambda x: x["gap"])
    min_gap = min(site_stats, key=lambda x: x["gap"])
    md.append(f"- **Gap ASD−TD (diferencia de medias de prob):** "
              f"rango de {min_gap['gap']:+.4f} ({min_gap['site']}) a "
              f"{max_gap['gap']:+.4f} ({max_gap['site']}).")

    # 2. Dirección de errores
    for s in site_stats:
        if s["n_asd"] > 0 and s["n_td"] > 0:
            total_e = s["asd_err"] + s["td_err"]
            if total_e > 0:
                md.append(f"- **{s['site']}**: ASD→TD={s['asd_err']} "
                          f"({s['asd_err']/s['n_asd']*100:.0f}% de ASD), "
                          f"TD→ASD={s['td_err']} "
                          f"({s['td_err']/s['n_td']*100:.0f}% de TD).")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Interpretaciones posibles (compatibles con los datos, no demostradas)")
    md.append("")
    md.append("Las siguientes son lecturas **consistentes** con las tablas anteriores. "
              "Los datos por sí solos no permiten distinguirlas:")
    md.append("")
    md.append("1. **Probabilidades poco separadas entre clases en sitios de bajo rendimiento.** "
              "Un gap ASD−TD pequeño sugiere que el clasificador no desplaza las medias de "
              "probabilidad entre clases. Puede reflejar falta de señal en las representaciones "
              "*o* una frontera de decisión mal ajustada — los datos aquí no distinguen entre ambas.")
    md.append("")
    md.append("2. **Errores concentrados en una clase.** Si la mayoría de errores van en una sola "
              "dirección (por ejemplo ASD→TD), es compatible con un problema de calibración del "
              "umbral (la separación existe pero el corte está desplazado). Si los errores se "
              "reparten entre ambas clases, es más compatible con baja separabilidad real.")
    md.append("")
    md.append("3. **Errores cerca del umbral vs lejos.** Errores con prob ≈ 0.5 indican "
              "«indecisión» del modelo. Errores con prob cerca de 0 o 1 indican «confianza "
              "equivocada» — el modelo está seguro pero la predicción es incorrecta. Este "
              "patrón sugiere que las features no capturan la señal diagnóstica del sujeto, "
              "o que el sujeto es atípico dentro de su clase.")
    md.append("")
    md.append("4. **Rama dominante por sitio.** Si la rama de conectividad tiene AUC alta pero "
              "el modelo completo falla, la limitación estaría en la fusión o el clasificador. "
              "Si la conectividad también tiene AUC baja, la limitación estaría en esa "
              "representación.")
    md.append("")
    md.append("## Lo que estos datos NO permiten afirmar")
    md.append("")
    md.append("- **No** permiten determinar si la causa del bajo rendimiento es falta de "
              "señal diagnóstica en el fMRI de esos sitios **o** mala generalización del "
              "modelo. Distinguir entre ambas requeriría diseños adicionales (etiquetas "
              "alternativas, análisis de subgrupos clínicos, transferencia controlada) que "
              "no forman parte de este análisis.")
    md.append("- **No** permiten afirmar causalidad a partir de las correlaciones observadas. "
              "Las observaciones son conjuntas, no causales.")
    md.append("- **No** permiten extrapolar a otras seeds ni a otros modelos: este informe "
              "describe únicamente la seed 42 del run original TwoTST.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {out_path}")
    print()
    print("Resumen de folds localizados:")
    for site in SITES:
        if site in site_data:
            print(f"  {site}: fold={site_data[site][0]}")
    print()
    print(f"Verificaciones pasadas: {'TODAS' if all_ok else 'ALGUNA FALLÓ'}")


if __name__ == "__main__":
    main()
