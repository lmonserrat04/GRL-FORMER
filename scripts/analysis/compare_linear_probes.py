# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Compara los resultados de linear_probe_perfold para varias representaciones.

Lee los <run>/linear_probe_perfold_<rep>/linear_probe_perfold_global.json
de las representaciones indicadas y produce:

  1. Tabla agregada (media, std, IC, Δ vs baseline, p-valor, Cohen's d).
  2. Tabla por sitio cruzando las 3 representaciones + baseline.
  3. Ranking por AUC medio.

Salidas:
  <out>/compare_linear_probes.csv
  <out>/compare_linear_probes.md

Uso:
    python scripts/analysis/compare_linear_probes.py \\
        --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds \\
        --representations fused h_ts_proj h_fc_proj
"""
import argparse
import json
from pathlib import Path

import pandas as pd


REP_LABELS = {
    "fused": "Fused",
    "h_ts_proj": "Temporal (h_ts_proj)",
    "h_fc_proj": "Connectivity (h_fc_proj)",
    "h_ts_raw": "Temporal raw (h_ts_raw)",
    "h_fc_raw": "Connectivity raw (h_fc_raw)",
}


def load_summary(run_dir: Path, rep: str) -> dict | None:
    path = run_dir / f"linear_probe_perfold_{rep}" / "linear_probe_perfold_global.json"
    if not path.exists():
        print(f"[WARN] no existe {path}")
        return None
    with open(path) as f:
        return json.load(f)


def fmt_p(p):
    if p is None:
        return "—"
    return f"{p:.4f}"


def fmt_num(x, fmt=".4f"):
    if x is None:
        return "—"
    return format(x, fmt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--representations", nargs="+",
                        default=["fused", "h_ts_proj", "h_fc_proj"])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir if args.out_dir else run_dir / "linear_probe_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar las 3 summaries ─────────────────────────────────────
    summaries = {}
    for rep in args.representations:
        s = load_summary(run_dir, rep)
        if s is not None:
            summaries[rep] = s

    if not summaries:
        raise SystemExit("No hay summaries que cargar")

    # Baseline común (seed 42) — se toma del primer summary
    first = next(iter(summaries.values()))
    base_mean = first["baseline_seed42"]["auc_mean"]
    base_std  = first["baseline_seed42"]["auc_std"]

    # ── Tabla 1: agregado por representación ─────────────────────
    print("="*100)
    print("RESUMEN AGREGADO — Linear probe per-fold (LOSO, seed 42)")
    print("="*100)

    header = (f"{'Representación':<26s}  "
              f"{'Probe AUC':<20s}  "
              f"{'IC 95%':<20s}  "
              f"{'Δ vs baseline':<15s}  "
              f"{'p (t)':<8s}  "
              f"{'p (Wilcox)':<10s}  "
              f"{'Cohen d':<8s}")
    print(header)
    print("-" * len(header))

    agg_rows = []
    for rep, s in summaries.items():
        probe = s["probe"]
        pt = s["paired_test"]
        label = REP_LABELS.get(rep, rep)
        line = (f"{label:<26s}  "
                f"{probe['auc_mean']:.4f} ± {probe['auc_std']:.4f}      "
                f"[{probe['ci95_lo']:.4f}, {probe['ci95_hi']:.4f}]   "
                f"{pt['delta_mean']:+.4f} ± {pt['delta_std']:.4f}  "
                f"{fmt_p(pt['p_ttest']):<8s}  "
                f"{fmt_p(pt['p_wilcoxon']):<10s}  "
                f"{fmt_num(pt['cohen_d'], '.3f'):<8s}")
        print(line)
        agg_rows.append({
            "representation": rep,
            "label": label,
            "probe_auc_mean": probe["auc_mean"],
            "probe_auc_std": probe["auc_std"],
            "probe_ci95_lo": probe["ci95_lo"],
            "probe_ci95_hi": probe["ci95_hi"],
            "delta_mean": pt["delta_mean"],
            "delta_std": pt["delta_std"],
            "p_ttest": pt["p_ttest"],
            "p_wilcoxon": pt["p_wilcoxon"],
            "cohen_d": pt["cohen_d"],
            "n_sites": pt["n_sites"],
        })

    print()
    print(f"Baseline (seed 42): AUC = {base_mean:.4f} ± {base_std:.4f}")
    print()

    # ── Tabla 2: por sitio ───────────────────────────────────────
    print("="*120)
    print("DETALLE POR SITIO")
    print("="*120)

    # Construir un DataFrame cruzando fold / site / baseline / probe de cada rep
    all_rows = {}
    for rep, s in summaries.items():
        for r in s["per_site"]:
            fold = int(r["fold"])
            if fold not in all_rows:
                all_rows[fold] = {
                    "fold": fold,
                    "n": int(r["n"]),
                    "n_td": int(r["n_td"]),
                    "n_asd": int(r["n_asd"]),
                    "baseline": r["auc_baseline"],
                }
            all_rows[fold][f"probe_{rep}"] = r["auc_probe"]

    df = pd.DataFrame(sorted(all_rows.values(), key=lambda x: x["fold"]))
    # Reordenar columnas: fold, n, n_td, n_asd, baseline, probe_fused, probe_h_ts_proj, probe_h_fc_proj
    cols = ["fold", "n", "n_td", "n_asd", "baseline"]
    for rep in args.representations:
        if f"probe_{rep}" in df.columns:
            cols.append(f"probe_{rep}")
    df = df[cols]

    # Print
    col_w = 14
    header2 = f"{'Fold':<5s} {'n':>4s} {'TD':>4s} {'ASD':>4s}  {'Baseline':>9s}"
    for rep in args.representations:
        if f"probe_{rep}" in df.columns:
            header2 += f"  {REP_LABELS.get(rep, rep)[:11]:>11s}"
    print(header2)
    print("-" * len(header2))

    for _, r in df.iterrows():
        line = (f"{int(r['fold']):<5d} {int(r['n']):>4d} {int(r['n_td']):>4d} "
                f"{int(r['n_asd']):>4d}  {r['baseline']:>9.4f}")
        for rep in args.representations:
            c = f"probe_{rep}"
            if c in df.columns:
                v = r[c]
                line += f"  {v:>11.4f}" if pd.notna(v) else f"  {'—':>11s}"
        print(line)

    print("-" * len(header2))
    # Media
    line_mean = f"{'MEAN':<5s} {'':>4s} {'':>4s} {'':>4s}  {df['baseline'].mean():>9.4f}"
    for rep in args.representations:
        c = f"probe_{rep}"
        if c in df.columns:
            line_mean += f"  {df[c].mean():>11.4f}"
    print(line_mean)

    print()

    # ── Tabla 3: diferencias por sitio vs baseline ────────────────
    print("="*120)
    print("Δ PROBE vs BASELINE (por sitio)")
    print("="*120)

    header3 = f"{'Fold':<5s}"
    for rep in args.representations:
        if f"probe_{rep}" in df.columns:
            header3 += f"  {REP_LABELS.get(rep, rep)[:11]:>11s}"
    print(header3)
    print("-" * len(header3))

    for _, r in df.iterrows():
        line = f"{int(r['fold']):<5d}"
        for rep in args.representations:
            c = f"probe_{rep}"
            if c in df.columns:
                v = r[c] - r["baseline"]
                line += f"  {v:>+11.4f}" if pd.notna(v) else f"  {'—':>11s}"
        print(line)

    print("-" * len(header3))
    line_mean_d = f"{'MEAN':<5s}"
    for rep in args.representations:
        c = f"probe_{rep}"
        if c in df.columns:
            mean_d = (df[c] - df["baseline"]).mean()
            line_mean_d += f"  {mean_d:>+11.4f}"
    print(line_mean_d)
    print()

    # ── Ranking ───────────────────────────────────────────────────
    print("="*60)
    print("RANKING por probe AUC medio")
    print("="*60)
    ranked = sorted(agg_rows, key=lambda x: -x["probe_auc_mean"])
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['label']:<26s}  {r['probe_auc_mean']:.4f} "
              f"± {r['probe_auc_std']:.4f}")
    print(f"      Baseline (seed 42)              {base_mean:.4f} ± {base_std:.4f}")
    print()

    # ── Guardar CSV ──────────────────────────────────────────────
    df_agg = pd.DataFrame(agg_rows)
    csv_agg = out_dir / "compare_linear_probes_aggregate.csv"
    df_agg.to_csv(csv_agg, index=False)
    print(f"Guardado: {csv_agg}")

    csv_by_site = out_dir / "compare_linear_probes_by_site.csv"
    df.to_csv(csv_by_site, index=False)
    print(f"Guardado: {csv_by_site}")

    # ── Guardar Markdown ────────────────────────────────────────
    md = ["# Comparación de linear probes per-fold", ""]
    md.append(f"**Run:** `{run_dir}`")
    md.append(f"**Representaciones:** {', '.join(args.representations)}")
    md.append("")
    md.append("## Tabla agregada")
    md.append("")
    md.append("| Representación | Probe AUC | IC 95% | Δ vs baseline | p (t) | p (Wilcox) | Cohen d |")
    md.append("|---|---|---|---|---|---|---|")
    for rep, s in summaries.items():
        probe = s["probe"]
        pt = s["paired_test"]
        label = REP_LABELS.get(rep, rep)
        md.append(f"| {label} | {probe['auc_mean']:.4f} ± {probe['auc_std']:.4f} | "
                  f"[{probe['ci95_lo']:.4f}, {probe['ci95_hi']:.4f}] | "
                  f"{pt['delta_mean']:+.4f} ± {pt['delta_std']:.4f} | "
                  f"{fmt_p(pt['p_ttest'])} | {fmt_p(pt['p_wilcoxon'])} | "
                  f"{fmt_num(pt['cohen_d'], '.3f')} |")
    md.append("")
    md.append(f"**Baseline (seed 42):** {base_mean:.4f} ± {base_std:.4f}")
    md.append("")
    md.append("## Detalle por sitio")
    md.append("")
    md_header = "| Fold | n | TD | ASD | Baseline |"
    md_sep = "|---|---|---|---|---|"
    for rep in args.representations:
        if f"probe_{rep}" in df.columns:
            md_header += f" {REP_LABELS.get(rep, rep)[:11]} |"
            md_sep += "---|"
    md.append(md_header)
    md.append(md_sep)
    for _, r in df.iterrows():
        line = f"| {int(r['fold'])} | {int(r['n'])} | {int(r['n_td'])} | {int(r['n_asd'])} | {r['baseline']:.4f} |"
        for rep in args.representations:
            c = f"probe_{rep}"
            if c in df.columns:
                v = r[c]
                line += f" {v:.4f} |" if pd.notna(v) else " — |"
        md.append(line)
    md.append("")
    md.append("## Ranking")
    md.append("")
    for i, r in enumerate(ranked, 1):
        md.append(f"{i}. **{r['label']}**: {r['probe_auc_mean']:.4f} ± {r['probe_auc_std']:.4f}")
    md.append(f"{len(ranked)+1}. **Baseline (seed 42)**: {base_mean:.4f} ± {base_std:.4f}")
    md.append("")

    md_out = out_dir / "compare_linear_probes.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
