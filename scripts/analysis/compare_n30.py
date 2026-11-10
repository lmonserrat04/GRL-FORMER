# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Compara AUC por sitio con filtro n>=30 entre múltiples experimentos.

Cada experimento se representa por una lista de runs (una por seed).
Para cada experimento:
  - Carga per-site AUC de cada seed (desde preds_fold_N.npz)
  - Calcula AUC mean ± std entre seeds por sitio
  - Filtra sitios con n>=30
  - Reporta AUC media entre sitios filtrados

Uso:
    python scripts/analysis/compare_n30.py \\
        --baseline-seeds <run_s42> <run_s123> ... \\
        --focal-seeds <run_s42> <run_s123> ... \\
        [--weighted-seeds ...] \\
        [--other <name> <run1> <run2> ...] \\
        --csv data/csv/data_train.csv \\
        --out <path.md>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def per_site_auc(run_dir: Path, sub_to_site: dict):
    """Carga AUC por sitio de un run."""
    ckpt_dir = run_dir / "checkpoints"
    npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
    if not npz_files:
        return None
    out = {}
    for f in npz_files:
        npz = np.load(f, allow_pickle=True)
        y = npz["labels"]
        p = npz["probs"]
        if "test_subject_ids" in npz.files:
            sub_ids = npz["test_subject_ids"]
        else:
            sub_ids = npz["subject_indices"][npz["test_idx"]]
        sites = [sub_to_site.get(int(s), "UNKNOWN") for s in sub_ids]
        site = max(set(sites), key=sites.count) if sites else "UNKNOWN"
        if len(np.unique(y)) < 2:
            continue
        out[site] = {"n": int(len(y)), "auc": float(roc_auc_score(y, p))}
    return out


def aggregate_experiment(run_dirs: list, sub_to_site: dict):
    """Para cada sitio: AUC mean ± std entre seeds."""
    per_seed = []
    for r in run_dirs:
        d = per_site_auc(r, sub_to_site)
        if d is not None:
            per_seed.append(d)

    if not per_seed:
        return None

    sites = set()
    for d in per_seed:
        sites |= set(d.keys())

    rows = []
    for site in sorted(sites):
        aucs = [d[site]["auc"] for d in per_seed if site in d]
        ns = [d[site]["n"] for d in per_seed if site in d]
        if not aucs:
            continue
        rows.append({
            "site": site,
            "n": int(np.median(ns)),
            "n_seeds": len(aucs),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def report(df, name, n_min, filter_n=False):
    if df is None:
        return None, None
    d = df.copy()
    if filter_n:
        d = d[d["n"] >= n_min]
    if len(d) == 0:
        return None, None
    return {
        "name": name,
        "n_sites": int(len(d)),
        "auc_mean": float(d["auc_mean"].mean()),
        "auc_std_between_sites": float(d["auc_mean"].std(ddof=1)) if len(d) > 1 else 0.0,
        "auc_mean_of_stds": float(d["auc_std"].mean()),
        "sites": d[["site", "n", "auc_mean", "auc_std"]].to_dict(orient="records"),
    }, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seeds", nargs="+", type=Path)
    parser.add_argument("--focal-seeds",    nargs="+", type=Path)
    parser.add_argument("--weighted-seeds", nargs="+", type=Path, default=None)
    parser.add_argument("--local-k8-seeds",  nargs="+", type=Path, default=None)
    parser.add_argument("--local-k16-seeds", nargs="+", type=Path, default=None)
    parser.add_argument("--extras-file", type=Path, default=None,
                        help="Archivo con formato: 'Nombre | path1 [path2 ...]' por línea")
    parser.add_argument("--csv", type=Path, default=Path("data/csv/data_train.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df_csv = pd.read_csv(args.csv)
    sub_to_site = dict(zip(df_csv["SUB_ID"].astype(int), df_csv["SITE_ID"].astype(str)))

    experiments = {}
    if args.baseline_seeds:
        experiments["Baseline (CE)"] = aggregate_experiment(args.baseline_seeds, sub_to_site)
    if args.focal_seeds:
        experiments["Focal loss"] = aggregate_experiment(args.focal_seeds, sub_to_site)
    if args.weighted_seeds:
        experiments["Weighted site"] = aggregate_experiment(args.weighted_seeds, sub_to_site)
    if args.local_k8_seeds:
        experiments["Local attn k=8"] = aggregate_experiment(args.local_k8_seeds, sub_to_site)
    if args.local_k16_seeds:
        experiments["Local attn k=16"] = aggregate_experiment(args.local_k16_seeds, sub_to_site)

    # Extras desde archivo
    if args.extras_file is not None:
        if not args.extras_file.exists():
            print(f"[WARN] extras-file no existe: {args.extras_file}")
        else:
            for line in args.extras_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" not in line:
                    print(f"[WARN] línea sin '|': {line}")
                    continue
                name, paths_str = line.split("|", 1)
                name = name.strip()
                paths = [Path(p.strip()) for p in paths_str.split() if p.strip()]
                if not paths:
                    continue
                print(f"[extras] {name}: {len(paths)} run(s)")
                experiments[name] = aggregate_experiment(paths, sub_to_site)

    # Build comparison tables
    results = {"all_sites": [], "n30": []}
    dfs = {}
    for name, df in experiments.items():
        r_all, _ = report(df, name, 30, filter_n=False)
        r_30, d30 = report(df, name, 30, filter_n=True)
        if r_all:
            results["all_sites"].append(r_all)
        if r_30:
            results["n30"].append(r_30)
            dfs[name] = d30

    # Print
    print("="*90)
    print("COMPARACIÓN n>=30 (media entre sitios con n>=30)")
    print("="*90)
    print(f"{'Experimento':<20s}  {'N sitios':>8s}  {'AUC mean':>10s}  "
          f"{'std entre sitios':>16s}  {'mean std entre seeds':>20s}")
    print("-"*90)
    for r in results["n30"]:
        print(f"{r['name']:<20s}  {r['n_sites']:>8d}  {r['auc_mean']:>10.4f}  "
              f"{r['auc_std_between_sites']:>16.4f}  {r['auc_mean_of_stds']:>20.4f}")

    print()
    print("="*90)
    print("COMPARACIÓN sin filtro (todos los sitios)")
    print("="*90)
    print(f"{'Experimento':<20s}  {'N sitios':>8s}  {'AUC mean':>10s}")
    print("-"*50)
    for r in results["all_sites"]:
        print(f"{r['name']:<20s}  {r['n_sites']:>8d}  {r['auc_mean']:>10.4f}")

    # Tabla por sitio cruzando experimentos
    print()
    print("="*90)
    print("DETALLE POR SITIO (n>=30)")
    print("="*90)
    sites = sorted(set().union(*[set(d["site"]) for d in dfs.values()]))
    header = f"{'Site':<12s} {'n':>4s}"
    for name in dfs:
        header += f"  {name[:14]:>14s}"
    print(header)
    print("-"*len(header))
    for site in sites:
        line = ""
        n_val = None
        for name, d in dfs.items():
            row = d[d["site"] == site]
            if len(row) > 0:
                n_val = int(row.iloc[0]["n"])
                line += f"  {row.iloc[0]['auc_mean']:>14.4f}"
            else:
                line += f"  {'—':>14s}"
        if n_val is not None:
            print(f"{site:<12s} {n_val:>4d}{line}")

    # Save JSON
    out_json = args.out.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump({
            "experiments": list(experiments.keys()),
            "n30": results["n30"],
            "all_sites": results["all_sites"],
        }, f, indent=2)
    print(f"\nGuardado: {out_json}")

    # Save MD
    md = ["# Comparación n>=30", ""]
    md.append("## Resumen (n>=30)")
    md.append("")
    md.append("| Experimento | N sitios | AUC mean | std entre sitios | mean std entre seeds |")
    md.append("|---|---|---|---|---|")
    for r in results["n30"]:
        md.append(f"| {r['name']} | {r['n_sites']} | {r['auc_mean']:.4f} | "
                  f"{r['auc_std_between_sites']:.4f} | {r['auc_mean_of_stds']:.4f} |")
    md.append("")
    md.append("## Detalle por sitio")
    md.append("")
    md_header = "| Site | n |"
    md_sep = "|---|---|"
    for name in dfs:
        md_header += f" {name} |"
        md_sep += "---|"
    md.append(md_header)
    md.append(md_sep)
    for site in sites:
        line = f"| {site} |"
        n_val = None
        vals = ""
        for name, d in dfs.items():
            row = d[d["site"] == site]
            if len(row) > 0:
                n_val = int(row.iloc[0]["n"])
                vals += f" {row.iloc[0]['auc_mean']:.4f} |"
            else:
                vals += " — |"
        if n_val is not None:
            line += f" {n_val} |{vals}"
            md.append(line)
    md_out = args.out
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
