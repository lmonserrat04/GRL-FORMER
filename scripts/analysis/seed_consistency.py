# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Analiza la consistencia de errores a través de múltiples seeds.

Uso:
    python scripts/analysis/seed_consistency.py <run_dirs...> --csv data/csv/data_train.csv

Ejemplo:
    python scripts/analysis/seed_consistency.py \
        experiments_finetune_only/20261001_233905_twotst_original_loso_preds \
        experiments_finetune_only/20261002_000950_twotst_seed123 \
        experiments_finetune_only/20261002_001840_twotst_seed2024 \
        experiments_finetune_only/20261002_002626_twotst_seed7 \
        experiments_finetune_only/20261002_003406_twotst_seed999 \
        --csv data/csv/data_train.csv

Salida (en el directorio del primer run):
    seed_consistency.csv          una fila por sujeto, con error_<seed>
    seed_consistency.json         resumen agregado
"""
import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_run_preds(run_dir: Path) -> dict:
    """
    Carga todas las predicciones de un run.

    Devuelve dict {sub_id: {"label": int, "prob": float, "pred": int,
                            "site": str, "fold": int}}
    """
    ckpt_dir = run_dir / "checkpoints"
    npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
    if not npz_files:
        raise SystemExit(f"No hay preds_fold_*.npz en {ckpt_dir}")

    preds = {}
    for f in npz_files:
        npz = np.load(f, allow_pickle=True)
        labels = npz["labels"]
        probs = npz["probs"]
        preds_arr = npz["preds"]
        sub_ids = npz["test_subject_ids"]

        for i, sid in enumerate(sub_ids):
            sid_int = int(sid)
            if sid_int in preds:
                continue  # por seguridad, en LOSO no se repite
            preds[sid_int] = {
                "label": int(labels[i]),
                "prob": float(probs[i]),
                "pred": int(preds_arr[i]),
                "fold": int(f.stem.split("_")[-1]),
            }
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+",
                        help="Directorios de runs a analizar (uno por seed)")
    parser.add_argument("--csv", type=Path, default=Path("data/csv/data_train.csv"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Directorio de salida (por defecto: primer run_dir)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"No existe CSV: {args.csv}")

    out_dir = args.out if args.out else args.run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar CSV para mapear SUB_ID → SITE_ID
    df_csv = pd.read_csv(args.csv)
    sub_to_site = dict(zip(df_csv["SUB_ID"].astype(int), df_csv["SITE_ID"].astype(str)))

    # ─── Cargar todos los runs ───────────────────────────────────────
    print(f"Cargando {len(args.run_dirs)} runs...")
    all_preds = []
    for rd in args.run_dirs:
        preds = load_run_preds(rd)
        all_preds.append(preds)
        n_subj = len(preds)
        n_err = sum(1 for p in preds.values() if p["pred"] != p["label"])
        print(f"  {rd.name}: {n_subj} sujetos, {n_err} errores "
              f"({n_err/n_subj*100:.1f}%)")

    # Verificar que todos los runs tienen los mismos sujetos
    sets = [set(p.keys()) for p in all_preds]
    common = set.intersection(*sets)
    print(f"\nSujetos en común entre los {len(args.run_dirs)} runs: {len(common)}")

    # ─── Construir tabla por sujeto ──────────────────────────────────
    rows = []
    for sid in sorted(common):
        row = {
            "sub_id": sid,
            "site": sub_to_site.get(sid, "UNKNOWN"),
        }
        # Tomamos label de la primera seed (todas coinciden)
        row["label"] = all_preds[0][sid]["label"]

        errors = []
        probs = []
        for k, preds in enumerate(all_preds):
            p = preds[sid]
            err = int(p["pred"] != p["label"])
            errors.append(err)
            probs.append(p["prob"])
            row[f"err_seed_{k+1}"] = err

        row["n_errors"] = sum(errors)
        row["n_seeds"] = len(all_preds)
        row["frac_errors"] = row["n_errors"] / row["n_seeds"]
        row["mean_prob"] = float(np.mean(probs))
        row["std_prob"] = float(np.std(probs))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["n_errors", "site", "sub_id"], ascending=[False, True, True])

    # ─── Guardar CSV ─────────────────────────────────────────────────
    csv_out = out_dir / "seed_consistency.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nGuardado: {csv_out}")

    # ─── Análisis agregado ───────────────────────────────────────────
    print("\n" + "="*70)
    print("HISTOGRAMA: nº de seeds en que cada sujeto falla")
    print("="*70)
    hist = df["n_errors"].value_counts().sort_index()
    for n_err, count in hist.items():
        print(f"  Falla en {n_err}/{len(all_preds)} seeds: {count} sujetos "
              f"({count/len(df)*100:.1f}%)")

    # ─── Sujetos problemáticos (fallan ≥4/5) ─────────────────────────
    n_seeds = len(all_preds)
    threshold = n_seeds - 1  # ≥4 de 5
    problem = df[df["n_errors"] >= threshold]
    print(f"\n" + "="*70)
    print(f"SUJETOS PROBLEMÁTICOS (fallan ≥{threshold}/{n_seeds} seeds)")
    print("="*70)
    print(f"Total: {len(problem)} sujetos ({len(problem)/len(df)*100:.1f}%)")
    print(f"  ASD: {(problem['label']==1).sum()}")
    print(f"  TD:  {(problem['label']==0).sum()}")

    print(f"\nPor sitio:")
    site_counts = problem.groupby("site").agg(
        n_total=("sub_id", "count"),
        n_problem=("sub_id", "count"),
    ).reset_index()
    # Contar total por sitio en el dataset
    site_total = df.groupby("site").size().to_dict()
    for s in sorted(problem["site"].unique()):
        n_prob = len(problem[problem["site"] == s])
        n_tot = site_total.get(s, 0)
        pct = n_prob / n_tot * 100 if n_tot > 0 else 0
        print(f"  {s:<12s}: {n_prob:>3d}/{n_tot:>3d}  ({pct:>5.1f}%)")

    # ─── Sujetos bien clasificados (0 errores) ───────────────────────
    good = df[df["n_errors"] == 0]
    print(f"\n" + "="*70)
    print(f"SUJETOS SIEMPRE BIEN (0/{n_seeds} errores)")
    print("="*70)
    print(f"Total: {len(good)} sujetos ({len(good)/len(df)*100:.1f}%)")
    print(f"  ASD: {(good['label']==1).sum()}")
    print(f"  TD:  {(good['label']==0).sum()}")

    # ─── Guardar resumen JSON ────────────────────────────────────────
    summary = {
        "n_runs": len(args.run_dirs),
        "runs": [str(rd) for rd in args.run_dirs],
        "n_common_subjects": len(common),
        "histogram_n_errors": {int(k): int(v) for k, v in hist.items()},
        "problem_subjects": {
            "threshold": threshold,
            "n": len(problem),
            "n_asd": int((problem["label"]==1).sum()),
            "n_td": int((problem["label"]==0).sum()),
            "by_site": {s: int(len(problem[problem["site"]==s]))
                        for s in problem["site"].unique()},
        },
        "always_correct_subjects": {
            "n": len(good),
            "n_asd": int((good["label"]==1).sum()),
            "n_td": int((good["label"]==0).sum()),
        },
    }
    json_out = out_dir / "seed_consistency.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGuardado: {json_out}")

    # ─── Listado (top 20 problemáticos) ─────────────────────────────
    print(f"\n" + "="*70)
    print("TOP 20 SUJETOS MÁS PROBLEMÁTICOS")
    print("="*70)
    print(f"{'sub_id':>10s}  {'site':<12s}  {'label':>5s}  {'errors':>6s}  {'mean_prob':>9s}")
    for _, r in problem.head(20).iterrows():
        print(f"{r['sub_id']:>10d}  {r['site']:<12s}  {r['label']:>5d}  "
              f"{r['n_errors']:>3d}/{n_seeds}  {r['mean_prob']:>9.3f}")


if __name__ == "__main__":
    main()
