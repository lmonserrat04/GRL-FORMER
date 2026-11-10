# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Estabilidad de hard cases entre baseline (5 seeds) y focal loss (3 seeds).

Paso 1: hard_A = sujetos con n_errors >= 4 en baseline (5 seeds).
Paso 2: hard_B = sujetos con n_errors >= threshold en focal (3 seeds).
        Se prueban dos thresholds: >=2 (loose) y >=3 (strict).
Paso 3: comparar |A∩B| vs esperado por azar, Jaccard, enrich, p hipergeométrico.

Uso:
    python scripts/analysis/hard_cases_overlap.py \\
        --baseline-csv experiments_finetune_only/.../seed_consistency.csv \\
        --focal-runs <run_seed42> <run_seed123> <run_seed2024>

Salidas:
    <out>/hard_cases_overlap.json
    <out>/hard_cases_overlap.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def load_focal_errors(run_dirs: list[Path]) -> pd.DataFrame:
    """
    Carga los preds_fold_*.npz de cada run y construye una tabla:
        sub_id | label | err_seed_1 | err_seed_2 | ... | n_errors | n_seeds
    """
    per_seed = []
    for seed_idx, run_dir in enumerate(run_dirs):
        ckpt_dir = run_dir / "checkpoints"
        npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
        if not npz_files:
            raise SystemExit(f"No hay preds_fold_*.npz en {ckpt_dir}")

        sub_errors = {}
        sub_labels = {}
        for f in npz_files:
            npz = np.load(f, allow_pickle=True)
            labels = npz["labels"]
            preds  = npz["preds"]
            if "test_subject_ids" in npz.files:
                sub_ids = npz["test_subject_ids"]
            else:
                sub_ids = npz["subject_indices"][npz["test_idx"]]
            for i, sid in enumerate(sub_ids):
                sid = int(sid)
                if sid in sub_errors:
                    continue
                sub_errors[sid] = int(preds[i] != labels[i])
                sub_labels[sid] = int(labels[i])
        per_seed.append((sub_errors, sub_labels))
        print(f"  seed {seed_idx+1}: {run_dir.name}  ({len(sub_errors)} sujetos)")

    # Sujetos comunes entre las 3 seeds (debería ser 831)
    common = set(per_seed[0][0].keys())
    for sub_errors, _ in per_seed[1:]:
        common &= set(sub_errors.keys())
    common = sorted(common)
    print(f"  Sujetos comunes: {len(common)}")

    rows = []
    for sid in common:
        row = {"sub_id": sid, "label": per_seed[0][1][sid]}
        errs = []
        for k, (sub_errors, _) in enumerate(per_seed):
            e = sub_errors[sid]
            row[f"err_seed_{k+1}"] = e
            errs.append(e)
        row["n_errors"] = sum(errs)
        row["n_seeds"] = len(per_seed)
        rows.append(row)

    return pd.DataFrame(rows)


def overlap_stats(set_a: set, set_b: set, n_total: int, label: str) -> dict:
    """Calcula solapamiento, Jaccard, enrich y p hipergeométrico."""
    a = len(set_a)
    b = len(set_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = inter / union if union > 0 else float("nan")
    expected = a * b / n_total
    enrichment = inter / expected if expected > 0 else float("nan")

    # Hipergeométrico: P(X >= inter) donde X ~ Hypergeom(N=n_total, K=a, n=b)
    # P(X >= k) = hypergeom.sf(k-1, N, K, n)
    p_greater = float(hypergeom.sf(inter - 1, n_total, a, b))

    return {
        "label": label,
        "n_A": a,
        "n_B": b,
        "n_inter": inter,
        "n_union": union,
        "jaccard": float(jaccard),
        "expected_by_chance": float(expected),
        "enrichment": float(enrichment),
        "p_hypergeom_greater": p_greater,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--focal-runs", type=Path, nargs="+", required=True,
                        help="3 runs de focal (una seed cada uno)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.baseline_csv.exists():
        raise SystemExit(f"No existe {args.baseline_csv}")
    if len(args.focal_runs) < 2:
        raise SystemExit(f"Se esperan al menos 2 runs, recibidos {len(args.focal_runs)}")

    out_dir = (args.out_dir if args.out_dir else
               args.focal_runs[0].parent / "hard_cases_overlap")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar baseline ─────────────────────────────────────────
    print("=== Baseline seed_consistency ===")
    df_base = pd.read_csv(args.baseline_csv)
    print(f"  {len(df_base)} sujetos")
    hard_A = set(df_base[df_base["n_errors"] >= 4]["sub_id"].astype(int))
    print(f"  hard_A (n_errors >= 4): {len(hard_A)} sujetos")

    # ── Cargar focal ────────────────────────────────────────────
    print("\n=== Focal loss runs ===")
    df_focal = load_focal_errors(args.focal_runs)
    print(f"  Total sujetos en focal: {len(df_focal)}")

    # hard_B simétrico a hard_A: n_errors >= 4 (sobre N seeds de focal)
    n_seeds_focal = len(args.focal_runs)
    hard_B = set(df_focal[df_focal["n_errors"] >= 4]["sub_id"].astype(int))
    print(f"  hard_B (n_errors >= 4, sobre {n_seeds_focal} seeds): {len(hard_B)} sujetos")

    # ── Verificar alineamiento ──────────────────────────────────
    common_subs = set(df_base["sub_id"].astype(int)) & set(df_focal["sub_id"].astype(int))
    n_total = len(common_subs)
    print(f"\n=== Sujetos comunes baseline ∩ focal: {n_total} ===")

    # Restringir A y B a los sujetos comunes (por seguridad)
    hard_A = hard_A & common_subs
    hard_B = hard_B & common_subs

    # ── Calcular solapamiento ───────────────────────────────────
    stats = overlap_stats(hard_A, hard_B, n_total,
                          f"hard_A (5 seeds baseline, >=4) vs hard_B ({n_seeds_focal} seeds focal, >=4)")

    s = stats
    print("\n" + "="*70)
    print(s["label"])
    print("="*70)
    print(f"  |A| = {s['n_A']}")
    print(f"  |B| = {s['n_B']}")
    print(f"  |A∩B| = {s['n_inter']}")
    print(f"  |A∪B| = {s['n_union']}")
    print(f"  Jaccard = {s['jaccard']:.4f}")
    print(f"  Esperado por azar = {s['expected_by_chance']:.2f}")
    print(f"  Enrichment = {s['enrichment']:.2f}×")
    print(f"  p (hipergeom, P(X≥k)) = {s['p_hypergeom_greater']:.4e}")

    # ── Guardar JSON ────────────────────────────────────────────
    summary = {
        "baseline_csv": str(args.baseline_csv),
        "focal_runs": [str(r) for r in args.focal_runs],
        "n_total_common": n_total,
        "hard_A_criterion": "n_errors >= 4 (baseline, 5 seeds)",
        "hard_B_criterion": f"n_errors >= 4 (focal, {n_seeds_focal} seeds)",
        "overlap": stats,
    }
    json_out = out_dir / "hard_cases_overlap.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGuardado: {json_out}")

    # ── Markdown ────────────────────────────────────────────────
    md = ["# Estabilidad de hard cases — baseline vs focal", ""]
    md.append(f"**Baseline:** `{args.baseline_csv}`")
    md.append(f"**Focal runs:**")
    for r in args.focal_runs:
        md.append(f"- `{r}`")
    md.append("")
    md.append(f"**Sujetos comunes baseline ∩ focal:** {n_total}")
    md.append("")
    md.append("## Definiciones")
    md.append("")
    md.append("- **hard_A:** n_errors ≥ 4 en baseline (5 seeds)")
    md.append(f"- **hard_B:** n_errors ≥ 4 en focal ({n_seeds_focal} seeds)")
    md.append("")
    md.append("## Solapamiento")
    md.append("")
    md.append("| A | B | A∩B | A∪B | Jaccard | Esperado azar | Enrichment | p (hipergeom) |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.append(f"| {stats['n_A']} | {stats['n_B']} | {stats['n_inter']} | "
              f"{stats['n_union']} | {stats['jaccard']:.4f} | "
              f"{stats['expected_by_chance']:.1f} | {stats['enrichment']:.2f}× | "
              f"{stats['p_hypergeom_greater']:.2e} |")
    md.append("")
    md.append("## Interpretación")
    md.append("")
    md.append("- **Jaccard** = |A∩B| / |A∪B|. 0 = sin solapamiento; 1 = idénticos.")
    md.append("- **Enrichment** = observado / esperado por azar. >1 = más solapamiento del esperado.")
    md.append("- **p (hipergeom)** = probabilidad de observar |A∩B| o más por azar. <0.05 = solapamiento significativo.")
    md.append("")
    md.append("**Lectura:**")
    md.append("")
    md.append("- Si **enrichment ≫ 1 y p < 0.05** → los hard cases son consistentes entre pérdidas. La dificultad está en los datos, no en el criterio de entrenamiento.")
    md.append("- Si **enrichment ≈ 1 y p > 0.05** → hard_A y hard_B son grupos casi independientes. Cada pérdida identifica sujetos difíciles distintos. La dificultad es dependiente del modelo.")
    md.append("- Si **enrichment intermedio** (1.5-2×) → señal parcial de consistencia. Interpretar con cautela.")

    md_out = out_dir / "hard_cases_overlap.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
