# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
AUC por sitio promediado entre múltiples seeds, con IC 95% bootstrap
pareado por sujeto.

Para cada sitio (que en LOSO coincide con un fold):
  - Point estimate: mean(AUC_seed_1..N) sobre todos los sujetos del sitio.
  - IC 95%: bootstrap pareado por sujeto.
       * Remuestrear N sujetos con reemplazo.
       * Cada sujeto mantiene JUNTAS sus N predicciones (una por seed).
       * Recalcular el AUC de cada seed sobre el remuestreo.
       * Promediar los N AUCs.
       * Repetir B veces (descartando remuestreos con una sola clase).
       * Percentiles 2.5 / 97.5 del vector de B medias.

Uso:
    python scripts/analysis/site_auc_multiseed.py \
        <run_seed1> <run_seed2> ... \
        --csv data/csv/data_train.csv \
        --n-bootstrap 2000 \
        --out-dir <dir_salida>

Salida:
    site_auc_multiseed.csv     tabla por sitio
    site_auc_multiseed.json    resumen estructurado
    site_auc_multiseed.md      informe legible
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────────────────────────────
# Carga
# ──────────────────────────────────────────────────────────────────────

def load_run_by_subject(run_dir: Path) -> dict:
    """
    Carga todas las predicciones de un run.

    Devuelve: {sub_id (int): (label (int), prob (float))}
    """
    ckpt_dir = run_dir / "checkpoints"
    npz_files = sorted(ckpt_dir.glob("preds_fold_*.npz"))
    if not npz_files:
        raise SystemExit(f"No hay preds_fold_*.npz en {ckpt_dir}")

    out = {}
    for f in npz_files:
        npz = np.load(f, allow_pickle=True)
        labels = npz["labels"]
        probs = npz["probs"]
        if "test_subject_ids" in npz.files:
            sub_ids = npz["test_subject_ids"]
        else:
            sub_ids = npz["subject_indices"][npz["test_idx"]]
        for i, sid in enumerate(sub_ids):
            sid_int = int(sid)
            if sid_int in out:
                continue  # en LOSO cada sujeto aparece una vez
            out[sid_int] = (int(labels[i]), float(probs[i]))
    return out


# ──────────────────────────────────────────────────────────────────────
# Bootstrap pareado por sujeto
# ──────────────────────────────────────────────────────────────────────

def paired_bootstrap_mean_auc(
    labels: np.ndarray,          # (N,)
    probs_matrix: np.ndarray,    # (N, S)  — una columna por seed
    n_bootstrap: int,
    rng: np.random.Generator,
    max_attempts_factor: int = 20,
) -> np.ndarray:
    """
    Bootstrap pareado por sujeto.

    En cada iteración:
      - Remuestrea N sujetos con reemplazo (índices).
      - Recalcula el AUC de cada seed sobre el remuestreo.
      - Promedia los S AUCs.
    Si el remuestreo tiene una sola clase, se descarta y se vuelve a muestrear.

    Devuelve array (n_bootstrap,) de medias.
    """
    n, n_seeds = probs_matrix.shape
    means = []
    attempts = 0
    max_attempts = n_bootstrap * max_attempts_factor

    while len(means) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue  # remuestreo con una sola clase → descartar

        aucs_b = []
        for s in range(n_seeds):
            p_b = probs_matrix[idx, s]
            aucs_b.append(roc_auc_score(y_b, p_b))
        means.append(float(np.mean(aucs_b)))

    if len(means) < n_bootstrap:
        print(f"  [WARN] solo se obtuvieron {len(means)}/{n_bootstrap} "
              f"remuestreos válidos tras {attempts} intentos")

    return np.array(means)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+",
                        help="Un run por seed")
    parser.add_argument("--csv", type=Path,
                        default=Path("data/csv/data_train.csv"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla del bootstrap (no de entrenamiento)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directorio de salida (default: primer run)")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else args.run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV mapping ──────────────────────────────────────────────────
    df_csv = pd.read_csv(args.csv)
    sub_to_site = dict(zip(df_csv["SUB_ID"].astype(int),
                            df_csv["SITE_ID"].astype(str)))

    n_seeds = len(args.run_dirs)

    # ── Cargar todos los runs ────────────────────────────────────────
    print(f"Cargando {n_seeds} runs...")
    runs_by_subject = []
    for rd in args.run_dirs:
        m = load_run_by_subject(rd)
        runs_by_subject.append(m)
        print(f"  {rd.name}: {len(m)} sujetos")

    # ── Construir matriz por sitio ───────────────────────────────────
    # Para cada sitio: sub_ids comunes entre los N runs
    sites_to_subs = defaultdict(set)
    for m in runs_by_subject:
        for sid in m.keys():
            site = sub_to_site.get(sid, "UNKNOWN")
            sites_to_subs[site].add(sid)

    per_site = {}
    for site, sids in sites_to_subs.items():
        common = set(sids)
        for m in runs_by_subject:
            common &= set(m.keys())
        common = sorted(common)
        if not common:
            continue

        n = len(common)
        labels = np.array([runs_by_subject[0][sid][0] for sid in common])
        probs_matrix = np.zeros((n, n_seeds), dtype=np.float64)
        for s, m in enumerate(runs_by_subject):
            for i, sid in enumerate(common):
                probs_matrix[i, s] = m[sid][1]

        per_site[site] = {
            "labels": labels,
            "probs": probs_matrix,
        }

    # ── Point estimate + IC pareado por sitio ───────────────────────
    print(f"\nCalculando AUC mean ± IC bootstrap pareado por sujeto "
          f"(B={args.n_bootstrap})...\n")

    rows = []
    for site in sorted(per_site.keys()):
        d = per_site[site]
        y = d["labels"]
        P = d["probs"]
        n = len(y)

        if len(np.unique(y)) < 2:
            continue

        # Point estimate: mean(AUC_seed_1..N) sobre los N sujetos originales
        aucs = [roc_auc_score(y, P[:, s]) for s in range(n_seeds)]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1)) if n_seeds > 1 else 0.0

        # Bootstrap pareado por sujeto
        rng = np.random.default_rng(args.seed + (hash(site) % 10000))
        boot_means = paired_bootstrap_mean_auc(
            y, P, args.n_bootstrap, rng
        )

        if len(boot_means) >= 2:
            lo, hi = np.percentile(boot_means, [2.5, 97.5])
        else:
            lo, hi = None, None

        rows.append({
            "site": site,
            "n": n,
            "n_seeds": n_seeds,
            "auc_mean": mean_auc,
            "auc_std": std_auc,
            "auc_ci_lo": float(lo) if lo is not None else None,
            "auc_ci_hi": float(hi) if hi is not None else None,
            "n_boot_valid": int(len(boot_means)),
        })

    df_out = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)

    # ── Guardar CSV ─────────────────────────────────────────────────
    csv_out = out_dir / "site_auc_multiseed.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"Guardado: {csv_out}")

    # ── Guardar JSON ────────────────────────────────────────────────
    json_out = out_dir / "site_auc_multiseed.json"
    with open(json_out, "w") as f:
        json.dump({
            "n_runs": n_seeds,
            "runs": [str(r) for r in args.run_dirs],
            "n_bootstrap": args.n_bootstrap,
            "bootstrap": "paired by subject (same subject -> keep N probs together)",
            "per_site": df_out.to_dict(orient="records"),
        }, f, indent=2)
    print(f"Guardado: {json_out}")

    # ── Print tabla ─────────────────────────────────────────────────
    print("\n" + "="*86)
    print(f"AUC por sitio — {n_seeds} seeds, IC 95% bootstrap pareado por sujeto "
          f"(B={args.n_bootstrap})")
    print("="*86)
    print(f"{'site':<12s} {'n':>4s} {'auc mean':>10s} {'std':>7s}  {'IC 95%':<22s}")
    print("-"*86)
    for _, r in df_out.iterrows():
        ic_str = (f"[{r['auc_ci_lo']:.4f}, {r['auc_ci_hi']:.4f}]"
                  if r["auc_ci_lo"] is not None else "—")
        print(f"{r['site']:<12s} {int(r['n']):>4d} {r['auc_mean']:>10.4f} "
              f"{r['auc_std']:>7.4f}  {ic_str:<22s}")

    print()
    print(f"Media global entre sitios: {df_out['auc_mean'].mean():.4f}")

    # ── Informe Markdown ────────────────────────────────────────────
    md = ["# AUC por sitio — multi-seed (IC bootstrap pareado por sujeto)", ""]
    md.append(f"**Seeds:** {n_seeds}")
    for r in args.run_dirs:
        md.append(f"- `{r.name}`")
    md.append(f"**Bootstrap:** {args.n_bootstrap} remuestreos pareados por sujeto")
    md.append("")
    md.append("## Tabla por sitio")
    md.append("")
    md.append("| Sitio | n | AUC mean | std | IC 95% |")
    md.append("|---|---|---|---|---|")
    for _, r in df_out.iterrows():
        ic_str = (f"[{r['auc_ci_lo']:.4f}, {r['auc_ci_hi']:.4f}]"
                  if r["auc_ci_lo"] is not None else "—")
        md.append(f"| {r['site']} | {int(r['n'])} | {r['auc_mean']:.4f} | "
                  f"{r['auc_std']:.4f} | {ic_str} |")
    md.append("")
    md.append(f"**Media global entre sitios:** {df_out['auc_mean'].mean():.4f}")
    md.append("")
    md.append("## Notas")
    md.append("- **AUC mean**: mean(AUC_seed_1..N) sobre todos los sujetos del sitio.")
    md.append("- **IC 95%**: bootstrap pareado por sujeto. "
              "Cada remuestreo selecciona sujetos con reemplazo y conserva "
              "sus N probs (una por seed) juntas; se recalcula el AUC de "
              "cada seed y se promedian.")
    md.append("- **Descarte**: remuestreos con una sola clase se descartan "
              "y se vuelve a muestrear.")

    md_out = out_dir / "site_auc_multiseed.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
