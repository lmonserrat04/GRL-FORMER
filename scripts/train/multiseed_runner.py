# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Multi-seed runner genérico.

Ejecuta un script de training (run_local_attn.py, etc.) una vez por seed,
recolecta los results.json y agrega AUC ± std entre seeds.

Diseño:
- Reanudable: salta seeds cuyo results.json ya exista.
- Agnóstico al script: se pasa el path del script por CLI.
- Agrega en {out_root}/multiseed_summary.json.

Uso:
    python scripts/train/multiseed_runner.py \
        --run-script scripts/train/run_local_attn.py \
        --seeds 42 123 2024 \
        --base-args "--k 8" \
        --out-root experiments_local_attn_k8_multiseed
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_one_seed(
    run_script: Path,
    seed: int,
    base_args: list[str],
    out_root: Path,
    skip_if_exists: bool,
) -> Path | None:
    """
    Ejecuta el script para una seed. Devuelve el path del results.json
    (dentro del directorio del seed) o None si falló.
    """
    seed_dir = out_root / f"seed_{seed}"
    results_json = seed_dir / "results.json"

    if skip_if_exists and results_json.exists():
        print(f"  [SKIP] seed={seed} (ya existe {results_json})")
        return results_json

    seed_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(run_script),
        "--seed", str(seed),
        "--out-root", str(out_root),
    ] + base_args

    print(f"  [RUN] seed={seed}")
    print(f"        cmd: {' '.join(shlex.quote(c) for c in cmd)}")

    try:
        result = subprocess.run(cmd, check=True, cwd=str(run_script.parents[2]))
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] seed={seed} exit={e.returncode}")
        return None

    # El script de training crea su propio exp_id; buscamos el results.json
    # más reciente dentro de out_root que sea de esta seed.
    candidates = sorted(
        out_root.glob(f"*_seed_{seed}/results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        # Alternativa: seed_dir/results.json
        if results_json.exists():
            return results_json
        print(f"  [WARN] seed={seed}: no se encontró results.json")
        return None

    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-script", type=Path, required=True,
                        help="Script de training (acepta --seed y --out-root)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--base-args", type=str, default="",
                        help="Args extra para pasar al script (string)")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Salta seeds cuyo results.json ya existe")
    args = parser.parse_args()

    if not args.run_script.exists():
        raise SystemExit(f"No existe {args.run_script}")

    base_args = shlex.split(args.base_args) if args.base_args else []
    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed runner")
    print(f"  script:   {args.run_script}")
    print(f"  seeds:    {args.seeds}")
    print(f"  args:     {base_args}")
    print(f"  out_root: {args.out_root}")
    print()

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        results_json = run_one_seed(
            args.run_script, seed, base_args,
            args.out_root, args.skip_if_exists,
        )
        if results_json is None or not results_json.exists():
            print(f"  [WARN] sin results para seed={seed}")
            continue

        with open(results_json) as f:
            s = json.load(f)
        s["seed"] = seed
        all_results.append(s)

        auc = s.get("auc", {}).get("mean", 0.0)
        print(f"  AUC: {auc:.4f}")

    if not all_results:
        raise SystemExit("No hay seeds completadas.")

    # ─── Agregado ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGREGADO ENTRE SEEDS")
    print(f"{'='*70}")

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    agg = {}
    for metric in metric_names:
        vals = [s[metric]["mean"] for s in all_results if metric in s]
        if not vals:
            continue
        agg[metric] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "per_seed": [
                {"seed": s["seed"], "mean": float(s[metric]["mean"])}
                for s in all_results if metric in s
            ],
        }
        print(f"  {metric:<12s}: {agg[metric]['mean']:.4f} "
              f"+/- {agg[metric]['std']:.4f}  "
              f"[{agg[metric]['min']:.4f}, {agg[metric]['max']:.4f}]")

    print(f"\nPer-seed AUC:")
    for s in all_results:
        print(f"  seed {s['seed']}: "
              f"{s.get('auc', {}).get('mean', 0.0):.4f} "
              f"+/- {s.get('auc', {}).get('std', 0.0):.4f}")

    out = {
        "n_seeds": len(all_results),
        "seeds": [s["seed"] for s in all_results],
        "aggregated": agg,
        "per_seed": all_results,
    }

    out_path = args.out_root / "multiseed_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
