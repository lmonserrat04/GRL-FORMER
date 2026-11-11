# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Linear probe LOSO sobre la representación temporal h_ts.

Protocolo
---------
Para cada fold N (sitio held-out):
  1. Reconstruir el modelo finetuneado N y cargar
     checkpoints/best_finetune_fold_N.pt.
  2. Congelar el modelo (eval + no_grad): el encoder NO se entrena.
  3. Extraer h_ts para train + val (todos los sujetos de los 18 sitios)
     y para test (sujetos del sitio N, nunca vistos por el modelo N).
  4. Estandarizar con train + val y entrenar una regresión logística.
  5. Evaluar AUC en el sitio held-out.

La logística es deliberadamente débil: solo puede trazar una frontera
lineal. Si h_ts no permite separar ASD/TD con una frontera lineal, la
información diagnóstica no está accesible de forma lineal en esa
representación.

Representaciones
----------------
  h_ts_raw  : salida del Transformer temporal TST1 (512 dims), antes de
              la proyección contrastiva.
  h_ts_proj : salida después de proj_head_1 (128 dims). Es el vector que
              entra a la fusión y al clasificador auxiliar.
  h_fc_raw  : salida del Transformer de conectividad TST2 (256 dims).
  h_fc_proj : salida después de proj_head_2 (128 dims).
  fused     : salida del módulo de fusión (vector que entra al clasificador).

Uso
---
    python scripts/analysis/linear_probe_h_ts.py \
        --run-dir experiments_finetune_only/20261110_200401_ts_aux_loss_lambda1_loso \
        --ckpt-contrastive experiments/20260928_160330_transformer_baseline/checkpoints/contrastive_global.pt \
        --eval-protocol loso \
        --representations h_ts_raw h_ts_proj h_fc_raw h_fc_proj fused

Salidas (por defecto en <run-dir>/linear_probe_h_ts/):
    linear_probe_h_ts_by_site.csv
    linear_probe_h_ts_global.json
    linear_probe_h_ts_report.md
    features/fold_N_{train,test}.npz   (solo con --save-features)
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.loaders.dataloader import get_finetune_loaders
from training.setup import build_experiment


VALID_REPS = ("h_ts_raw", "h_ts_proj", "h_fc_raw", "h_fc_proj", "fused")
DEFAULT_REPS = ("h_ts_raw", "h_ts_proj")


def rep_column(rep: str) -> str:
    """Nombre de columna del probe.

    ``auc_fused`` ya está reservado para la referencia del modelo completo
    (preds_fold_*.npz); el probe sobre la representación fused se guarda
    como ``auc_fused_probe`` para no duplicar columnas.
    """
    return "auc_fused_probe" if rep == "fused" else f"auc_{rep}"


def resolve_contrastive_checkpoint(run_dir: Path, config: dict,
                                   explicit: str | None) -> str:
    """Localiza el checkpoint contrastive necesario para construir proj heads."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if config.get("CKPT_CONTRASTIVE"):
        candidates.append(Path(config["CKPT_CONTRASTIVE"]))

    results_path = run_dir / "results.json"
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
            ckpt_dir = results.get("ckpt_dir")
            if ckpt_dir:
                candidates.append(Path(ckpt_dir) / "contrastive_global.pt")
        except Exception:
            pass

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise SystemExit(
        "No encuentro contrastive_global.pt. Pásalo explícitamente con "
        "--ckpt-contrastive <ruta>. Sin ese checkpoint no se pueden construir "
        "las projection heads y la arquitectura no coincide con el fine-tune."
    )


def make_eval_loader(loader: DataLoader, batch_size: int) -> DataLoader:
    """DataLoader sin shuffle ni drop_last (para no perder sujetos)."""
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def extract_h_ts(model, loader: DataLoader, device: str) -> dict:
    """Extrae representaciones de un loader, sin entrenar nada.

    Devuelve h_ts_raw, h_ts_proj, h_fc_raw, h_fc_proj y fused.
    """
    model.eval()
    raw_chunks, ts_proj_chunks = [], []
    fc_raw_chunks, fc_proj_chunks, fused_chunks = [], [], []
    label_chunks = []

    with torch.inference_mode():
        for batch in loader:
            ts = batch["timeseries"].to(device, non_blocking=True)
            pcc = batch["pcc_vector"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy()

            h_raw = model.transformer_ts(ts, mode="finetune")
            h_fc_raw = model.transformer_fc(pcc, mode="finetune")
            if model.proj_head_1 is not None:
                h_proj = model.proj_head_1(h_raw)
            else:
                h_proj = h_raw
            if model.proj_head_2 is not None:
                h_fc_proj = model.proj_head_2(h_fc_raw)
            else:
                h_fc_proj = h_fc_raw
            fused = model.fusion(h_proj, h_fc_proj)

            raw_chunks.append(h_raw.detach().cpu().numpy())
            ts_proj_chunks.append(h_proj.detach().cpu().numpy())
            fc_raw_chunks.append(h_fc_raw.detach().cpu().numpy())
            fc_proj_chunks.append(h_fc_proj.detach().cpu().numpy())
            fused_chunks.append(fused.detach().cpu().numpy())
            label_chunks.append(y)

    if not raw_chunks:
        raise RuntimeError("Loader vacío al extraer representaciones.")

    return {
        "h_ts_raw": np.concatenate(raw_chunks, axis=0).astype(np.float32),
        "h_ts_proj": np.concatenate(ts_proj_chunks, axis=0).astype(np.float32),
        "h_fc_raw": np.concatenate(fc_raw_chunks, axis=0).astype(np.float32),
        "h_fc_proj": np.concatenate(fc_proj_chunks, axis=0).astype(np.float32),
        "fused": np.concatenate(fused_chunks, axis=0).astype(np.float32),
        "labels": np.concatenate(label_chunks, axis=0).astype(np.int64),
    }


def fit_logistic_probe(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, C: float,
                       max_iter: int) -> np.ndarray:
    """Estandariza con train y ajusta una regresión logística simple."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_test_s)[:, 1]


def load_fused_baseline_auc(preds_path: Path, y_test: np.ndarray) -> float:
    """AUC del modelo completo (fused), solo como referencia."""
    if not preds_path.exists():
        return float("nan")

    data = np.load(preds_path, allow_pickle=True)
    if "labels" not in data or "probs" not in data:
        return float("nan")

    labels = np.asarray(data["labels"]).reshape(-1)
    probs = np.asarray(data["probs"]).reshape(-1)
    if labels.shape != y_test.shape or not np.array_equal(labels, y_test):
        print(f"  [WARN] {preds_path.name}: labels baseline no alinean; "
              f"se omite AUC fused.")
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def bootstrap_mean_auc(labels_per_site: list[np.ndarray],
                       probs_per_site: list[np.ndarray],
                       n_bootstrap: int, seed: int) -> np.ndarray:
    """Bootstrap de sujetos dentro de cada sitio; media macro de AUC."""
    if n_bootstrap <= 0 or not labels_per_site:
        return np.array([])

    rng = np.random.default_rng(seed)
    means = []
    attempts = 0
    max_attempts = n_bootstrap * 20

    while len(means) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        site_aucs = []
        ok = True
        for labels, probs in zip(labels_per_site, probs_per_site):
            n = len(labels)
            idx = rng.integers(0, n, size=n)
            y_b = labels[idx]
            p_b = probs[idx]
            if len(np.unique(y_b)) < 2:
                ok = False
                break
            site_aucs.append(float(roc_auc_score(y_b, p_b)))
        if ok:
            means.append(float(np.mean(site_aucs)))

    return np.asarray(means, dtype=np.float64)


def json_safe(value):
    """Convierte tipos numpy y NaN a JSON válido."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def fmt(value: float, ndigits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{ndigits}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe LOSO sobre h_ts con encoder congelado."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run con config.yaml, checkpoints/ y preds_fold_*.npz")
    parser.add_argument("--ckpt-contrastive", default=None,
                        help="contrastive_global.pt para construir las proj heads. "
                             "Si se omite, se busca en config.yaml o results.json.")
    parser.add_argument("--representations", nargs="+", default=list(DEFAULT_REPS),
                        choices=list(VALID_REPS),
                        help="Representaciones a probar (default: h_ts_raw h_ts_proj).")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverso de la regularización L2 de la logística.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Bootstrap para IC95%% de la AUC media; 0 lo desactiva.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <run-dir>/linear_probe_h_ts/")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subconjunto de folds (para smoke tests).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override FINETUNING.BATCH_SIZE para extracción.")
    parser.add_argument("--eval-protocol", choices=("loso", "kfold"),
                        default="loso",
                        help="Protocolo de reconstrucción de splits. Para estos "
                             "runs usa loso, aunque el config copiado diga kfold.")
    parser.add_argument("--save-features", action="store_true",
                        help="Guardar features train/test por fold.")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = run_dir / "config.yaml"
    ckpt_dir = run_dir / "checkpoints"
    if not config_path.exists():
        raise SystemExit(f"No existe {config_path}")
    if not ckpt_dir.exists():
        raise SystemExit(f"No existe {ckpt_dir}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["EXPERIMENT_TYPE"] = "finetune"
    config["DEVICE"] = args.device
    config["NUM_WORKERS"] = 0

    original_protocol = config.get("EVAL_PROTOCOL", "kfold")
    if original_protocol != args.eval_protocol:
        print(f"[INFO] EVAL_PROTOCOL en config={original_protocol!r}; "
              f"se fuerza {args.eval_protocol!r} para reconstruir los folds.")
    config["EVAL_PROTOCOL"] = args.eval_protocol

    ckpt_contrastive = resolve_contrastive_checkpoint(
        run_dir, config, args.ckpt_contrastive
    )
    config["CKPT_CONTRASTIVE"] = ckpt_contrastive

    out_dir = (args.out_dir if args.out_dir is not None
               else run_dir / "linear_probe_h_ts")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    if args.save_features:
        feat_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(ckpt_dir.glob("best_finetune_fold_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        raise SystemExit(f"No hay best_finetune_fold_*.pt en {ckpt_dir}")

    fold_ids = [int(p.stem.split("_")[-1]) for p in ckpts]
    if args.folds is not None:
        fold_ids = [f for f in fold_ids if f in args.folds]
        if not fold_ids:
            raise SystemExit("Ningún fold coincide con --folds.")

    bs = args.batch_size or int(config["FINETUNING"]["BATCH_SIZE"])

    print(f"Run:               {run_dir}")
    print(f"Checkpoints:       {len(fold_ids)} folds {fold_ids}")
    print(f"Contrastive:       {ckpt_contrastive}")
    print(f"Protocolo:         {args.eval_protocol}")
    print(f"Representaciones:  {', '.join(args.representations)}")
    print(f"Encoder:           congelado (eval + inference_mode)")
    print(f"Clasificador:      LogisticRegression(C={args.C}, max_iter={args.max_iter})")
    print(f"Entrenamiento:     train + val (todos los sujetos de los 18 sitios)")
    print(f"Device:            {args.device}")
    print(f"Out:               {out_dir}\n")

    rows = []
    labels_per_site = {rep: [] for rep in args.representations}
    probs_per_site = {rep: [] for rep in args.representations}

    for fold_idx in fold_ids:
        print(f"=== Fold {fold_idx} ===")

        exp = build_experiment(
            config, fold_idx=fold_idx,
            ckpt_contrastive=config["CKPT_CONTRASTIVE"],
        )
        model = exp.model

        ckpt = torch.load(
            ckpt_dir / f"best_finetune_fold_{fold_idx}.pt",
            map_location="cpu",
            weights_only=False,
        )
        state = (ckpt["model_state_dict"]
                 if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                 else ckpt)
        if isinstance(state, dict) and state and all(
            k.startswith("module.") for k in state
        ):
            state = {k[len("module."):]: v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise SystemExit(
                f"No se pudo cargar {ckpt_dir / f'best_finetune_fold_{fold_idx}.pt'} "
                f"con la arquitectura reconstruida. Revisa CKPT_CONTRASTIVE/"
                f"AUX_TS_HEAD. Error original:\n{exc}"
            ) from exc

        model.to(args.device).eval()

        train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
            config,
            batch_size=bs,
            num_workers=0,
            fold_idx=fold_idx,
            n_folds=config.get("N_FOLDS", 5),
            seed=config.get("SEED", 42),
            eval_protocol=args.eval_protocol,
        )

        # train_loader usa drop_last=True durante el fine-tune. Para el probe
        # necesitamos TODOS los sujetos de los 18 sitios, así que lo rehacemos
        # sin drop_last.
        train_eval_loader = make_eval_loader(train_loader, bs)

        feat_train = extract_h_ts(model, train_eval_loader, args.device)
        feat_val = extract_h_ts(model, val_loader, args.device)
        feat_test = extract_h_ts(model, test_loader, args.device)

        y_train = np.concatenate(
            [feat_train["labels"], feat_val["labels"]], axis=0
        )
        y_test = feat_test["labels"]

        site = split_info.get("test_site", f"fold_{fold_idx}")
        n_train = len(y_train)
        n_test = len(y_test)
        n_td = int((y_test == 0).sum())
        n_asd = int((y_test == 1).sum())

        fused_auc = load_fused_baseline_auc(
            ckpt_dir / f"preds_fold_{fold_idx}.npz", y_test
        )

        print(f"  site={site}  train+val={n_train}  test={n_test} "
              f"(TD={n_td}, ASD={n_asd})  fused_auc={fmt(fused_auc)}")

        row = {
            "fold": fold_idx,
            "site": site,
            "n_train": n_train,
            "n_test": n_test,
            "n_td": n_td,
            "n_asd": n_asd,
            "auc_fused": fused_auc,
        }

        for rep in args.representations:
            X_train = np.concatenate(
                [feat_train[rep], feat_val[rep]], axis=0
            )
            X_test = feat_test[rep]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"  [SKIP] {rep}: una sola clase en train o test")
                row[rep_column(rep)] = float("nan")
                continue

            probs = fit_logistic_probe(
                X_train, y_train, X_test,
                C=args.C, max_iter=args.max_iter,
            )
            auc = float(roc_auc_score(y_test, probs))
            row[rep_column(rep)] = auc
            labels_per_site[rep].append(y_test)
            probs_per_site[rep].append(probs)
            print(f"  probe {rep:<10s} AUC={auc:.4f}  "
                  f"dims={X_train.shape[1]}  train={X_train.shape[0]}")

        rows.append(row)

        if args.save_features:
            train_payload = {
                rep: np.concatenate([feat_train[rep], feat_val[rep]], axis=0)
                for rep in args.representations
            }
            test_payload = {rep: feat_test[rep] for rep in args.representations}
            np.savez(
                feat_dir / f"fold_{fold_idx}_train.npz",
                labels=y_train,
                site=str(site),
                **train_payload,
            )
            np.savez(
                feat_dir / f"fold_{fold_idx}_test.npz",
                labels=y_test,
                site=str(site),
                **test_payload,
            )

        del exp, model, train_loader, val_loader, test_loader
        del train_eval_loader, feat_train, feat_val, feat_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("Sin folds válidos.")

    df = pd.DataFrame(rows)
    base_cols = ["fold", "site", "n_train", "n_test", "n_td", "n_asd"]
    rep_cols = [rep_column(rep) for rep in args.representations]
    df = df[base_cols + rep_cols + ["auc_fused"]]
    df = df.sort_values("fold").reset_index(drop=True)

    csv_out = out_dir / "linear_probe_h_ts_by_site.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nGuardado: {csv_out}")

    representations_summary = {}
    for rep in args.representations:
        aucs = df[rep_column(rep)].dropna().to_numpy(dtype=float)
        if aucs.size == 0:
            representations_summary[rep] = {
                "n_sites": 0,
                "auc_mean": None,
                "auc_std": None,
                "ci95_lo": None,
                "ci95_hi": None,
            }
            continue

        mean = float(aucs.mean())
        std = float(aucs.std(ddof=1)) if aucs.size >= 2 else 0.0
        boot = bootstrap_mean_auc(
            labels_per_site[rep], probs_per_site[rep],
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        if boot.size >= 2:
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            ci_lo, ci_hi = float(ci_lo), float(ci_hi)
        else:
            ci_lo, ci_hi = None, None

        representations_summary[rep] = {
            "n_sites": int(aucs.size),
            "auc_mean": mean,
            "auc_std": std,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        }
        print(f"Resumen {rep}: AUC={mean:.4f} ± {std:.4f} "
              f"(n={aucs.size})")

    baseline_aucs = df["auc_fused"].dropna().to_numpy(dtype=float)
    baseline_summary = {
        "n_sites": int(baseline_aucs.size),
        "auc_mean": float(baseline_aucs.mean()) if baseline_aucs.size else None,
        "auc_std": (float(baseline_aucs.std(ddof=1))
                    if baseline_aucs.size >= 2 else
                    (0.0 if baseline_aucs.size == 1 else None)),
    }

    per_site = []
    for record in df.to_dict(orient="records"):
        per_site.append({k: json_safe(v) for k, v in record.items()})

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "ckpt_contrastive": str(ckpt_contrastive),
        "protocol": args.eval_protocol,
        "encoder": "frozen",
        "classifier": "LogisticRegression",
        "C": float(args.C),
        "max_iter": int(args.max_iter),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "train_split": "train + val (18 sites, all subjects)",
        "test_split": "held-out site (LOSO)",
        "representations": {
            rep: {k: json_safe(v) for k, v in values.items()}
            for rep, values in representations_summary.items()
        },
        "fused_baseline": {
            k: json_safe(v) for k, v in baseline_summary.items()
        },
        "per_site": per_site,
    }

    json_out = out_dir / "linear_probe_h_ts_global.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Guardado: {json_out}")

    md = [
        "# Linear probe LOSO sobre h_ts",
        "",
        f"**Run:** `{run_dir}`",
        f"**Checkpoints:** `{ckpt_dir}`",
        f"**Protocolo:** {args.eval_protocol.upper()}",
        f"**Representaciones:** {', '.join(f'`{r}`' for r in args.representations)}",
        f"**Clasificador:** `LogisticRegression(C={args.C}, max_iter={args.max_iter})`",
        "",
        "## Metodología",
        "",
        "Para cada fold N (sitio held-out):",
        "",
        "1. Se reconstruye el modelo finetuneado N y se carga "
        "`best_finetune_fold_N.pt`.",
        "2. El modelo se congela (`eval` + `torch.inference_mode`): "
        "ningún peso del encoder se actualiza.",
        "3. Se extrae `h_ts` de **train + val** (todos los sujetos de los "
        "18 sitios) y del **test** (sujetos del sitio N).",
        "4. Se estandariza con train + val y se entrena una regresión "
        "logística simple.",
        "5. Se evalúa el AUC en el sitio held-out.",
        "",
        "`h_ts_raw` es la salida del Transformer temporal (512 dims), antes "
        "de la proyección contrastiva. `h_ts_proj` es la representación de "
        "128 dims que entra a la fusión y al clasificador auxiliar.",
        "",
        "## Resultados globales",
        "",
        "| Representación | AUC media | std | IC 95% | Sitios |",
        "|---|---|---|---|---|",
    ]

    for rep in args.representations:
        s = representations_summary[rep]
        md.append(
            f"| `{rep}` | {fmt(s['auc_mean'])} | {fmt(s['auc_std'])} | "
            f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}] | {s['n_sites']} |"
        )

    md.extend([
        "",
        f"**AUC fused del modelo completo (referencia):** "
        f"{fmt(baseline_summary['auc_mean'])} ± "
        f"{fmt(baseline_summary['auc_std'])} "
        f"({baseline_summary['n_sites']} sitios). "
        "No es un probe: es el MLP entrenado sobre la fusión.",
        "",
        "## Resultados por sitio",
        "",
    ])

    header = "| Fold | Sitio | train+val | test | TD | ASD |"
    separator = "|---|---|---|---|---|---|"
    for rep in args.representations:
        header += f" AUC `{rep}` |"
        separator += "---|"
    header += " AUC fused |"
    separator += "---|"
    md.append(header)
    md.append(separator)

    for _, r in df.iterrows():
        line = (f"| {int(r['fold'])} | {r['site']} | {int(r['n_train'])} | "
                f"{int(r['n_test'])} | {int(r['n_td'])} | {int(r['n_asd'])} |")
        for rep in args.representations:
            line += f" {fmt(float(r[rep_column(rep)]))} |"
        line += f" {fmt(float(r['auc_fused']))} |"
        md.append(line)

    md.extend([
        "",
        "## Lectura",
        "",
        "- Un AUC del probe claramente por encima de 0.5 indica que `h_ts` "
        "contiene información diagnóstica accesible con una frontera lineal.",
        "- Un AUC cercano a 0.5 indica que la señal no está disponible "
        "linealmente en esa representación, aunque el clasificador no lineal "
        "del modelo pudiera explotarla.",
        "- El probe se entrena solo con los 18 sitios de train+val y se evalúa "
        "siempre en el sitio excluido; el sitio held-out nunca participa en "
        "el ajuste de la logística.",
        "",
    ])

    md_out = out_dir / "linear_probe_h_ts_report.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    print(f"Guardado: {md_out}")


if __name__ == "__main__":
    main()
