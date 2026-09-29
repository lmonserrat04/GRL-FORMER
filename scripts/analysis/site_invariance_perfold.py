"""
Site-invariance test PER-FOLD.

A diferencia del pooled, aquí evaluamos el site classifier dentro de cada
fold por separado. Esto elimina el confound de mezclar features de modelos
distintos (cada fold tiene su propio modelo finetuneado).

Uso:
    python site_invariance_perfold.py <root1> [<root2> ...]
"""
# --- sys.path bootstrap (scripts movidos a subcarpetas) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data.loaders.dataloader import load_raw_data, get_finetune_loaders
from training.setup import build_experiment


SEED = 42
warnings.filterwarnings("ignore")


def find_base_experiment():
    exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
    for d in reversed(exp_dirs):
        p = Path(d)
        if (p / "checkpoints" / "contrastive_global.pt").exists():
            return p
    raise SystemExit("No hay experimento base")


def load_base_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["CKPT_TST1"] = str(exp_dir / "checkpoints" / "best_pt_ts_fold_0.pt")
    cfg["CKPT_TST2"] = str(exp_dir / "checkpoints" / "best_pt_fc_fold_0.pt")
    cfg["CKPT_CONTRASTIVE"] = str(exp_dir / "checkpoints" / "contrastive_global.pt")
    cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["EXPERIMENT_TYPE"] = "finetune"
    return cfg


def extract_fused_features(model, loader, device):
    captured = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            captured["fused"] = output[0].detach()
        else:
            captured["fused"] = output.detach()

    handle = model.fusion.register_forward_hook(hook)
    features = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            _ = model(ts, pcc, return_domain_logits=False)
            features.append(captured["fused"].cpu().numpy())
    handle.remove()
    return np.concatenate(features, axis=0)


def site_classifier_cv(features, sites, n_splits=5, seed=SEED):
    # Si hay menos de 2 sitios o algún sitio con <2 muestras, no se puede
    unique, counts = np.unique(sites, return_counts=True)
    if len(unique) < 2:
        return None
    min_count = int(counts.min())
    n_splits = min(n_splits, min_count)
    if n_splits < 2:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, baccs, f1s = [], [], []
    for tr, te in skf.split(X, sites):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X[tr], sites[tr])
        preds = clf.predict(X[te])
        accs.append(accuracy_score(sites[te], preds))
        baccs.append(balanced_accuracy_score(sites[te], preds))
        f1s.append(f1_score(sites[te], preds, average="macro"))

    return {
        "n_samples": len(features),
        "n_sites": len(unique),
        "random_baseline": 1.0 / len(unique),
        "accuracy": float(np.mean(accs)),
        "balanced_accuracy": float(np.mean(baccs)),
        "macro_f1": float(np.mean(f1s)),
        "n_splits_used": n_splits,
    }


def test_config(config_dir: Path, base_config: dict, data: dict):
    name = config_dir.name
    ckpt_dir = config_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("best_finetune_fold_*.pt"))

    if not ckpt_files:
        return None

    n_ckpts = len(ckpt_files)
    cfg = base_config.copy()
    cfg["CHECKPOINTS_PATH"] = str(ckpt_dir)
    cfg["EVAL_PROTOCOL"] = "loso" if n_ckpts > 10 else "kfold"
    cfg["N_FOLDS"] = n_ckpts

    print(f"\n{'='*70}")
    print(f"CONFIG: {name}  ({n_ckpts} folds, {cfg['EVAL_PROTOCOL']})")
    print(f"{'='*70}")

    fold_results = []
    for ckpt_path in ckpt_files:
        fold_idx = int(ckpt_path.stem.split("_")[-1])
        try:
            exp = build_experiment(
                cfg, fold_idx=fold_idx,
                ckpt_contrastive=cfg["CKPT_CONTRASTIVE"],
            )
        except Exception as e:
            print(f"  fold {fold_idx}: SKIP ({e})")
            continue

        model = exp.model
        state = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(exp.device)

        _, _, test_loader, split_info = get_finetune_loaders(
            cfg, batch_size=32, num_workers=0,
            fold_idx=fold_idx, n_folds=n_ckpts,
            seed=cfg["SEED"], eval_protocol=cfg["EVAL_PROTOCOL"],
        )

        feats = extract_fused_features(model, test_loader, exp.device)
        test_idx = split_info["test_idx"]
        sites = data["site_ids"][test_idx]

        res = site_classifier_cv(feats, sites)
        if res is None:
            print(f"  fold {fold_idx}: n={len(feats)} — no se puede CV (1 sitio o muy pocas muestras)")
            continue

        res["fold"] = fold_idx
        fold_results.append(res)
        print(f"  fold {fold_idx}: n={res['n_samples']:3d}  "
              f"n_sites={res['n_sites']:2d}  "
              f"site_acc={res['accuracy']:.4f}  "
              f"bacc={res['balanced_accuracy']:.4f}  "
              f"(random={res['random_baseline']:.3f})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fold_results:
        return {"config": name, "error": "sin folds válidos"}

    # Agregado sobre folds
    accs = [r["accuracy"] for r in fold_results]
    baccs = [r["balanced_accuracy"] for r in fold_results]

    agg = {
        "config": name,
        "protocol": cfg["EVAL_PROTOCOL"],
        "n_folds": len(fold_results),
        "site_acc_mean": float(np.mean(accs)),
        "site_acc_std": float(np.std(accs)),
        "site_acc_min": float(np.min(accs)),
        "site_acc_max": float(np.max(accs)),
        "site_bacc_mean": float(np.mean(baccs)),
        "site_bacc_std": float(np.std(baccs)),
        "per_fold": fold_results,
    }
    return agg


def main(root_dirs):
    base_exp = find_base_experiment()
    print(f"Experimento base: {base_exp}")
    base_config = load_base_config(base_exp)
    data = load_raw_data(base_config)
    print(f"Sujetos: {len(data['labels'])}, sitios: {len(data['site_to_idx'])}")

    all_results = []
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue
            ckpt_dir = subdir / "checkpoints"
            if not list(ckpt_dir.glob("best_finetune_fold_*.pt")):
                continue
            res = test_config(subdir, base_config, data)
            if res:
                all_results.append(res)

    print("\n" + "="*90)
    print("SITE-INVARIANCE PER-FOLD — comparativa")
    print("="*90)
    print(f"{'config':<35s}  {'site_acc':<20s}  {'site_bacc':<20s}  {'folds':<8s}")
    print("-"*90)
    for r in all_results:
        if "error" in r:
            print(f"{r.get('config','?'):<35s}  ERROR: {r['error']}")
            continue
        print(f"{r['config']:<35s}  "
              f"{r['site_acc_mean']:.4f} ± {r['site_acc_std']:.4f}     "
              f"{r['site_bacc_mean']:.4f} ± {r['site_bacc_std']:.4f}     "
              f"{r['n_folds']}")

    with open("site_invariance_perfold_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] Guardado en site_invariance_perfold_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    main(args.roots)
