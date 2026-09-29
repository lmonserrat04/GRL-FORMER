"""
Site-invariance test CONDICIONAL por clase.

Elimina el confound de la correlación label-site: evalúa la info de sitio
DENTRO de cada clase (control-only, ASD-only). Si un classifier entrenado
solo con controles puede predecir sitio, entonces las features retienen
info de sitio "pura" (no explicable por label).

Uso:
    python site_invariance_conditional.py <root1> [<root2> ...]
"""
import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data.loaders.dataloader import load_raw_data, get_finetune_loaders
from training.setup import build_experiment


SEED = 42
KNN_K = 5
warnings.filterwarnings("ignore")


# ─── Utils ──────────────────────────────────────────────────────────
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


# ─── Métricas de site-invarianza ────────────────────────────────────
def knn_site_consistency(features, sites, k=KNN_K):
    """
    Fracción de vecinos (k más cercanos) del mismo sitio.
    Rango: [1/n_sites, 1.0]. Random baseline = 1/n_sites.
    """
    n = len(features)
    if n < k + 1:
        return None

    # Normalizar L2 para que cosine distance sea estable
    norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    X = features / norm

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # indices[:, 0] = self, excluirlo
    neighbors = indices[:, 1:]
    same = (sites[neighbors] == sites[:, None]).astype(float)
    return float(same.mean())


def site_classifier_cv(features, sites, n_splits=5, seed=SEED):
    """
    LogReg multiclase con StratifiedKFold. Si algún sitio tiene <2 muestras,
    devuelve None.
    """
    unique, counts = np.unique(sites, return_counts=True)
    if len(unique) < 2:
        return None
    min_count = int(counts.min())
    if min_count < 2:
        return None
    n_splits = min(n_splits, min_count)
    if n_splits < 2:
        return None

    X = StandardScaler().fit_transform(features)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, baccs = [], []
    for tr, te in skf.split(X, sites):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X[tr], sites[tr])
        preds = clf.predict(X[te])
        accs.append(accuracy_score(sites[te], preds))
        baccs.append(balanced_accuracy_score(sites[te], preds))

    return {
        "accuracy": float(np.mean(accs)),
        "balanced_accuracy": float(np.mean(baccs)),
        "n_samples": int(len(features)),
        "n_sites": int(len(unique)),
        "random_baseline": float(1.0 / len(unique)),
        "n_splits_used": n_splits,
    }


# ─── Test por config ────────────────────────────────────────────────
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

    print(f"\n{'='*80}")
    print(f"CONFIG: {name}  ({n_ckpts} folds, {cfg['EVAL_PROTOCOL']})")
    print(f"{'='*80}")

    per_fold = []

    for ckpt_path in ckpt_files:
        fold_idx = int(ckpt_path.stem.split("_")[-1])
        try:
            exp = build_experiment(cfg, fold_idx=fold_idx,
                                   ckpt_contrastive=cfg["CKPT_CONTRASTIVE"])
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
        labels = data["labels"][test_idx]

        # ─── Evaluación condicional por clase ──────────────────────
        fold_res = {"fold": fold_idx, "n_total": len(feats)}
        for cls, cls_name in [(0, "control"), (1, "asd")]:
            mask = labels == cls
            if mask.sum() < KNN_K + 1:
                continue
            f_c = feats[mask]
            s_c = sites[mask]

            knn = knn_site_consistency(f_c, s_c, k=KNN_K)
            lr = site_classifier_cv(f_c, s_c)

            fold_res[f"n_{cls_name}"] = int(mask.sum())
            fold_res[f"knn_{cls_name}"] = knn
            fold_res[f"lr_acc_{cls_name}"] = lr["accuracy"] if lr else None
            fold_res[f"lr_bacc_{cls_name}"] = lr["balanced_accuracy"] if lr else None
            fold_res[f"n_sites_{cls_name}"] = lr["n_sites"] if lr else None

        per_fold.append(fold_res)
        knn_c = fold_res.get("knn_control")
        knn_a = fold_res.get("knn_asd")
        print(f"  fold {fold_idx}: n_ctrl={fold_res.get('n_control', 0):3d} "
              f"knn_ctrl={knn_c:.4f}  |  "
              f"n_asd={fold_res.get('n_asd', 0):3d} knn_asd={knn_a:.4f}"
              if knn_c and knn_a else
              f"  fold {fold_idx}: (faltan muestras)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not per_fold:
        return {"config": name, "error": "sin folds válidos"}

    # ─── Agregado ─────────────────────────────────────────────────
    def agg(key):
        vals = [f[key] for f in per_fold if f.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }

    summary = {
        "config": name,
        "protocol": cfg["EVAL_PROTOCOL"],
        "n_folds": len(per_fold),
        "knn_control": agg("knn_control"),
        "knn_asd": agg("knn_asd"),
        "lr_acc_control": agg("lr_acc_control"),
        "lr_acc_asd": agg("lr_acc_asd"),
        "per_fold": per_fold,
    }
    return summary


# ─── Main ───────────────────────────────────────────────────────────
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
            if not list((subdir / "checkpoints").glob("best_finetune_fold_*.pt")):
                continue
            res = test_config(subdir, base_config, data)
            if res:
                all_results.append(res)

    print("\n" + "="*100)
    print("SITE-INVARIANCE CONDICIONAL — comparativa")
    print("="*100)
    print(f"{'config':<35s}  {'knn_ctrl':<16s}  {'knn_asd':<16s}  "
          f"{'lr_acc_ctrl':<16s}  {'lr_acc_asd':<16s}")
    print("-"*100)
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<35s}  ERROR: {r['error']}")
            continue
        kc = r.get("knn_control"); ka = r.get("knn_asd")
        lc = r.get("lr_acc_control"); la = r.get("lr_acc_asd")
        def fmt(d):
            if d is None: return "N/A"
            return f"{d['mean']:.4f}±{d['std']:.4f}"
        print(f"{r['config']:<35s}  {fmt(kc):<16s}  {fmt(ka):<16s}  "
              f"{fmt(lc):<16s}  {fmt(la):<16s}")

    print("\nRandom baseline KNN = 1/n_sites ≈ 0.053")
    print("Interpretación: si GRL reduce knn_* y lr_acc_* vs no_grl → invarianza real")

    with open("site_invariance_conditional_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\n[OK] site_invariance_conditional_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    main(args.roots)
