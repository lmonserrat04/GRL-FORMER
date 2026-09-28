"""
Site-invariance test.

Compara la información de sitio contenida en las features 'fused' de
distintos modelos finetuneados. Si la GRL funciona, el site classifier
entrenado sobre features GRL debería tener PEOR accuracy que sobre
features sin GRL.

Uso:
    python site_invariance_test.py <root1> [<root2> ...]

Cada root debe contener subdirectorios <config>/checkpoints/best_finetune_fold_*.pt
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data.loaders.dataloader import load_raw_data, get_finetune_loaders
from training.setup import build_experiment
from training.train_finetune import finetune_fold


# ─── Configuración ──────────────────────────────────────────────────
SEED = 42
N_CV = 5
warnings.filterwarnings("ignore")


# ─── Localizar experimento base para config ─────────────────────────
def find_base_experiment():
    """Busca un experimento con contrastive_global.pt para reutilizar."""
    exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
    for d in reversed(exp_dirs):
        p = Path(d)
        if (p / "checkpoints" / "contrastive_global.pt").exists():
            return p
    raise SystemExit("No hay experimento base con contrastive_global.pt")


def load_base_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["CKPT_TST1"] = str(exp_dir / "checkpoints" / "best_pt_ts_fold_0.pt")
    cfg["CKPT_TST2"] = str(exp_dir / "checkpoints" / "best_pt_fc_fold_0.pt")
    cfg["CKPT_CONTRASTIVE"] = str(exp_dir / "checkpoints" / "contrastive_global.pt")
    cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["N_FOLDS"] = 5
    cfg["EVAL_PROTOCOL"] = "kfold"
    return cfg


# ─── Extracción de features ─────────────────────────────────────────
def extract_fused_features(model, loader, device):
    """Extrae features 'fused' capturándolas con un hook en model.fusion."""
    captured = {}

    def hook(module, input, output):
        # fused es el primer tensor de salida de la fusión
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


# ─── Classifier de sitio con CV ─────────────────────────────────────
def site_classifier_cv(features: np.ndarray, sites: np.ndarray,
                       n_splits: int = N_CV, seed: int = SEED) -> dict:
    """
    Entrena LogisticRegression multiclase con StratifiedKFold.
    Devuelve accuracy, balanced accuracy y macro F1.
    """
    # Normalizar features (importante para LogReg)
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Reducir n_splits si algún sitio tiene menos muestras
    min_class_count = min(
        (sites == s).sum() for s in np.unique(sites)
    )
    n_splits = min(n_splits, int(min_class_count))
    if n_splits < 2:
        return {"error": f"muy pocos sujetos por sitio (min={min_class_count})"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, baccs, f1s = [], [], []
    for tr_idx, te_idx in skf.split(features_norm, sites):
        clf = LogisticRegression(
            max_iter=2000, multi_class="multinomial",
            C=1.0, class_weight="balanced",
        )
        clf.fit(features_norm[tr_idx], sites[tr_idx])
        preds = clf.predict(features_norm[te_idx])

        accs.append(accuracy_score(sites[te_idx], preds))
        baccs.append(balanced_accuracy_score(sites[te_idx], preds))
        f1s.append(f1_score(sites[te_idx], preds, average="macro"))

    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_splits": n_splits,
        "n_samples": int(len(features)),
        "n_sites": int(len(np.unique(sites))),
        "random_baseline": float(1.0 / len(np.unique(sites))),
    }


# ─── Test por config ────────────────────────────────────────────────
def test_config(config_dir: Path, base_config: dict, data: dict) -> dict:
    """
    Carga los modelos finetuneados de una config, extrae features 'fused'
    del test set de cada fold, entrena site classifier y reporta.
    """
    name = config_dir.name
    ckpt_dir = config_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("best_finetune_fold_*.pt"))

    if not ckpt_files:
        return {"config": name, "error": "no hay best_finetune_fold_*.pt"}

    print(f"\n{'='*70}")
    print(f"CONFIG: {name}  ({len(ckpt_files)} folds)")
    print(f"{'='*70}")

    cfg = base_config.copy()
    cfg["EXPERIMENT_TYPE"] = "finetune"
    cfg["CHECKPOINTS_PATH"] = str(ckpt_dir)

    # Detectar protocolo por número de checkpoints
    n_ckpts = len(ckpt_files)
    if n_ckpts > 10:
        cfg["EVAL_PROTOCOL"] = "loso"
        cfg["N_FOLDS"] = n_ckpts  # loso usa un fold por sitio
        print(f"  detectado LOSO ({n_ckpts} folds)")
    else:
        cfg["EVAL_PROTOCOL"] = "kfold"
        cfg["N_FOLDS"] = n_ckpts
        print(f"  detectado K-fold ({n_ckpts} folds)")

    all_features, all_sites, all_labels = [], [], []

    for ckpt_path in ckpt_files:
        fold_idx = int(ckpt_path.stem.split("_")[-1])
        print(f"  fold {fold_idx}: extrayendo features...")

        exp = build_experiment(
            cfg, fold_idx=fold_idx,
            ckpt_contrastive=cfg["CKPT_CONTRASTIVE"],
        )
        model = exp.model
        state = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(exp.device)

        _, _, test_loader, split_info = get_finetune_loaders(
            cfg, batch_size=32, num_workers=0,
            fold_idx=fold_idx, n_folds=cfg["N_FOLDS"],
            seed=cfg["SEED"], eval_protocol="kfold",
        )

        feats = extract_fused_features(model, test_loader, exp.device)
        test_idx = split_info["test_idx"]
        sites = data["site_ids"][test_idx]

        all_features.append(feats)
        all_sites.append(sites)
        all_labels.append(data["labels"][test_idx])

        # Liberar GPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    features = np.concatenate(all_features, axis=0)
    sites = np.concatenate(all_sites, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"  features totales: {features.shape}")

    result = site_classifier_cv(features, sites)
    result["config"] = name
    return result


# ─── Main ───────────────────────────────────────────────────────────
def main(root_dirs):
    base_exp = find_base_experiment()
    print(f"Experimento base: {base_exp}")

    base_config = load_base_config(base_exp)
    data = load_raw_data(base_config)
    print(f"Sujetos totales: {len(data['labels'])}, "
          f"sitios únicos: {len(data['site_to_idx'])}")

    all_results = []
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[SKIP] no existe {root_path}")
            continue
        for subdir in sorted(root_path.iterdir()):
            if not subdir.is_dir():
                continue
            ckpt_dir = subdir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            if not list(ckpt_dir.glob("best_finetune_fold_*.pt")):
                continue
            res = test_config(subdir, base_config, data)
            all_results.append(res)

    # ─── Reporte final ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("SITE-INVARIANCE TEST — resultado")
    print("="*80)
    print(f"{'config':<35s}  {'site_acc':<12s}  {'site_bacc':<12s}  {'site_f1':<12s}")
    print("-"*80)
    for r in all_results:
        if "error" in r:
            print(f"{r.get('config','?'):<35s}  ERROR: {r['error']}")
            continue
        print(f"{r['config']:<35s}  "
              f"{r['accuracy']:.4f}       "
              f"{r['balanced_accuracy']:.4f}       "
              f"{r['macro_f1']:.4f}")

    print()
    print("Interpretación:")
    print("  - Si el modelo CON GRL tiene site_acc MENOR que sin GRL →")
    print("    la GRL logró features invariantes al sitio.")
    print("  - El random baseline es 1/n_sites.")
    for r in all_results:
        if "error" not in r:
            print(f"    {r['config']}: random={r['random_baseline']:.4f} "
                  f"(n_sites={r['n_sites']}, n_samples={r['n_samples']})")

    out = Path("site_invariance_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] Guardado en {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="Directorios a testear")
    args = parser.parse_args()
    main(args.roots)
