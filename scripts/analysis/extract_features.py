# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Extrae representaciones internas de un run finetuneado.

Para cada fold:
  1. Construye el modelo con build_experiment (misma arquitectura)
  2. Carga best_finetune_fold_N.pt encima
  3. Corre inferencia sobre el test_loader
  4. Guarda features_fold_N.npz con:
       h_ts_raw   (N, 512)  encoder TST1
       h_ts_proj  (N, 128)  después de proj_head_1
       h_fc_raw   (N, 256)  encoder TST2
       h_fc_proj  (N, 128)  después de proj_head_2
       fused      (N, 128)  salida de la fusión
       logits     (N, 2)
       probs      (N,)
       labels     (N,)
       sub_ids    (N,)      SUB_ID de cada muestra
       optimal_thr (1,)     umbral de Youden del fold

Uso:
    python scripts/analysis/extract_features.py \
        --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds

Asume que el run tiene:
  - config.yaml  (arquitectura + CKPT_CONTRASTIVE)
  - checkpoints/best_finetune_fold_N.pt para cada fold
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from training.setup import build_experiment
from data.loaders.dataloader import get_finetune_loaders


def extract_fold(config: dict, fold_idx: int, ckpt_path: Path, device: str) -> dict:
    """Extrae features de un fold."""
    exp = build_experiment(
        config, fold_idx=fold_idx,
        ckpt_contrastive=config.get("CKPT_CONTRASTIVE"),
    )
    model = exp.model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {len(unexpected)}")

    model.to(device).eval()

    phase = config["FINETUNING"]
    _, _, test_loader, split_info = get_finetune_loaders(
        config, batch_size=phase["BATCH_SIZE"],
        num_workers=config.get("NUM_WORKERS", 0),
        fold_idx=fold_idx, n_folds=config.get("N_FOLDS", 5),
        seed=config.get("SEED", 42),
        eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
    )

    h_ts_raw_all   = []
    h_ts_proj_all  = []
    h_fc_raw_all   = []
    h_fc_proj_all  = []
    fused_all      = []
    logits_all     = []
    probs_all      = []
    labels_all     = []

    with torch.no_grad():
        for batch in test_loader:
            ts  = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y   = batch["label"].to(device)

            h_ts_raw = model.transformer_ts(ts,  mode='finetune')
            h_fc_raw = model.transformer_fc(pcc, mode='finetune')

            h_ts_proj = (model.proj_head_1(h_ts_raw)
                         if model.proj_head_1 is not None else h_ts_raw)
            h_fc_proj = (model.proj_head_2(h_fc_raw)
                         if model.proj_head_2 is not None else h_fc_raw)

            fused  = model.fusion(h_ts_proj, h_fc_proj)
            logits = model.classifier(fused)
            probs  = torch.softmax(logits, dim=1)[:, 1]

            h_ts_raw_all.append(h_ts_raw.cpu().numpy())
            h_ts_proj_all.append(h_ts_proj.cpu().numpy())
            h_fc_raw_all.append(h_fc_raw.cpu().numpy())
            h_fc_proj_all.append(h_fc_proj.cpu().numpy())
            fused_all.append(fused.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            labels_all.append(y.cpu().numpy())

    si = split_info["subject_indices"]
    ti = split_info["test_idx"]
    sub_ids = np.asarray(si)[np.asarray(ti)]

    return {
        "h_ts_raw":   np.concatenate(h_ts_raw_all,   axis=0).astype(np.float32),
        "h_ts_proj":  np.concatenate(h_ts_proj_all,  axis=0).astype(np.float32),
        "h_fc_raw":   np.concatenate(h_fc_raw_all,   axis=0).astype(np.float32),
        "h_fc_proj":  np.concatenate(h_fc_proj_all,  axis=0).astype(np.float32),
        "fused":      np.concatenate(fused_all,      axis=0).astype(np.float32),
        "logits":     np.concatenate(logits_all,     axis=0).astype(np.float32),
        "probs":      np.concatenate(probs_all,      axis=0).astype(np.float32),
        "labels":     np.concatenate(labels_all,     axis=0).astype(np.int64),
        "sub_ids":    sub_ids.astype(np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/features/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"No existe {run_dir}")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"No existe {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["EXPERIMENT_TYPE"] = "finetune"

    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("best_finetune_fold_*.pt"))
    if not ckpts:
        raise SystemExit(f"No hay best_finetune_fold_*.pt en {ckpt_dir}")

    out_dir = args.out_dir if args.out_dir else (run_dir / "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run:       {run_dir}")
    print(f"Device:    {args.device}")
    print(f"Checkpoints: {len(ckpts)}")
    print(f"Output:    {out_dir}\n")

    folds_to_run = sorted([int(c.stem.split("_")[-1]) for c in ckpts])
    if args.folds is not None:
        folds_to_run = [f for f in folds_to_run if f in args.folds]

    for fold_idx in folds_to_run:
        ckpt_path = ckpt_dir / f"best_finetune_fold_{fold_idx}.pt"
        out_path = out_dir / f"features_fold_{fold_idx}.npz"
        if out_path.exists():
            print(f"[SKIP] fold {fold_idx}: {out_path.name} ya existe")
            continue

        print(f"[RUN] fold {fold_idx}")
        feats = extract_fold(config, fold_idx, ckpt_path, args.device)

        npz_preds = ckpt_dir / f"preds_fold_{fold_idx}.npz"
        if npz_preds.exists():
            npz = np.load(npz_preds, allow_pickle=True)
            feats["optimal_thr"] = npz["optimal_thr"]
        else:
            feats["optimal_thr"] = np.array([0.5])

        np.savez(out_path, **feats)
        print(f"  → {out_path.name}  "
              f"ts_raw={feats['h_ts_raw'].shape}  "
              f"fused={feats['fused'].shape}")

    print("\n[OK] extracción completada")


if __name__ == "__main__":
    main()
