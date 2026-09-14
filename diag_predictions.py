"""Diagnóstico: distribución de predicciones por seed."""
import yaml, glob, numpy as np, torch
from pathlib import Path
from training.train_finetune import _load_projection_heads
from models.dual_stream import create_dual_stream_model
from data.loaders.dataloader import get_single_split_loaders
from sklearn.metrics import confusion_matrix, roc_auc_score

# Última corrida Option B
exp_dirs = sorted(glob.glob("experiments/*_option_b_single_split_5seeds"))
if not exp_dirs:
    raise SystemExit("No hay corrida Option B")
exp_dir = Path(exp_dirs[-1])
print(f"Diagnóstico de: {exp_dir}")

with open(exp_dir / "config.yaml") as f:
    config = yaml.safe_load(f)
config["EXP_DIR"] = str(exp_dir)
config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")

config["CKPT_TST1"] = str(exp_dir / "checkpoints" / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(exp_dir / "checkpoints" / "best_pt_fc_fold_0.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024, 7, 999]

for seed in SEEDS:
    seed_dir = exp_dir / "checkpoints" / f"seed_{seed}"
    ckpt = seed_dir / "contrastive_global.pt"
    if not ckpt.exists():
        print(f"seed {seed}: no hay checkpoint, salto")
        continue

    config["CKPT_CONTRASTIVE"] = str(ckpt)
    config["SEED"] = seed

    # Cargar modelo finetuneado
    ft_path = seed_dir / "finetune.pt"
    if not ft_path.exists():
        print(f"seed {seed}: falta finetune.pt, salto")
        continue

    p1, p2 = _load_projection_heads(config, device)
    ds = config["DUAL_STREAM"]
    fh = config.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")
    model = create_dual_stream_model(
        n_rois=config["N_ROIS"], time_points=config["MAX_SEQ_LEN"],
        pcc_dim=config["TST2"]["PCC_DIM"],
        tst1_emb_dim=config["TST1"]["D_MODEL"], tst2_d_model=config["TST2"]["D_MODEL"],
        fusion_type=ds["FUSION_TYPE"], fusion_hidden_dim=fh,
        num_classes=ds["NUM_CLASSES"], dropout=ds["CLASSIFIER_DROPOUT"],
        mlp_dims=ds.get("MLP_DIMS"), proj_head_1=p1, proj_head_2=p2,
    ).to(device)
    blob = torch.load(ft_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    # Split del seed
    _, _, test_loader, _ = get_single_split_loaders(
        config, batch_size=32, num_workers=0, seed=seed)
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            logits = model(ts, pcc)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    labels = np.array(labels); preds = np.array(preds); probs = np.array(probs)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print(f"\nseed {seed}: labels={np.bincount(labels, minlength=2).tolist()} "
          f"preds={np.bincount(preds, minlength=2).tolist()}")
    print(f"  probs: mean={probs.mean():.3f} std={probs.std():.3f} "
          f"min={probs.min():.3f} max={probs.max():.3f}")
    print(f"  confusion: {cm.tolist()}")
    if len(np.unique(labels)) > 1:
        print(f"  AUC (sanity): {roc_auc_score(labels, probs):.4f}")
