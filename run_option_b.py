"""Protocolo del paper Table 3: single split 70/10/20 × 5 seeds.
Reutiliza pretrains del último experimento. Re-corre contrastive + finetune por seed."""
import yaml, json, glob, numpy as np, torch, shutil
from pathlib import Path
from datetime import datetime

# Localizar pretrains
candidates = sorted(glob.glob("experiments/*/checkpoints/best_pt_ts_fold_0.pt"))
src_ckpt = Path(candidates[-1]).parent
src_exp = src_ckpt.parent
print(f"Pretrains de: {src_exp}")

with open(src_exp / "config.yaml") as f:
    config = yaml.safe_load(f)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
new_dir = Path(f"experiments/{ts}_option_b_single_split_5seeds")
(new_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(new_dir / "logs").mkdir(parents=True, exist_ok=True)
shutil.copy(src_exp / "config.yaml", new_dir / "config.yaml")

config["EXP_DIR"] = str(new_dir)
config["CHECKPOINTS_PATH"] = str(new_dir / "checkpoints")
config["LOGS_PATH"] = str(new_dir / "logs")
config["EXP_ID"] = new_dir.name

# Copiar pretrains
for name in ["best_pt_ts_fold_0.pt", "best_pt_fc_fold_0.pt"]:
    shutil.copy(src_ckpt / name, new_dir / "checkpoints" / name)
config["CKPT_TST1"] = str(new_dir / "checkpoints" / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(new_dir / "checkpoints" / "best_pt_fc_fold_0.pt")

# Kfold del paper NO: usamos single split. El contrastive también usará single split.
from training.train_contrastive import run_contrastive_global
from training.train_finetune import finetune_fold
from data.loaders.dataloader import get_single_split_loaders
from utils.metrics import bootstrap_confidence_interval, get_reproducibility_info

SEEDS = [42, 123, 2024, 7, 999]
metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
all_seed_metrics = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed); np.random.seed(seed)
    cfg = dict(config)
    cfg["SEED"] = seed
    ckpt_dir = Path(cfg["CHECKPOINTS_PATH"]) / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg["CHECKPOINTS_PATH"] = str(ckpt_dir)

    # Contrastive con este seed (split 70/10/20 distinto)
    print(f"\n>>> Contrastive (seed={seed})")
    run_contrastive_global(cfg, save_dir=ckpt_dir)
    cfg["CKPT_CONTRASTIVE"] = str(ckpt_dir / "contrastive_global.pt")

    # Finetune sobre el split único de este seed
    # Monkeypatch: hacemos que finetune_fold use el split único del seed,
    # no K-fold. Usamos un fold_idx ficticio y forzamos el splitter.
    print(f"\n>>> Finetune (seed={seed}, split 70/10/20)")

    # Truco: crear los loaders directamente con single split
    from training.train_finetune import train_epoch, validate, _load_projection_heads
    from models.dual_stream import create_dual_stream_model
    import torch.nn as nn
    from tqdm import tqdm

    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    phase = cfg["FINETUNING"]; ds = cfg["DUAL_STREAM"]
    fh = cfg.get("FUSION", {}).get("ATTENTION_POOLING", {}).get("HIDDEN_DIM")

    tr, va, te, split_info = get_single_split_loaders(
        cfg, batch_size=phase["BATCH_SIZE"],
        num_workers=cfg.get("NUM_WORKERS", 0), seed=seed,
    )

    p1, p2 = _load_projection_heads(cfg, device)
    model = create_dual_stream_model(
        n_rois=cfg["N_ROIS"], time_points=cfg["MAX_SEQ_LEN"],
        pcc_dim=cfg["TST2"]["PCC_DIM"],
        tst1_emb_dim=cfg["TST1"]["D_MODEL"], tst2_d_model=cfg["TST2"]["D_MODEL"],
        fusion_type=ds["FUSION_TYPE"], fusion_hidden_dim=fh,
        num_classes=ds["NUM_CLASSES"], dropout=ds["CLASSIFIER_DROPOUT"],
        mlp_dims=ds.get("MLP_DIMS"), proj_head_1=p1, proj_head_2=p2,
    ).to(device)
    model.load_pretrained_tst1(cfg["CKPT_TST1"], strict=False)
    model.load_pretrained_tst2(cfg["CKPT_TST2"], strict=False)
    model.unfreeze_encoders()
    for p in model.proj_head_1.parameters(): p.requires_grad = False
    for p in model.proj_head_2.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=float(phase["LR"]), weight_decay=float(phase["WEIGHT_DECAY"]))
    epochs = phase["N_EPOCHS"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                            eta_min=float(phase["LR"])*0.01)
    criterion = nn.CrossEntropyLoss()

    patience = phase.get("PATIENCE", 20)
    best_auc, best_state, pc = -1.0, None, 0
    with tqdm(range(1, epochs + 1), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"[seed {seed}] Epoch {epoch}")
            tl = train_epoch(model, tr, optimizer, criterion, device)
            vl, vm = validate(model, va, criterion, device)
            scheduler.step()
            auc = vm["auc"]
            tepoch.set_postfix(train=f"{tl:.4f}", val=f"{vl:.4f}", auc=f"{auc:.4f}")
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pc = 0
            else:
                pc += 1
                if pc >= patience:
                    tqdm.write(f"  ⏹ Early stop epoch {epoch} (best AUC={best_auc:.4f})")
                    break
    if best_state: model.load_state_dict(best_state)

    # Test
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in te:
            ts_b = batch["timeseries"].to(device)
            pcc_b = batch["pcc_vector"].to(device)
            y = batch["label"].to(device)
            logits = model(ts_b, pcc_b)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    # Subject-level si aplica (con sliding window sería >1, aquí es 1 por sujeto)
    from utils.metrics import compute_metrics, aggregate_window_predictions_to_subject_level
    si = split_info["subject_indices"]; ti = split_info["test_idx"]
    if len(ti) > len(np.unique(si[ti])):
        yt, yp, ypr = aggregate_window_predictions_to_subject_level(
            labels, preds, probs, ti, si, strategy=cfg.get("SUBJECT_AGG", "majority_vote"))
    else:
        yt, yp, ypr = np.array(labels), np.array(preds), np.array(probs)

    m = compute_metrics(yt, yp, ypr)
    m["seed"] = seed
    all_seed_metrics.append(m)
    print(f"  seed {seed}: AUC={m['auc']:.4f} ACC={m['accuracy']:.4f} "
          f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f} F1={m['f1']:.4f}")

# Resumen
summary = {"protocol": "single_split_5seeds", "seeds": SEEDS, "exp_id": new_dir.name}
print(f"\n{'='*60}")
print(f"RESUMEN — single split × 5 seeds (mean ± std [95% CI])")
print(f"{'='*60}")
for name in metric_names:
    vals = [m[name] for m in all_seed_metrics]
    mean_v, std_v, lo, hi = bootstrap_confidence_interval(vals, n_bootstrap=1000, ci=0.95, seed=42)
    summary[name] = {"mean": mean_v, "std": std_v, "ci95_lower": lo, "ci95_upper": hi}
    label = name.upper() if name == "auc" else name.capitalize()
    print(f"  {label:12s}: {mean_v:.4f} ± {std_v:.4f}  [{lo:.4f}, {hi:.4f}]")

summary["all_seeds"] = all_seed_metrics
summary["reproducibility"] = get_reproducibility_info()

with open(new_dir / "results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
print(f"\n✅ {new_dir / 'results.json'}")
