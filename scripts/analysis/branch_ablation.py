# --- sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

"""
Branch ablation sobre encoders congelados (TwoTST, LOSO).

Para cada fold LOSO N:
  1. Se reconstruye el modelo del fold N y se carga best_finetune_fold_N.pt.
     Los encoders quedan CONGELADOS (no se reentrenan).
  2. Se extraen features congeladas para train_N, val_N, test_N:
       h_ts_proj = proj_head_1(transformer_ts(ts))      (128d)
       h_fc_proj = proj_head_2(transformer_fc(pcc))     (128d)
       fused     = fusion(h_ts_proj, h_fc_proj)         (128d)
  3. Se entrenan 3 clasificadores nuevos (misma arquitectura y
     hiperparámetros que tag_classifier + FINETUNING):
       A (temporal)         input = h_ts_proj
       B (connectivity)     input = h_fc_proj
       C (fused frozen)     input = fused     [control]
     Early stopping en val_N AUC.
  4. Se evalúan A, B, C sobre test_N.
  5. Se reporta también el modelo original (joint training) desde
     preds_fold_N.npz como referencia D.

Sin fuga de información:
  - Los encoders solo vieron train_N durante el finetune original.
  - Los clasificadores nuevos solo se entrenan con features de train_N.
  - Early stopping con val_N (sitos ajenos al test_N).
  - test_N nunca se usa para entrenar ni para early stopping.

Salida (no sobrescribe nada del run original):
  <run_dir>/branch_ablation/fold_N.json
  <run_dir>/branch_ablation/branch_ablation_by_site.csv
  <run_dir>/branch_ablation/branch_ablation_global.json
  <run_dir>/branch_ablation/branch_ablation_report.md

Uso:
  python scripts/analysis/branch_ablation.py \
      --run-dir experiments_finetune_only/20261001_233905_twotst_original_loso_preds

  # Solo folds específicos:
  python scripts/analysis/branch_ablation.py --run-dir <dir> --folds 5 9
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

from training.setup import build_experiment
from data.loaders.dataloader import get_finetune_loaders
from models.mlp_head import create_mlp_head


def _set_seed(seed: int):
    """Reproducibilidad: intenta utils.seed.set_seed; si no, fallback."""
    try:
        from utils.seed import set_seed
        set_seed(seed)
    except Exception:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def extract_features_from_loader(model, loader, device):
    """
    Pasa un loader por los encoders congelados del modelo y devuelve:
      - h_ts_proj  (N, 128)  proj_head_1(transformer_ts(ts))
      - h_fc_proj  (N, 128)  proj_head_2(transformer_fc(pcc))
      - fused      (N, 128)  fusion(h_ts_proj, h_fc_proj)
      - labels     (N,)
    """
    model.eval()
    h_ts_list, h_fc_list, fused_list, labels_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ts  = batch["timeseries"].to(device)
            pcc = batch["pcc_vector"].to(device)
            y   = batch["label"].to(device)

            h_ts_raw = model.transformer_ts(ts,  mode='finetune')
            h_fc_raw = model.transformer_fc(pcc, mode='finetune')

            h_ts_proj = (model.proj_head_1(h_ts_raw)
                         if model.proj_head_1 is not None else h_ts_raw)
            h_fc_proj = (model.proj_head_2(h_fc_raw)
                         if model.proj_head_2 is not None else h_fc_raw)

            fused = model.fusion(h_ts_proj, h_fc_proj)

            h_ts_list.append(h_ts_proj.cpu().numpy())
            h_fc_list.append(h_fc_proj.cpu().numpy())
            fused_list.append(fused.cpu().numpy())
            labels_list.append(y.cpu().numpy())

    return {
        "h_ts_proj": np.concatenate(h_ts_list, axis=0).astype(np.float32),
        "h_fc_proj": np.concatenate(h_fc_list, axis=0).astype(np.float32),
        "fused":     np.concatenate(fused_list, axis=0).astype(np.float32),
        "labels":    np.concatenate(labels_list, axis=0).astype(np.int64),
    }


def train_branch_classifier(X_train, y_train, X_val, y_val, config, seed=42):
    """
    Entrena un MLP con la misma arquitectura y hiperparámetros que
    el tag_classifier del finetune original.

    Retorna (model, best_val_auc, best_epoch).
    """
    _set_seed(seed)

    input_dim = X_train.shape[1]
    mlp_dims = list(config["MLP_HEAD"]["MLP_DIMS"])   # [256, 64, 2]
    dropout  = float(config["MLP_HEAD"]["DROPOUT"])
    fin      = config["FINETUNING"]

    model = create_mlp_head([input_dim] + mlp_dims, dropout, act_name='gelu')
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(fin["LR"]),
        weight_decay=float(fin.get("WEIGHT_DECAY", 0.0)),
    )
    sched_params = fin.get("SCHEDULER_PARAMS", {}) or {}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(sched_params.get("T_max", fin["N_EPOCHS"])),
        eta_min=float(sched_params.get("eta_min", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t   = torch.from_numpy(X_val)

    n_train    = len(X_train_t)
    batch_size = int(fin["BATCH_SIZE"])
    epochs     = int(fin["N_EPOCHS"])
    patience   = int(fin.get("PATIENCE", 20))

    best_val_auc = -1.0
    best_state   = None
    best_epoch   = 0
    pc           = 0

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = idx[i:i + batch_size]
            if len(batch_idx) < 2:
                continue
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].numpy()

        if len(np.unique(y_val)) > 1:
            val_auc = float(roc_auc_score(y_val, val_probs))
        else:
            val_auc = 0.5

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val_auc, best_epoch


def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return torch.softmax(logits, dim=1)[:, 1].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("data/csv/data_train.csv"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/branch_ablation/")
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

    for key in ("MLP_HEAD", "FINETUNING"):
        if key not in config:
            raise SystemExit(f"Falta {key} en config")

    out_dir = args.out_dir if args.out_dir else (run_dir / "branch_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_csv = pd.read_csv(args.csv)
    sub_to_site = dict(zip(df_csv["SUB_ID"].astype(int),
                            df_csv["SITE_ID"].astype(str)))

    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("best_finetune_fold_*.pt"))
    if not ckpts:
        raise SystemExit(f"No hay best_finetune_fold_*.pt en {ckpt_dir}")

    folds = sorted([int(c.stem.split("_")[-1]) for c in ckpts])
    if args.folds is not None:
        folds = [f for f in folds if f in args.folds]

    seed = int(config.get("SEED", 42))
    phase = config["FINETUNING"]

    print(f"Run:     {run_dir}")
    print(f"Device:  {args.device}")
    print(f"Folds:   {folds}")
    print(f"Seed:    {seed}")
    print(f"Output:  {out_dir}\n")

    all_results = []

    for fold_idx in folds:
        fold_json = out_dir / f"fold_{fold_idx}.json"
        if fold_json.exists():
            print(f"[SKIP] fold {fold_idx} (ya existe {fold_json.name})")
            with open(fold_json) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n=== Fold {fold_idx} ===")

        # 1) Rebuild model + load checkpoint
        exp = build_experiment(
            config, fold_idx=fold_idx,
            ckpt_contrastive=config.get("CKPT_CONTRASTIVE"),
        )
        model = exp.model

        ckpt_path = ckpt_dir / f"best_finetune_fold_{fold_idx}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.to(args.device).eval()

        # 2) Loaders (mismos splits que el finetune original)
        train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
            config, batch_size=phase["BATCH_SIZE"],
            num_workers=0, fold_idx=fold_idx,
            n_folds=int(config.get("N_FOLDS", 5)),
            seed=seed,
            eval_protocol=config.get("EVAL_PROTOCOL", "kfold"),
        )

        # 3) Extract features from all splits
        feats_train = extract_features_from_loader(model, train_loader, args.device)
        feats_val   = extract_features_from_loader(model, val_loader,   args.device)
        feats_test  = extract_features_from_loader(model, test_loader,  args.device)

        # 4) Train 3 branch classifiers
        clf_ts,    val_auc_ts,    ep_ts    = train_branch_classifier(
            feats_train["h_ts_proj"], feats_train["labels"],
            feats_val["h_ts_proj"],   feats_val["labels"],
            config, seed=seed,
        )
        clf_fc,    val_auc_fc,    ep_fc    = train_branch_classifier(
            feats_train["h_fc_proj"], feats_train["labels"],
            feats_val["h_fc_proj"],   feats_val["labels"],
            config, seed=seed,
        )
        clf_fused, val_auc_fused, ep_fused = train_branch_classifier(
            feats_train["fused"],     feats_train["labels"],
            feats_val["fused"],       feats_val["labels"],
            config, seed=seed,
        )

        # 5) Evaluate on test
        y_test = feats_test["labels"]
        probs_ts    = predict_probs(clf_ts,    feats_test["h_ts_proj"])
        probs_fc    = predict_probs(clf_fc,    feats_test["h_fc_proj"])
        probs_fused = predict_probs(clf_fused, feats_test["fused"])

        if len(np.unique(y_test)) > 1:
            auc_ts    = float(roc_auc_score(y_test, probs_ts))
            auc_fc    = float(roc_auc_score(y_test, probs_fc))
            auc_fused = float(roc_auc_score(y_test, probs_fused))
        else:
            auc_ts = auc_fc = auc_fused = float("nan")

        # 6) Original model probs (joint training)
        npz_orig = np.load(ckpt_dir / f"preds_fold_{fold_idx}.npz", allow_pickle=True)
        y_orig    = npz_orig["labels"]
        probs_orig = npz_orig["probs"]
        auc_orig = float(roc_auc_score(y_orig, probs_orig)) if len(np.unique(y_orig)) > 1 else float("nan")

        # 7) Site
        if "test_subject_ids" in npz_orig.files:
            test_sub_ids = npz_orig["test_subject_ids"]
        else:
            test_sub_ids = npz_orig["subject_indices"][npz_orig["test_idx"]]
        sites = [sub_to_site.get(int(s), "UNKNOWN") for s in test_sub_ids]
        site = max(set(sites), key=sites.count) if sites else "UNKNOWN"

        result = {
            "fold": fold_idx,
            "site": site,
            "n": int(len(y_test)),
            "n_td": int((y_test == 0).sum()),
            "n_asd": int((y_test == 1).sum()),
            "val_auc_ts":    float(val_auc_ts),
            "val_auc_fc":    float(val_auc_fc),
            "val_auc_fused": float(val_auc_fused),
            "best_epoch_ts":    int(ep_ts),
            "best_epoch_fc":    int(ep_fc),
            "best_epoch_fused": int(ep_fused),
            "auc_temporal":     auc_ts,
            "auc_connectivity": auc_fc,
            "auc_fused_frozen": auc_fused,
            "auc_full_original": auc_orig,
        }
        all_results.append(result)
        with open(fold_json, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  site={site:<12s} n={result['n']:>3d}  "
              f"auc_ts={auc_ts:.3f}  auc_fc={auc_fc:.3f}  "
              f"auc_fused={auc_fused:.3f}  auc_orig={auc_orig:.3f}")

    # ── Agregar y guardar ────────────────────────────────────────────
    df = pd.DataFrame(all_results).sort_values("fold").reset_index(drop=True)
    df.to_csv(out_dir / "branch_ablation_by_site.csv", index=False)

    metric_cols = ["auc_temporal", "auc_connectivity", "auc_fused_frozen", "auc_full_original"]

    print("\n" + "="*110)
    print("RESUMEN — Branch ablation LOSO (todos los sitios)")
    print("="*110)
    header = f"{'site':<12s} {'n':>4s}  " + "  ".join(f"{c.replace('auc_',''):>16s}" for c in metric_cols)
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        line = f"{r['site']:<12s} {int(r['n']):>4d}  "
        for c in metric_cols:
            line += f"{r[c]:>16.3f}  "
        print(line)
    print("-" * len(header))
    means = df[metric_cols].mean()
    line = f"{'MEAN':<12s} {'':>4s}  "
    for c in metric_cols:
        line += f"{means[c]:>16.3f}  "
    print(line)

    focus = df[df["site"].isin(["MAX_MUN", "SBL"])].reset_index(drop=True)
    print("\n" + "="*110)
    print("FOCO: MAX_MUN y SBL")
    print("="*110)
    for _, r in focus.iterrows():
        print(f"\n{r['site']} (n={int(r['n'])}, TD={int(r['n_td'])}, ASD={int(r['n_asd'])})")
        for c in metric_cols:
            print(f"  {c.replace('auc_',''):<20s}: {r[c]:.4f}")

    # JSON
    with open(out_dir / "branch_ablation_global.json", "w") as f:
        json.dump({
            "run_dir": str(run_dir),
            "seed": seed,
            "n_folds": len(df),
            "per_site": df.to_dict(orient="records"),
            "global_mean": {c: float(means[c]) for c in metric_cols},
            "focus_sites": focus.to_dict(orient="records"),
        }, f, indent=2)

    # Markdown
    md = ["# Branch ablation — TwoTST (LOSO)", ""]
    md.append(f"**Run:** `{run_dir}`")
    md.append(f"**Seed:** {seed}")
    md.append(f"**Folds:** {len(df)}")
    md.append("")
    md.append("## Metodología")
    md.append("")
    md.append("Para cada fold LOSO N:")
    md.append("1. Reconstruir el modelo del fold N y cargar `best_finetune_fold_N.pt`. "
              "Encoders CONGELADOS (no se reentrenan).")
    md.append("2. Extraer features congeladas de train_N, val_N, test_N:")
    md.append("   - `h_ts_proj` = proj_head_1(transformer_ts(ts)) — 128d")
    md.append("   - `h_fc_proj` = proj_head_2(transformer_fc(pcc)) — 128d")
    md.append("   - `fused` = fusion(h_ts_proj, h_fc_proj) — 128d")
    md.append("3. Entrenar 3 clasificadores (misma arquitectura que `tag_classifier`, "
              "mismos hiperparámetros que `FINETUNING`):")
    md.append("   - **Temporal**: input=`h_ts_proj`")
    md.append("   - **Connectivity**: input=`h_fc_proj`")
    md.append("   - **Fused (frozen, control)**: input=`fused`")
    md.append("   Early stopping en val_N AUC.")
    md.append("4. Evaluar cada clasificador sobre test_N.")
    md.append("5. Reportar también el modelo original (joint training) desde `preds_fold_N.npz`.")
    md.append("")
    md.append("**Sin fuga:**")
    md.append("- Los encoders solo vieron `train_N` durante el finetune.")
    md.append("- Los clasificadores se entrenan solo con features de `train_N`.")
    md.append("- Early stopping en `val_N` (sitos ajenos al test_N).")
    md.append("- `test_N` nunca se usa para entrenar ni para early stopping.")
    md.append("")
    md.append("## Tabla por sitio")
    md.append("")
    md.append("| Sitio | n | Temporal | Connectivity | Fused (frozen) | Full (original) |")
    md.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(f"| {r['site']} | {int(r['n'])} | "
                  f"{r['auc_temporal']:.3f} | {r['auc_connectivity']:.3f} | "
                  f"{r['auc_fused_frozen']:.3f} | {r['auc_full_original']:.3f} |")
    md.append("")
    md.append("## Media global")
    md.append("")
    md.append("| Métrica | Media |")
    md.append("|---|---|")
    for c in metric_cols:
        md.append(f"| {c} | {means[c]:.4f} |")
    md.append("")
    md.append("## Foco: MAX_MUN y SBL")
    md.append("")
    md.append("| Sitio | Temporal | Connectivity | Fused (frozen) | Full (original) |")
    md.append("|---|---|---|---|---|")
    for _, r in focus.iterrows():
        md.append(f"| {r['site']} | "
                  f"{r['auc_temporal']:.3f} | {r['auc_connectivity']:.3f} | "
                  f"{r['auc_fused_frozen']:.3f} | {r['auc_full_original']:.3f} |")

    with open(out_dir / "branch_ablation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nGuardado en {out_dir}:")
    print(f"  branch_ablation_by_site.csv")
    print(f"  branch_ablation_global.json")
    print(f"  branch_ablation_report.md")


if __name__ == "__main__":
    main()
