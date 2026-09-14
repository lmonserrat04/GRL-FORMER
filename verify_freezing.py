"""Verifica qué se congela en cada fase."""
import yaml, torch
from pathlib import Path

config = yaml.safe_load(open("config/config.yaml"))
device = torch.device("cpu")

# ============================================================
# FASE CONTRASTIVE
# ============================================================
print("="*60)
print("FASE CONTRASTIVE (paper Sec. 3.3 + README repo)")
print("Esperado: freeze TST1, unfreeze TST2, projections entrenables")
print("="*60)

from models.transformer_ts import create_transformer_ts
from models.transformer_fc import create_transformer_fc
from training.tasks.contrastive import ContrastiveWrapper

tst1 = create_transformer_ts({
    "n_rois": config["N_ROIS"], "emb_dim": config["TST1"]["D_MODEL"],
    "n_layers": config["TST1"]["NUM_ENCODER_LAYERS"], "n_heads": config["TST1"]["N_HEADS"],
    "dim_feedforward": config["TST1"]["DIM_FEEDFORWARD"], "dropout": config["TST1"]["ENC_DROP"],
    "max_seq_len": config["MAX_SEQ_LEN"], "use_cls_token": config["TST1"]["USE_CLS_TOKEN"],
})
tst2 = create_transformer_fc({
    "pcc_dim": config["TST2"]["PCC_DIM"], "d_model": config["TST2"]["D_MODEL"],
    "n_layers": config["TST2"]["NUM_ENCODER_LAYERS"], "n_heads": config["TST2"]["N_HEADS"],
    "dim_feedforward": config["TST2"]["DIM_FEEDFORWARD"], "dropout": config["TST2"]["ENC_DROP"],
})
cm = ContrastiveWrapper(
    dim_ts=tst1.emb_dim, dim_fc=tst2.d_model,
    hidden_dim=config["T_CONTRASTIVE"]["PROJ_HIDDEN_DIM"],
    output_dim=config["T_CONTRASTIVE"]["PROJ_OUTPUT_DIM"],
    temperature=config["T_CONTRASTIVE"]["TEMPERATURE"],
)

# Aplicar la lógica del run_contrastive_global
for p in tst1.parameters(): p.requires_grad = False
for p in tst2.parameters(): p.requires_grad = True

def count(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name:15s}: {trainable:>10,} / {total:>10,} entrenables")
    return trainable, total

print("\nEstado tras aplicar freezing:")
count(tst1, "TST1")
count(tst2, "TST2")
count(cm,   "ContrastiveWrapper")

# Veredicto
tst1_frozen = all(not p.requires_grad for p in tst1.parameters())
tst2_frozen = all(not p.requires_grad for p in tst2.parameters())
cm_trainable = all(p.requires_grad for p in cm.parameters())
print(f"\n  ¿TST1 congelado?           {'✓' if tst1_frozen else '❌'}")
print(f"  ¿TST2 descongelado?        {'✓' if not tst2_frozen else '❌'}")
print(f"  ¿Proj heads entrenables?   {'✓' if cm_trainable else '❌'}")

# ============================================================
# FASE FINETUNE
# ============================================================
print("\n" + "="*60)
print("FASE FINETUNE (paper Table 2 + OPTIMAL_CONFIGURATION)")
print("Esperado: TST1+TST2 unfrozen, projections frozen")
print("="*60)

from models.dual_stream import create_dual_stream_model
from training.tasks.contrastive import ProjectionHead
from data.loaders.dataloader import load_raw_data

# Verificar que existe el checkpoint contrastive
ckpt_path = None
for candidate in sorted(Path("experiments").glob("*/checkpoints/contrastive_global.pt")):
    ckpt_path = candidate
if ckpt_path is None:
    print("⚠ No hay checkpoint contrastive, uso proj heads random para test")
    p1 = ProjectionHead(config["TST1"]["D_MODEL"],
                        config["T_CONTRASTIVE"]["PROJ_HIDDEN_DIM"],
                        config["T_CONTRASTIVE"]["PROJ_OUTPUT_DIM"])
    p2 = ProjectionHead(config["TST2"]["D_MODEL"],
                        config["T_CONTRASTIVE"]["PROJ_HIDDEN_DIM"],
                        config["T_CONTRASTIVE"]["PROJ_OUTPUT_DIM"])
else:
    print(f"  Cargando proj heads de: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p1 = ProjectionHead(config["TST1"]["D_MODEL"],
                        config["T_CONTRASTIVE"]["PROJ_HIDDEN_DIM"],
                        config["T_CONTRASTIVE"]["PROJ_OUTPUT_DIM"])
    p2 = ProjectionHead(config["TST2"]["D_MODEL"],
                        config["T_CONTRASTIVE"]["PROJ_HIDDEN_DIM"],
                        config["T_CONTRASTIVE"]["PROJ_OUTPUT_DIM"])
    p1.load_state_dict(ckpt["proj_head_1_state_dict"])
    p2.load_state_dict(ckpt["proj_head_2_state_dict"])

for p in p1.parameters(): p.requires_grad = False
for p in p2.parameters(): p.requires_grad = False

ds = config["DUAL_STREAM"]
model = create_dual_stream_model(
    n_rois=config["N_ROIS"], time_points=config["MAX_SEQ_LEN"],
    pcc_dim=config["TST2"]["PCC_DIM"],
    tst1_emb_dim=config["TST1"]["D_MODEL"], tst2_d_model=config["TST2"]["D_MODEL"],
    fusion_type=ds["FUSION_TYPE"],
    fusion_hidden_dim=config["FUSION"]["ATTENTION_POOLING"]["HIDDEN_DIM"],
    num_classes=ds["NUM_CLASSES"], dropout=ds["CLASSIFIER_DROPOUT"],
    mlp_dims=ds.get("MLP_DIMS"), proj_head_1=p1, proj_head_2=p2,
)
model.unfreeze_encoders()
for p in model.proj_head_1.parameters(): p.requires_grad = False
for p in model.proj_head_2.parameters(): p.requires_grad = False

print("\nEstado tras aplicar freezing:")
def count_dual(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name:20s}: {trainable:>10,} / {total:>10,} entrenables")

count_dual(model, "DualStreamModel TOTAL")
print()
count_dual(model.transformer_ts, "  TST1")
count_dual(model.transformer_fc, "  TST2")
count_dual(model.fusion, "  Fusion")
count_dual(model.classifier, "  Classifier MLP")
count_dual(model.proj_head_1, "  proj_head_1")
count_dual(model.proj_head_2, "  proj_head_2")

# Veredicto
tst1_ok = all(p.requires_grad for p in model.transformer_ts.parameters())
tst2_ok = all(p.requires_grad for p in model.transformer_fc.parameters())
p1_frozen = all(not p.requires_grad for p in model.proj_head_1.parameters())
p2_frozen = all(not p.requires_grad for p in model.proj_head_2.parameters())
clf_ok = all(p.requires_grad for p in model.classifier.parameters())
fuse_ok = all(p.requires_grad for p in model.fusion.parameters())

print(f"\n  ¿TST1 entrenable?          {'✓' if tst1_ok else '❌'}")
print(f"  ¿TST2 entrenable?          {'✓' if tst2_ok else '❌'}")
print(f"  ¿Fusión entrenable?        {'✓' if fuse_ok else '❌'}")
print(f"  ¿Classifier entrenable?    {'✓' if clf_ok else '❌'}")
print(f"  ¿proj_head_1 congelada?    {'✓' if p1_frozen else '❌'}")
print(f"  ¿proj_head_2 congelada?    {'✓' if p2_frozen else '❌'}")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"  Contrastive (Fase 3): TST1 frozen, TST2 unfrozen  {'✓' if tst1_frozen and not tst2_frozen else '❌'}")
print(f"  Finetune    (Fase 4): TST1+TST2 unfrozen, proj frozen  {'✓' if tst1_ok and tst2_ok and p1_frozen and p2_frozen else '❌'}")
