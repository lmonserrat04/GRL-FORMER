"""
Verifica GRL: lambda=1 invierte, lambda=-1 no invierte (identidad).
Reutiliza checkpoints guardados (no re-entrena).
"""
# --- sys.path bootstrap (scripts movidos a subcarpetas) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# --- end bootstrap ---

import glob
from pathlib import Path

import torch
import yaml

from data.loaders.dataloader import get_finetune_loaders
from training.setup import build_experiment


# ─── Config del experimento guardado ────────────────────────────────
exp_dirs = sorted(glob.glob("experiments/*_transformer_baseline"))
if not exp_dirs:
    raise SystemExit("No hay experimentos previos")
exp_dir = Path(exp_dirs[-1])
print(f"Usando: {exp_dir}")

with open(exp_dir / "config.yaml") as f:
    config = yaml.safe_load(f)

config["EXP_DIR"] = str(exp_dir)
config["CHECKPOINTS_PATH"] = str(exp_dir / "checkpoints")
config["CKPT_TST1"] = str(exp_dir / "checkpoints" / "best_pt_ts_fold_0.pt")
config["CKPT_TST2"] = str(exp_dir / "checkpoints" / "best_pt_fc_fold_0.pt")
config["CKPT_CONTRASTIVE"] = str(exp_dir / "checkpoints" / "contrastive_global.pt")
config["EXPERIMENT_TYPE"] = "finetune"
config["DEVICE"] = "cpu"
config["N_FOLDS"] = 5


exp = build_experiment(config, fold_idx=0,
                       ckpt_contrastive=config["CKPT_CONTRASTIVE"])
model = exp.model
model.eval()


tr, va, _, _ = get_finetune_loaders(
    config, batch_size=8, num_workers=0,
    fold_idx=0, n_folds=5, seed=config["SEED"], eval_protocol="kfold",
)
batch = next(iter(tr))
ts = batch["timeseries"]
pcc = batch["pcc_vector"]
site = batch["site_id"]


# ─── Hook en 'fused' ────────────────────────────────────────────────
captured = {}

def hook(module, input, output):
    captured["fused"] = output
    output.register_hook(lambda g: captured.update({"grad": g.detach().clone()}))

handle = model.fusion.register_forward_hook(hook)


def compute_fused_grad(lambda_value):
    captured.clear()
    model.grl_lambda = lambda_value
    model.zero_grad()
    tag_logits, domain_logits = model(ts, pcc, return_domain_logits=True)
    domain_loss = torch.nn.functional.cross_entropy(domain_logits, site)
    domain_loss.backward()
    return captured["grad"].clone(), domain_loss.item()


# ─── Test 1: lambda=-1 (identidad: -(-1)·g = +g) ────────────────────
print("\n" + "="*60)
print("TEST 1: lambda=-1 (GRL actua como identidad)")
print("="*60)
grad_identity, loss_1 = compute_fused_grad(-1.0)
print(f"domain_loss:        {loss_1:.4f}")
print(f"fused grad abs sum: {grad_identity.abs().sum().item():.6e}")


# ─── Test 2: lambda=1 (invierte) ────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: lambda=1 (GRL invierte)")
print("="*60)
grad_inverted, loss_2 = compute_fused_grad(1.0)
print(f"domain_loss:        {loss_2:.4f}")
print(f"fused grad abs sum: {grad_inverted.abs().sum().item():.6e}")


# ─── Test 3: lambda=0 (anula) ───────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: lambda=0 (GRL anula el gradiente)")
print("="*60)
grad_zero, loss_3 = compute_fused_grad(0.0)
print(f"fused grad abs sum: {grad_zero.abs().sum().item():.6e}")


# ─── Verificaciones ─────────────────────────────────────────────────
print("\n" + "="*60)
print("VERIFICACION")
print("="*60)

# 1. GRL invierte: grad(λ=1) == -grad(λ=-1)
assert torch.allclose(grad_inverted, -grad_identity, rtol=1e-4, atol=1e-8), \
    f"FALLO: grad(λ=1) != -grad(λ=-1)"
print("[OK] grad(λ=1) = -grad(λ=-1)  -> GRL invierte el gradiente")

# 2. GRL anula con λ=0
assert grad_zero.abs().sum().item() < 1e-10, "FALLO: λ=0 no anula"
print("[OK] grad(λ=0) = 0            -> GRL anula con lambda 0")

# 3. Forward no cambia con λ
assert abs(loss_1 - loss_2) < 1e-5, "FALLO: forward cambia con λ"
print("[OK] forward identico con cualquier lambda")

# 4. domain_classifier recibe gradiente normal
print("\nVerificando domain_classifier:")
model.grl_lambda = 1.0
model.zero_grad()
_, domain_logits = model(ts, pcc, return_domain_logits=True)
torch.nn.functional.cross_entropy(domain_logits, site).backward()

dc_grad = model.domain_classifier.mlp_head[-1].weight.grad
assert dc_grad is not None and dc_grad.abs().sum() > 0, \
    "domain_classifier no recibio gradiente"
print(f"[OK] domain_classifier grad abs sum: {dc_grad.abs().sum().item():.6e}")

# 5. tag_classifier limpio
print("\nVerificando tag_classifier:")
tc_weight = model.tag_classifier.mlp_head[-1].weight
if tc_weight.grad is None or tc_weight.grad.abs().sum().item() < 1e-10:
    print("[OK] tag_classifier sin gradiente (correcto)")
else:
    raise AssertionError(f"tag_classifier recibio gradiente: {tc_weight.grad.abs().sum().item()}")

handle.remove()
print("\n[OK] GRL verificada correctamente")
