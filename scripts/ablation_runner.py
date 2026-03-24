"""
Ablation Runner
===============
Runs training ablations with composable configuration flags.
Each flag is an independent axis — combine freely.

Examples:
    # Default baseline (ae_size=192, encoder trainable)
    python scripts/ablation_runner.py

    # Smaller autoencoder
    python scripts/ablation_runner.py --ae_size 96

    # Freeze encoder in Stage 2
    python scripts/ablation_runner.py --freeze_encoder

    # Combine both
    python scripts/ablation_runner.py --ae_size 96 --freeze_encoder

    # Resume from checkpoint
    python scripts/ablation_runner.py --ae_size 96 --freeze_encoder --resume
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import OrderedDict

# ==========================================
# 1. PATH
# ==========================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXTERNAL_DIR = os.path.join(BASE_PATH, 'external', 'Black-Box-Defense')

if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if EXTERNAL_DIR not in sys.path:
    sys.path.append(EXTERNAL_DIR)

# ==========================================
# 2. IMPORTS
# ==========================================
from src.models import ZO_AE_DS_Defense
from src.data import get_loaders
from src.zo_estimators import estimate_gradient_cge, estimate_gradient_rge
from src.utils import AverageMeter, accuracy, save_checkpoint

try:
    from archs.cifar_resnet import resnet  # type: ignore
except Exception as e:
    print(f"ACTUAL PYTHON ERROR: {e}")
    sys.exit(1)

# ==========================================
# 3. HELPERS
# ==========================================
def build_run_name(args):
    """
    Auto-generate an output directory name from the flags used.
    Only includes flags that differ from the baseline defaults.
    """
    tags = []

    # AE size (default is 192)
    tags.append(f"ae{args.ae_size}")

    # Freeze encoder (default is False / trainable)
    if args.freeze_encoder:
        tags.append("frozenenc")

    # Noise sigma (default is 0.5)
    if args.noise_sigma != 0.5:
        tags.append(f"sigma{args.noise_sigma}")

    # ZO method (default is cge)
    if args.zo_method == 'rge':
        tags.append(f"rge_q{args.rge_q}")

    # --- ADD NEW TAGS HERE as you add more flags ---

    return "_".join(tags)


def load_victim(device):
    """Load and freeze the pre-trained ResNet-110."""
    print("=> Loading pre-trained ResNet-110 for CIFAR-10...", flush=True)
    victim = resnet(depth=110, num_classes=10).to(device)
    resnet_path = os.path.join(EXTERNAL_DIR, 'trained_models', 'CIFAR-10', 'Classifiers', 'resnet110.pth.tar')

    checkpoint = torch.load(resnet_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[2:] if k.startswith('1.') else k
        new_state_dict[name] = v

    victim.load_state_dict(new_state_dict)
    victim.eval()
    for param in victim.parameters():
        param.requires_grad = False
    print("=> Victim model loaded and frozen.", flush=True)
    return victim


# ==========================================
# 4. TRAINING FUNCTIONS
# ==========================================
def train_stage_one(model, train_loader, start_epoch, total_epochs, device, output_dir):
    print(f"\n--- Stage 1: AE Pre-training (Target: {total_epochs} Epochs) ---", flush=True)
    optimizer = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(start_epoch, total_epochs):
        losses = AverageMeter()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            reconstructed = model.forward_ae(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            if i % 50 == 0:
                print(f"Stage 1 | Epoch [{epoch+1}/{total_epochs}] Batch [{i}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}", flush=True)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stage': 1
        }, filename=os.path.join(output_dir, "checkpoint_stage1.pth.tar"))


def train_stage_two(model, train_loader, classifier, start_epoch, total_epochs, q, device, output_dir,
                    freeze_encoder=False, noise_sigma=0.5, zo_method='cge'):
    print(f"\n--- Stage 2: ZO Training (method={zo_method}, q={q}, Target: {total_epochs} Epochs) ---", flush=True)

    # Build the trainable parameter list based on flags
    trainable_params = list(model.denoiser.parameters())
    if not freeze_encoder:
        trainable_params += list(model.encoder.parameters())
        print("   Encoder: TRAINABLE", flush=True)
    else:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("   Encoder: FROZEN", flush=True)

    optimizer = Adam(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.denoiser.train()
    model.encoder.eval() if freeze_encoder else model.encoder.train()
    model.decoder.eval()
    classifier.eval()

    for param in model.decoder.parameters():
        param.requires_grad = False

    for epoch in range(start_epoch, total_epochs):
        losses_total = AverageMeter()
        top1 = AverageMeter()

        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                original_pre = classifier(images).argmax(1).detach().clone()

            noisy_inputs = images + torch.randn_like(images).to(device) * noise_sigma

            denoised = model.denoiser(noisy_inputs)
            z = model.encoder(denoised)
            z.requires_grad_(True)

            with torch.no_grad():
                output_0 = classifier(model.decoder(z))
                loss_0 = criterion(output_0, original_pre)

            # ZO Stability Loss
            if zo_method == 'cge':
                grad_est = estimate_gradient_cge(z, classifier, model.decoder, original_pre, criterion, mu=0.005)
            else:
                grad_est = estimate_gradient_rge(z, classifier, model.decoder, original_pre, criterion, loss_0, mu=0.005, q=q)

            z_flat = torch.flatten(z, start_dim=1)
            surrogate_loss = torch.sum(z_flat * grad_est, dim=-1).mean()

            total_loss = surrogate_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc1 = accuracy(output_0, targets, topk=(1,))[0]
            losses_total.update(total_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            if i % 10 == 0:
                print(f"Stage 2 | Epoch [{epoch+1}/{total_epochs}] Batch [{i}/{len(train_loader)}] "
                      f"ZO Stab Loss: {losses_total.avg:.4f} "
                      f"Acc: {top1.avg:.2f}%", flush=True)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stage': 2
        }, filename=os.path.join(output_dir, "checkpoint_stage2.pth.tar"))


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run composable training ablations')

    # Ablation flags — each is an independent axis
    parser.add_argument('--ae_size', type=int, default=192, choices=[192, 96],
                        help='Autoencoder latent dimension (default: 192)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder during Stage 2 (only denoiser trains)')
    parser.add_argument('--noise_sigma', type=float, default=0.5,
                        help='Gaussian noise std dev during Stage 2 training (default: 0.5)')
    parser.add_argument('--zo_method', type=str, default='cge', choices=['cge', 'rge'],
                        help='ZO gradient estimation method (default: cge)')
    parser.add_argument('--rge_q', type=int, default=None,
                        help='Number of random directions for RGE (default: latent dim). Ignored for CGE.')
    # --- ADD NEW FLAGS HERE ---

    # Training control
    parser.add_argument('--stage1_epochs', type=int, default=100, help='Stage 1 epochs (default: 100)')
    parser.add_argument('--stage2_epochs', type=int, default=50, help='Stage 2 epochs (default: 50)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()

    # Auto-set q: for CGE it's always latent dim, for RGE use --rge_q or default to latent dim
    if args.zo_method == 'cge':
        q = args.ae_size
    else:
        q = args.rge_q if args.rge_q is not None else args.ae_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = build_run_name(args)
    output_dir = os.path.join("outputs", f"ablation_{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ABLATION RUN: {run_name}")
    print(f"  Config: ae_size={args.ae_size}, freeze_encoder={args.freeze_encoder}, noise_sigma={args.noise_sigma}, zo_method={args.zo_method}, q={q}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")

    train_loader, _ = get_loaders(batch_size=128, workers=8)
    victim = load_victim(device)

    defense = ZO_AE_DS_Defense(victim_model=victim, ae_size=args.ae_size).to(device)
    start_epoch, current_stage = 0, 1

    if args.resume:
        for stage_file in [
            os.path.join(output_dir, "checkpoint_stage2.pth.tar"),
            os.path.join(output_dir, "checkpoint_stage1.pth.tar"),
        ]:
            if os.path.exists(stage_file):
                ckpt = torch.load(stage_file, map_location=device)
                defense.load_state_dict(ckpt['state_dict'])
                start_epoch, current_stage = ckpt['epoch'], ckpt['stage']
                print(f"=> Resumed Stage {current_stage} from Epoch {start_epoch}", flush=True)
                break

    # Stage 1
    if current_stage == 1:
        train_stage_one(defense, train_loader, start_epoch, args.stage1_epochs, device, output_dir)
        start_epoch, current_stage = 0, 2

    # Stage 2
    if current_stage == 2:
        train_stage_two(defense, train_loader, victim, start_epoch, args.stage2_epochs, q, device, output_dir,
                        freeze_encoder=args.freeze_encoder, noise_sigma=args.noise_sigma, zo_method=args.zo_method)

    print(f"\n--- ABLATION '{run_name}' COMPLETE ---")
    print(f"    Checkpoints saved to: {output_dir}/")
