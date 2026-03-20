import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

# ==========================================
# 1. THE BULLETPROOF PATH HACK
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
from src.zo_estimators import estimate_gradient_cge
from src.utils import AverageMeter, accuracy, save_checkpoint

# Importing the CIFAR-specific architecture
try:
    from archs.cifar_resnet import resnet # type: ignore
except Exception as e:
    print(f"ACTUAL PYTHON ERROR: {e}")
    sys.exit(1)

# ==========================================
# 3. TRAINING FUNCTIONS
# ==========================================
def train_stage_one(model, train_loader, start_epoch, total_epochs, device):
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
                print(f"Stage 1 | Epoch [{epoch+1}/{total_epochs}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.6f}", flush=True)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stage': 1
        }, filename="outputs/checkpoint_stage1.pth.tar")

def train_stage_two(model, train_loader, classifier, start_epoch, total_epochs, q, device):
    print(f"\n--- Stage 2: ZO Training (q={q}, Target: {total_epochs} Epochs) ---", flush=True)
    
    # Keeping LR at 1e-4 as discussed for stability
    optimizer = Adam(list(model.denoiser.parameters()) + list(model.encoder.parameters()), lr=1e-4)
    
    # FIX 1: Prevent "Hive Mind" batch averaging
    criterion = nn.CrossEntropyLoss(reduction='none') 
    
    model.denoiser.train()
    model.encoder.train()
    model.decoder.eval()
    classifier.eval()

    # FIX 4: Explicitly freeze decoder parameters to prevent memory leaks in the ZO loop
    for param in model.decoder.parameters():
        param.requires_grad = False

    for epoch in range(start_epoch, total_epochs):
        losses_total = AverageMeter()
        top1 = AverageMeter()
        
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            with torch.no_grad():
                original_pre = classifier(images).argmax(1).detach().clone()
            
            noisy_inputs = images + torch.randn_like(images).to(device) * 0.5 
            
            denoised = model.denoiser(noisy_inputs)
            z = model.encoder(denoised)
            z.requires_grad_(True)
            
            with torch.no_grad():
                output_0 = classifier(model.decoder(z))
                loss_0 = criterion(output_0, original_pre)
            
            # ZO Stability Loss
            grad_est = estimate_gradient_cge(z, classifier, model.decoder, original_pre, criterion, mu=0.005)
            
            # FIX 2 & 3: Flatten Z to prevent broadcast explosion, and average the batch
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

        # Save checkpoint... (keep your existing save logic here)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stage': 2
        }, filename="outputs/checkpoint_stage2.pth.tar")

        
# ==========================================
# 4. MAIN EXECUTION (CLOUD SETTINGS)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs("outputs", exist_ok=True)
    
    # Cranked up batch size and workers for the L4 GPU
    train_loader, _ = get_loaders(batch_size=128, workers=8) 
    
    # --- LOAD THE RESNET-110 ---
    print("=> Loading pre-trained ResNet-110 for CIFAR-10...", flush=True)
    victim = resnet(depth=110, num_classes=10).to(device)
    resnet_path = os.path.join(EXTERNAL_DIR, 'trained_models', 'CIFAR-10', 'Classifiers', 'resnet110.pth.tar')
    
    checkpoint = torch.load(resnet_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[2:] if k.startswith('1.') else k
        new_state_dict[name] = v
        
    victim.load_state_dict(new_state_dict)
    victim.eval()
    for param in victim.parameters():
        param.requires_grad = False
    print("=> Victim model loaded and frozen.", flush=True)
    
    defense = ZO_AE_DS_Defense(victim_model=victim).to(device)
    start_epoch, current_stage = 0, 1

    if args.resume:
        for stage_file in ["outputs/checkpoint_stage2.pth.tar", "outputs/checkpoint_stage1.pth.tar"]:
            if os.path.exists(stage_file):
                ckpt = torch.load(stage_file, map_location=device)
                defense.load_state_dict(ckpt['state_dict'])
                start_epoch, current_stage = ckpt['epoch'], ckpt['stage']
                print(f"=> Resumed Stage {current_stage} from Epoch {start_epoch}", flush=True)
                break

    # Stage 1: Extended to 100 Epochs for proper AE convergence
    if current_stage == 1:
        train_stage_one(defense, train_loader, start_epoch, total_epochs=100, device=device)
        start_epoch, current_stage = 0, 2
    
    # Stage 2: Extended to 50 Epochs to match paper's ZO fine-tuning
    if current_stage == 2:
        train_stage_two(defense, train_loader, victim, start_epoch, total_epochs=50, q=192, device=device)
    
    print("--- ALL TRAINING COMPLETE ---", flush=True)