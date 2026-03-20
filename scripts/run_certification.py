import sys
import os
import argparse
import torch
import numpy as np

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
from src.certify import Smooth

try:
    from archs.cifar_resnet import resnet # type: ignore
except Exception as e:
    print(f"ACTUAL PYTHON ERROR: {e}")
    sys.exit(1)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Certify the ZO-AE-DS Defense')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoint_stage2.pth.tar', help='Path to the trained Stage 2 model')
    parser.add_argument('--sigma', type=float, default=0.5, help='Noise standard deviation (0.5 matches variance 0.25 from the paper)')
    parser.add_argument('--N0', type=int, default=100, help='Number of samples to guess the top class')
    parser.add_argument('--N', type=int, default=10000, help='Number of samples to certify the radius (Paper uses 100,000, 10k is faster for testing)')
    parser.add_argument('--alpha', type=float, default=0.001, help='Failure probability tolerance')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size for the noise sampling')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of test images to evaluate (Full set is 10000)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Certification on {device.upper()} ---")

    # 1. Load the Test Data
    _, test_loader = get_loaders(batch_size=1) # Batch size 1 because we bombard 1 image with N noise samples

    # 2. Load the Victim Model
    print("=> Loading frozen ResNet-110...")
    victim = resnet(depth=110, num_classes=10).to(device)
    resnet_path = os.path.join(EXTERNAL_DIR, 'trained_models', 'CIFAR-10', 'Classifiers', 'resnet110.pth.tar')
    checkpoint = torch.load(resnet_path, map_location=device)
    
    # Handle module prefix if it exists in the checkpoint
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[2:] if k.startswith('1.') else k
        new_state_dict[name] = v
    victim.load_state_dict(new_state_dict)
    victim.eval()

    # 3. Load the Trained Defense Pipeline
    print(f"=> Loading trained defense from {args.checkpoint}...")
    defense = ZO_AE_DS_Defense(victim_model=victim).to(device)
    
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        defense.load_state_dict(ckpt['state_dict'])
        print(f"   Successfully loaded Stage {ckpt.get('stage', 'Unknown')} checkpoint from Epoch {ckpt.get('epoch', 'Unknown')}.")
    else:
        print(f"   [!] WARNING: Could not find checkpoint at {args.checkpoint}. Evaluating untrained model!")

    defense.eval()

    # 4. Initialize the Randomized Smoothing Wrapper
    smoothed_classifier = Smooth(base_classifier=defense, num_classes=10, sigma=args.sigma)

    # 5. Certification Loop
    print(f"\n=> Certifying {args.num_examples} images with N={args.N} and sigma={args.sigma}...")
    
    # Store results: (correctness boolean, radius)
    results = []

    for i, (image, target) in enumerate(test_loader):
        if i >= args.num_examples:
            break
            
        image, target = image.to(device), target.item()
        
        # Run the Cohen certification algorithm
        prediction, radius = smoothed_classifier.certify(
            x=image, 
            n0=args.N0, 
            n=args.N, 
            alpha=args.alpha, 
            batch_size=args.batch_size
        )
        
        correct = (prediction == target)
        results.append((correct, radius))
        
        # Print progress every 10 images
        if (i + 1) % 10 == 0:
            print(f"   Evaluated [{i+1}/{args.num_examples}] images...")

    # 6. Calculate Certified Accuracy at Specific Radii
    print("\n--- Final Certification Results ---")
    radii_to_check = [0.00, 0.25, 0.50, 0.75] # These are the columns from Table 2 in the paper
    total_evaluated = len(results)

    for r in radii_to_check:
        # A model is "Certified Correct" if it predicted the right class AND its safe radius is >= r
        certified_correct_count = sum(1 for correct, radius in results if correct and radius >= r)
        certified_acc = (certified_correct_count / total_evaluated) * 100.0
        
        label = "Standard Accuracy (SA)" if r == 0.0 else f"Certified Accuracy (CA) at r={r}"
        print(f"{label}: {certified_acc:.2f}%")
        
    print("\nDone! To match the paper's exact numbers, run with --num_examples 10000 and --N 100000 on a heavy compute node.")