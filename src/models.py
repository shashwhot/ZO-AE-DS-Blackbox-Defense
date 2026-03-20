import sys
import os
import torch
import torch.nn as nn

# 1. Point Python to external folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../external/Black-Box-Defense')))

# 2. Import the architectures
from archs.dncnn import DnCNN
from archs.cae import Cifar_Encoder_192, Cifar_Decoder_192, Cifar_Encoder_96, Cifar_Decoder_96

class ZO_AE_DS_Defense(nn.Module):
    """
    The unified defensive architecture: Denoiser -> Encoder -> Decoder -> Victim Model
    """
    def __init__(self, victim_model, ae_size=192):
        super(ZO_AE_DS_Defense, self).__init__()
        
        # The Denoiser (DnCNN)
        self.denoiser = DnCNN(image_channels=3, depth=17, n_channels=64)
        
        # The Autoencoder (AE) Ablation Setup
        if ae_size == 192:
            self.encoder = Cifar_Encoder_192()
            self.decoder = Cifar_Decoder_192()
        elif ae_size == 96:
            self.encoder = Cifar_Encoder_96()
            self.decoder = Cifar_Decoder_96()
        else:
            raise ValueError("Unsupported AE size. Choose 192 or 96.")
            
        # The Black-Box Victim Model
        self.victim_model = victim_model
        
        # Freeze the victim model
        for param in self.victim_model.parameters():
            param.requires_grad = False

    def forward_ae(self, x):
        """Used for Stage 1: Pre-training the Autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def forward(self, x):
        """Used for Stage 2: End-to-End Defense inference"""
        denoised = self.denoiser(x)
        encoded = self.encoder(denoised)
        decoded = self.decoder(encoded)
        prediction = self.victim_model(decoded)
        return prediction