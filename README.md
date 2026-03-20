# Black-Box Model Defense 

This repository contains a modular training and evaluation pipeline for implementing a robust defense against adversarial attacks in black-box models. The project focuses on a Zeroth-Order (ZO) Autoencoder-based Denoiser (ZO-AE-DS) designed to purify adversarial inputs before they reach a target classifier.

## Project Overview
The implementation follows a two-stage training strategy that operates without requiring access to the internal gradients of the target model.

* **Stage 1 (Autoencoder Pre-training):** Trains a denoiser-encoder-decoder chain using reconstruction loss to establish baseline purification capabilities.
* **Stage 2 (ZO Fine-tuning):** Employs Zeroth-Order optimization to adapt the denoiser to a specific black-box victim model, maximizing classification accuracy under adversarial pressure.

## Script Functionality

### scripts/train_pipeline.py
* **Automated Transitions:** Orchestrates the transition from Stage 1 (100 epochs) to Stage 2 (50 epochs).
* **State Management:** Implements checkpointing and resume logic to preserve model weights, optimizer states, and epoch counts.
* **Data Integration:** Configures data loaders for CIFAR-10/MNIST and interfaces with the pre-trained black-box victim models.

### scripts/run_certification.py
* **Robustness Evaluation:** Assesses the purified model's performance using a randomized smoothing wrapper.
* **Accuracy Certification:** Calculates Standard Accuracy (SA) and Certified Accuracy (CA) at specific radii (0.00, 0.25, 0.50, and 0.75) to quantify defense stability.

### src/zo_estimators.py
* **Gradient Approximation:** Has the math for both Random Gradient Estimation (RGE) and Coordinate-wise Gradient Estimation (CGE).
* **Central Difference Calculation:** Utilizes finite difference probing (perturbing the latent space by $\pm \mu$) to estimate the loss surface without backpropagation.

### src/models.py
* **Defensive Architecture:** Defines the `ZO_AE_DS_Defense` class, integrating a DnCNN denoiser with a latent-space Autoencoder.
* **Integrated Inference:** Provides a unified forward pass that purifies input images through the denoiser chain before classification by the frozen victim model.

## Credits and Acknowledgments
This work is built upon the research and codebase provided by the authors of:
**"How to Robustify Black-Box Machine Learning Models"**

* **Original Repository:** [https://github.com/damon-demon/Black-Box-Defense](https://github.com/damon-demon/Black-Box-Defense)
* **Research Paper:** [https://arxiv.org/abs/2203.14195](https://arxiv.org/abs/2203.14195)

## How I Ran This
1. **Environment:** Google Cloud Platform (GCP) utilizing an NVIDIA L4 GPU.
2. **Setup:** The original repository was integrated as an external dependency to provide backbone architectures and pre-trained classifiers.
3. **Execution:** The pipeline is executed via `python scripts/train_pipeline.py`.
