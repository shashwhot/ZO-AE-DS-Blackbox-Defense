# Robust Black-Box Model Defense 

This repository contains my custom training pipeline and modular implementation for training a robust defense against black-box attacks, optimized for **NVIDIA L4 GPUs** on Google Cloud.

## Project Overview
The goal of this project is to implement the "Zeroth-Order (ZO) Autoencoder-based Denoiser" defense. I took the core logic from the original research and built a modular pipeline to handle:
* **Two-Stage Training:** Stage 1 (Autoencoder Pre-training) and Stage 2 (ZO Fine-tuning).
* **Cloud Optimization:** Scaled batch sizes (128/256) and optimized data loading for high-performance VMs.
* **Checkpointing:** Full resume functionality for long-running cloud marathons.

##  Credits & Acknowledgments
This work is built upon the research and codebase provided by the authors of:
**"How to Robustify Black-Box Machine Learning Models"**

I have cloned their official repository as the backbone for the models and zeroth-order estimators. 
* **Original Repo:** https://github.com/damon-demon/Black-Box-Defense
* **Paper:** https://arxiv.org/abs/2203.14195

##  How I Ran This
1. **Environment:** Google Cloud Platform (GCP) with an NVIDIA L4 GPU.
2. **Setup:** I cloned the official repository into an `external/` folder.
3. **Execution:** I ran my custom `scripts/train_pipeline.py` which interfaces with the external models to perform the 150-epoch training protocol.