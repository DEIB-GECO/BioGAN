# BioGAN: Graph-Informed Generative Modeling for Synthetic Transcriptomics Data

BioGAN is a novel generative framework that integrates **Graph Neural Networks (GNNs)** into a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** to generate biologically realistic synthetic transcriptomics data. The model leverages known gene-gene relationships to guide the generation process, aiming to improve the **biological plausibility**, **fairness**, and **privacy** of synthetic omics data. 

This project accompanies the manuscript:

> *"Graph-Informed Generative Modeling for Transcriptomic Data Synthesis" — Francesca Pia Panaccione et al., 2025 (preprint/manuscript link to be added)*

---

## 🧬 Motivation

Modern AI tools in genomics are limited by data scarcity, heterogeneity, acquisition costs, and privacy regulations (e.g., **GDPR**, **HIPAA**). BioGAN addresses these issues by:

- Synthesizing realistic transcriptomic profiles conditioned on biological structure.
- Incorporating prior biological knowledge through GNNs to improve fidelity.
- Providing a versatile architecture adaptable to various omics settings.

---

##  Key Features

- **Graph-aware generator**: Integrates GNN layers to model regulatory interactions.
- **High-dimensional support**: Designed for transcriptomic data with tens of thousands of features.
- **Robust validation**: Multi-faceted evaluation pipeline including:
  - Classification accuracy on downstream tasks
  - Distributional similarity metrics (e.g., Wasserstein distance)
  - Feature-level biological consistency checks

---

## 📁 Repository Structure

```bash
BioGAN/
├── src/                    # Source code for models, training, utils
│   ├── models/             # WGAN-GP and GNN modules
│   ├── data/               # Dataloaders and preprocessing scripts
│   ├── train.py            # Main training pipeline
│   └── evaluate.py         # Evaluation scripts and metrics
├── configs/                # YAML files for experimental configuration
├── notebooks/              # Jupyter notebooks for exploration and plots
├── results/                # Output figures, logs, and checkpoints
├── requirements.txt        # Python dependencies
└── README.md               # This file
