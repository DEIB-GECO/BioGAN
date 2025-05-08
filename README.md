# BioGAN: Graph-Informed Generative Modeling for Synthetic Transcriptomics Data

BioGAN is a novel generative framework that integrates **Graph Neural Networks (GNNs)** into a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** to generate biologically realistic synthetic transcriptomics data. The model leverages known gene-gene relationships to guide the generation process, aiming to improve the **biological plausibility**, **fairness**, and **privacy** of synthetic omics data. 

This project accompanies the manuscript:

> *"BioGAN: Enhancing Transcriptomic Data Generation with Biological Knowledges" — Francesca Pia Panaccione et al., 2025 (..)*

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
src/
├── BIOGAN_H2GCN/          # Experimental variant of BioGAN using H2GCN architecture
├── BioGAN_GCN/            # Core BioGAN model implementation based on GCN
├── wpgan/                 # Implementation of Wasserstein GAN with Gradient Penalty
├── metrics/               # Evaluation metrics for synthetic data quality
├── utils/                 # Utility functions (e.g., graph processing, logging)
├── data_loader.py         # Scripts for preprocessing transcriptomic data
├── losses.py              # Custom loss functions (Wasserstein, gradient penalty, etc.)
├── train_model.py         # Main training script for the BioGAN model
