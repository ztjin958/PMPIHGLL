# PMPIHGLL: A deep learning model for predicting metabolite-protein interaction

PMPIHGLL is a comprehensive toolkit and dataset collection for studying protein-metabolite interactions using hypergraph-based deep learning. It is designed for researchers in bioinformatics, computational biology, and drug discovery.

## Model Architecture Overview
### The proposed framework consists of three main stages: (A) Raw Feature Extraction, (B) Feature Improvement, and (C) Prediction.
![Model Architecture](https://github.com/ztjin958/PMPIHGLL/blob/main/Figure%201_01.png)
#### (A) Raw Feature Extraction
In the initial stage, raw biological data is transformed into high-dimensional feature representations:
- Metabolites: SMILES strings are processed using ChemGPT to generate raw metabolite feature matrix (FM).
- Proteins: Amino acid sequences are encoded using ProtT5 to generate raw protein feature matrix (FP).
#### (B) Feature Improvement
This stage enhances the raw features via HyperConv, channel-wise attention mechanism and 1D-convolutional neural network:
- Graph-based Knowledge Integration:
	* Chemical-chemical interaction and protein-protein interaction information is retrieved from from STITCH and STRING, respectively. K-Nearest Neighbors (KNN) algorithm is applied to construct hypergraphs (HGm and HGp) using different K values (Km1,Km2 for HGm and (Kp1,Kp2 for HGp).
- Hypergraph Convolution & Contrastive Learning:
	* Dual HyperConv extracts complex high-level features of metabolites and proteins. Contrastive learning is further employed to enhance the quality of high-level features.
- Channel-wise Attention Mechanism:
	* Features from different HyperConv (FM~, FP~) undergo row average pooling followed by a fully connected neural network for obtaining attention weights.
	* The features are refined by the attention weights and convolution operator (FM¨ and FP¨).
	* 1-D Convolutional Neural Networks (1-D CNN) are used to fuse two metabolite and protein features (FM^ and FP^).
#### (C) Prediction
The final stage performs the interaction inference:
- Feature Fusion: The raw features (FM, FP) and the improved features (FM^,FP^) are integrated to form the final comprehensive representations.
- Interaction Scoring: The fused features for both metabolites and proteins are fed into a fully connected layer to predict the probability of interaction.

## Background

Understanding metabolite-protein interactions is crucial for elucidating biological processes and drug mechanisms. This project provides curated datasets and PyTorch-based code for building, training, and evaluating a deep learning model.

## Features

- Ready-to-use datasets for human, E. coli, yeast
- Hypergraph neural network (HGNN) model implementation (PyTorch & torch-geometric)
- Data preprocessing and feature extraction scripts
- Reproducible training and evaluation pipeline
- Large file support (split for GitHub compatibility)

## Directory Structure

```
piazza/           # E. coli dataset
PMIDB/human/      # Human protein-metabolite interaction data
stitch_ecoli/     # E. coli dataset
stitch_yeast/     # Yeast dataset
Model.py          # Model definition (HGNN)
Prepare.py        # Data preprocessing and feature generation
main.py           # Main training & evaluation script
utils.py          # Utility functions (metrics, matrix conversion, etc.)
requirements.txt  # Python dependencies
```

## Dataset Details

- **edges.csv / m_p_links.csv**: Protein-metabolite interaction pairs
- **m_m_links.tsv / p_p_links.tsv**: Metabolite-metabolite & protein-protein similarity networks
- **meta.smi**: metabolite SMILES strings
- **meta_ChemGPT-19M.npy / protein_large_model.npy**: Precomputed feature matrices
- **protein_edge.edgelist / meta_edge.edgelist**: Graph edge lists for network construction
- **p_p_links_part*.tsv**: Large files split for GitHub upload

## Model & Code Overview

- `Model.py`: Implement a hypergraph neural network (HGNN) for learning on protein/metabolite graphs
- `Prepare.py`: Load and processe raw data, build graph structures, generate features
- `main.py`: Orchestrate training, cross-validation, and evaluation; support GPU/CPU
- `utils.py`: Helper functions for matrix conversion, metrics, and data handling

## Installation & Usage

1. Clone the repository:
	```bash
	git clone https://github.com/ztjin958/PMPIHGLL.git
	cd PMPIHGLL
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Prepare data (if needed, see Prepare.py for details)
4. Run the main script:
	```bash
	python main.py
	```

## Requirements

- Python 3.7+
- numpy, pandas, tqdm, torch, torch-geometric, scikit-learn

## Citation & Contribution

If you use this project, please cite:
https://github.com/ztjin958/PMPIHGLL

Contributions, issues, and pull requests are welcome!



