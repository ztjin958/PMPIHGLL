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
- **Local node-level 5-fold CV** (`main_local_cv.py`) for within-dataset evaluation
- **Transductive cross-source train/test** (e.g. Piazza train → PMIDB/ecoil test)
- Large file support (split for GitHub compatibility; Git LFS for embeddings)

## Directory Structure

```
piazza/                    # Piazza E. coli PMI & features
PMIDB/human/               # Human protein-metabolite interaction data
PMIDB/ecoil/               # PMIDB E. coli PMI & features
piazza_pmidb_ecoil/        # Merged features (piazza + PMIDB/ecoil); see merge script
stitch_ecoli/              # STITCH E. coli (400 / 700 PMI subsets)
stitch_yeast/              # STITCH yeast (400 / 700 PMI subsets)
Model.py                   # Model definition (HGNN)
Prepare.py                 # Legacy data preprocessing
prepare_core.py            # Unified prepare for local CV / main_core
datasets_processed.py      # Dataset paths & links configuration
main.py                    # Main training & evaluation script
main_local_cv.py           # Local 5-fold CV (protein or meta split)
main_stitch_train_piazza_test.py  # Transductive train/test (stitch-piazza | piazza-pmidb)
merge_piazza_pmidb_ecoil.py       # Build piazza_pmidb_ecoil/ merged directory
utils.py                   # Utility functions (metrics, matrix conversion, etc.)
requirements.txt           # Python dependencies
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
3. Prepare data (if needed, see `Prepare.py` / `prepare_core.py` for details)
4. Run the main script (standard edge-level CV):
	```bash
	python main.py
	```

### Local node-level 5-fold CV (`local_cv`)

Within a single dataset, hold out **protein nodes** or **metabolite nodes** for testing (5 folds × R RUS rounds; default R=10, PMIDB human uses R=1).

Supported datasets: `stitch_ecoli_400`, `stitch_ecoli_700`, `stitch_yeast_400`, `stitch_yeast_700`, `piazza`, `pmidb_human`.

```bash
# Generic entry
python main_local_cv.py --dataset piazza --split protein
python main_local_cv.py --dataset stitch_ecoli_400 --split meta

# Per-dataset shortcuts (equivalent)
python main_local_piazza_protein.py
python main_local_stitch_ecoli_400_meta.py
```

Outputs are written under `model_{dataset}_local_{protein|meta}/`.

Processed PMI variants (`*_processed.csv`) and per-dataset READMEs are provided under each data directory.

### Transductive Piazza train → PMIDB/ecoil test (`piazza_pmidb_ecoil`)

Cross-source evaluation: train on Piazza PMI, test on PMIDB/ecoil PMI. The graph uses the **union of nodes** from `piazza_pmidb_ecoil/` (merged m_m / p_p hyperedges and embeddings).

**Step 1 — merge data** (run once, or after updating source PMI):

```bash
python merge_piazza_pmidb_ecoil.py
```

This reads `piazza/` and `PMIDB/ecoil/` and writes `piazza_pmidb_ecoil/`.

**Step 2 — train & test**:

```bash
python main_stitch_train_piazza_test.py --mode piazza-pmidb
```

Optional flags: `--processed` (use `*_processed.csv`), `--no-rus`, `--drop-test-only-proteins FRAC`.

Output directory: `model_stitch_train_piazza_test_piazza_train_pmidb_test/`.

On Windows, if OpenMP conflicts occur: `set KMP_DUPLICATE_LIB_OK=TRUE` before running.

## Requirements

- Python 3.7+
- numpy, pandas, tqdm, torch, torch-geometric, scikit-learn

## Citation & Contribution

If you use this project, please cite:
https://github.com/ztjin958/PMPIHGLL

Contributions, issues, and pull requests are welcome!



