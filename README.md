# FGWAlign

This is the official implementation of the following manuscript:
> Fused Gromov-Wasserstein Alignment for Graph Edit Distance Computation and Beyond

For a detailed description of the experimental setup for downstream applications in FGWAlign, please refer to the [Appendix](https://github.com/squareRoot3/FGWAlign/blob/main/appendix.pdf).

## Directory Structure

```clean
.
├── dataset/                      # Datasets
│   ├── AIDS/                     # AIDS and AIDS700 datasets
│   ├── IMDB/                     # IMDB dataset 
│   ├── Linux/                    # Linux dataset
│   └── generate_synthetic_dataset.py  # Script for generating synthetic datasets
├── GABoost                       # Source code for [GABoost](https://github.com/oceaneLIU/GABoost/)
├── src/                          # Source code for FGWAlign
│   ├── baselines.py              # Assignment and search-based GED baselines
│   ├── FGWAlign.py               # Core FGWAlign implementation
│   ├── test_multirel.py          # GED experiments on multi-relational graphs
│   ├── test_real.py              # GED experiments on real-world graphs  
│   └── test_synthetic.py         # GED experiments on synthetic graphs
├── anomaly_detection.py          # Graph-level anomaly detection experiments
├── test_align_batch.py           # Batched graph alignment for megadiff_changes
├── test_align.py                 # Additional alignment experiments
├── tutorial.ipynb                # Getting started tutorial
└── utils.py                      # Utility functions for graph alignment
```

## Setup

```bash
# Create virtual environment
pip install --upgrade pip
pip install virtualenv
virtualenv FGWAlign
source FGWAlign/bin/activate

# For conda users
conda create -n FGWAlign
conda activate FGWAlign
conda install pip

# Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Change cu118 to your CUDA version
pip install pygmtools pot networkx scipy memory_profiler munkres
```

## Datasets

- Real-world datasets are included in the `dataset` folder, sourced from [GEDGNN](https://github.com/ChengzhiPiao/GEDGNN) and [TagSim](https://github.com/jiyangbai/TaGSim).
- Generate the synthetic dataset in our manuscript:
  ```bash
  python dataset/generate_synthetic_dataset.py
  ```

## Experiments

### Getting Started

We recommend starting with `tutorial.ipynb`, which demonstrates FGWAlign on the example in our manuscript.

### FGWAlign Variants on Real-World Datasets

```bash
# Light Version
python src/test_real.py --dataset AIDS --light --topk 1
python src/test_real.py --dataset Linux --light --topk 1
python src/test_real.py --dataset IMDB --light --topk 1
# Fast Version
python src/test_real.py --dataset AIDS
python src/test_real.py --dataset Linux
python src/test_real.py --dataset IMDB
# Full Version
python src/test_real.py --dataset AIDS --patience 20
python src/test_real.py --dataset Linux --patience 20
python src/test_real.py --dataset IMDB --patience 20
```

### GED Baselines

```bash
# Run classic GED baselines on datasets (AIDS/Linux/IMDB)
for dataset in AIDS Linux IMDB; do
    # Assignment-based methods
    for method in RRWM IPFP Spectral VJ; do
        python src/test_real.py --dataset $dataset --method $method
    done
    # A* variants with specific beam settings
    python src/test_real.py --dataset $dataset --method AStar-beam --beam $([ $dataset == "IMDB" ] && echo "1" || echo "5")
    python src/test_real.py --dataset $dataset --method LSa --beam $([ $dataset == "IMDB" ] && echo "5" || echo "100")
    python src/test_real.py --dataset $dataset --method SM
done
```

### Synthetic Dataset Experiments
```bash
# Generate dataset
python dataset/generate_synthetic_dataset.py

# Run Evaluation (GPU by default)
python src/test_syn.py
```

### Ablation Studies

1. Edge Type Modeling (AIDS dataset)

```bash
python src/test_multirel.py --method FGWAlign_rel
python src/test_multirel.py --method FGWAlign
```

2. Random Exploration Patience (T)
```bash
for dataset in AIDS Linux IMDB; do
    for patience in 1 2 5 10 20; do
        python src/test_real.py --dataset $dataset --patience $patience
    done
done
```

3. Diverse Projection Candidates (K)
```bash
for dataset in AIDS Linux IMDB; do
    for topk in 1 2 5 10 20; do
        python src/test_real.py --dataset AIDS --topk $topk
    done
done
```

### Downstream Application I: Graph Alignment

We exactly follow the settings in [GABoost](https://github.com/oceaneLIU/GABoost/) for the evaluation. The datasets include Douban, Movie, and Megadiff_changes, which consist of pairs of heterogeneous graphs with their ground-truth alignments.
```bash
python test_align.py --dataset douban
python test_align.py --dataset movie
# Run the megadiff_changes dataset
unzip GABoost/dataset/megadiff_changes.zip -d GABoost/dataset/
python test_align_batch.py --run_GABoost
```

### Downstream Application II: Graph-level Anomaly Detection

We exactly follow the settings in [SIGNET](https://github.com/yixinliu233/signet) for the evaluation. The task aims to recognize anomalous graphs from a set of graphs by predicting an anomaly score for each graph sample. We evaluate FGWAlign on six benchmark datasets: PROTEINS-F, ENZYMES, AIDS, BZR, DD, and NCI1.
```bash
python anomaly_detection.py --dataset PROTEINS  
# the dataset can be replaced by ENZYMES/AIDS/BZR/DD/NCI1
```