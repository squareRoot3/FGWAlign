# FGWAlign

This is the official implementation of the following manuscript:
> Fused Gromov-Wasserstein Alignment for Graph Edit Distance Computation and Beyond

## Directory Structure

```
.
├── dataset/                   # All datasets
│   ├── AIDS/                  # AIDS and AIDS700 datasets
│   ├── IMDB/                  # IMDB dataset  
│   ├── Linux/                 # Linux dataset
│   └── generate_synthetic_dataset.py  # Code for synthetic dataset generation
├── results/                   # Experimental results
├── src/                       # Source code
    ├── baselines.py           # Assignment-based and search-based GED baselines
    ├── FGWAlign.py               # Implementation of the proposed FGWAlign method
    ├── test_multirel.py       # Evaluation code for multi-relational graphs
    ├── test_real.py           # Evaluation code for real-world graphs
    └── test_synthetic.py      # Evaluation code for synthetic graphs
├── readme.md
└── tutorial.ipynb
```

## Setup

```bash
# Create and activate a python virtual environment
pip install --upgrade pip
pip install virtualenv
virtualenv FGWAlign
source FGWAlign/bin/activate

# # For anaconda user
# conda create -n FGWAlign
# conda activate FGWAlign
# conda install pip

# Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # change cu118 to your CUDA version (cu121/cpu)
pip install pygmtools pot networkx scipy memory_profiler munkres
```

## Datasets

- Real-world datasets are included in the `dataset` folder, sourced from [GEDGNN](https://github.com/ChengzhiPiao/GEDGNN) and [TagSim](https://github.com/jiyangbai/TaGSim).
- Generate synthetic datasets:
  ```bash
  python dataset/generate_synthetic_dataset.py
  ```

## Experiments

### Getting Started

We recommend starting with `tutorial.ipynb` which demonstrates FGWAlign on the example in Figures 1 and 2 from our manuscript.

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
# For each dataset (AIDS/Linux/IMDB), test different patience values
for dataset in AIDS Linux IMDB; do
    for patience in 1 2 5 10 20; do
        python src/test_real.py --dataset $dataset --patience $patience
    done
done
```

3. Diverse Projection Candidates (K)
```bash
# For each dataset (AIDS/Linux/IMDB), test different topk values
for dataset in AIDS Linux IMDB; do
    for topk in 1 2 5 10 20; do
        python src/test_real.py --dataset AIDS --topk $topk
    done
done
```

### Graph Alignment Experiments

```bash
python test_align.py --dataset douban
python test_align.py --dataset movie
python test_align_batch.py  # For the megadiff_changes dataset.
```
