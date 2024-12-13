# Mini-Batch Training Experiments

This repository explores mini-batch training using various samplers, models, and datasets. We iterate through different combinations of these components and parameters to analyze performance and convergence.

## Getting Started
1. **Install dependencies** : pip install -r requirements.txt
2. **Setting** : --dataset (pubmed, reddit, ogbn-arxiv, ogbn-products, ogbn-papers100), --gpu (e.g. "0,1")
3. **Run python script** : Select a script upon sampler and model. We create separate scripts for tackling the different parameters

## Samplers

The following samplers are implemented for experiments:
1. **Cluster GCN** (referenced from [Cluster GCN codebase](https://github.com/dmlc/dgl/tree/master/examples/pytorch/cluster_gcn))
2. **Node Sampling**
3. **Saint Sampler**

## Models

The models used in experiments are:
1. **GAT**
2. **GCN**
3. **SAGE**

Model architectures can be found in `models.py`.

## Datasets

We use the following datasets:
- `pubmed`
- `ogbn-arxiv`
- `reddit`
- `ogbn-products`
- `ogbn-papers100M`

## Experimental Setup

We iterate over combinations of:
- Samplers
- Models
- Datasets
- Various hyperparameters (details available [here](https://arxiv.org/abs/2406.00552))

## Outcomes (Plot)
After running experiment, the script results in filename_dataset.json \
With this file, run **python3 plot.py --filename "pubmed_NSGAT.json"** to plot a single experiment

(TODO: Plot multiple experiments)

![image](https://github.com/user-attachments/assets/2d3ac93a-56c0-4eda-a5fd-211a3b056979)


The results are recorded with the following metrics:
- `dataset`
- `model`
- `batch_size`
- `budget`
- `num_partition`
- `gpu_model`
- `n_gpu`
- `n_layers`
- `n_hidden`
- `test_accuracy`
- `epoch_time`
- `n_50_accuracy`, `n_50_tta`, `n_50_epoch`
- `n_200_accuracy`, `n_200_tta`, `n_200_epoch`
