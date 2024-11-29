# Mini-Batch Training Experiments

This repository explores mini-batch training using various samplers, models, and datasets. We iterate through different combinations of these components and parameters to analyze performance and convergence.

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

## Outcomes
We directly parse the information from the log but we intend to update these process with proper settings from the codebase
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
