# Running AdaQP

This section provides details on running AdaQP with multi-node training and dataset partitioning. The data partitioning process is adapted from the original author's repository: [AdaQP Repo](https://github.com/raywan-110/AdaQP).

## Datasets

We conducted experiments on the following datasets:
- `pubmed`
- `reddit`
- `ogbn-arxiv`
- `ogbn-products`
- `ogbn-papers100M`

## Partitioning and Running

We followed the data partitioning process as detailed in the original AdaQP repository. Scripts for partitioning and configurations for running the experiments are provided in this repository for all datasets listed above.

## Scripts and Configurations

The provided scripts include:
- **Data Partitioning Script**: Automates the partitioning process for supported datasets.
- **Run Configuration**: Sample configurations for multi-node training and hyperparameter settings used during the experiments.

Refer to the respective script files in this repository for more details.
