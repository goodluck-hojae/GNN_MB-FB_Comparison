import os
import argparse
import time

import dgl
import dgl.nn as dglnn
import torch.nn.functional as F
import torchmetrics.functional as MF

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
# import tqdm
# from dgl.nn import SAGEConv
# from model import SAGE
# from dgl.data import RedditDataset
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.data import DGLDataset
import pandas as pd

# data_path = '/home/hojaeson_umass_edu/hojae_workspace/vldb/dgl/examples/pytorch/cluster_gcn/dataset/'
data_path = '/work/pi_mserafini_umass_edu/hojae/vldb/dgl/examples/pytorch/cluster_gcn/dataset'
def load_data(dataset):
    if dataset == 'reddit':
        return load_reddit()
    elif dataset == 'ogbn-papers100M':
        return load_subgraph()
    elif 'ogbn' in dataset:
        print("loading ogbn line 35")
        return load_ogb_dataset(dataset)
    elif dataset == 'pubmed':
        return load_pubmed()
    elif dataset == "orkut":
        return load_orkut()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))



def load_subgraph():
    g, _ = dgl.load_graphs('ogbn-papers100M_frac_100.0_hops_2_subgraph.bin')
    # print(g)
    g = g[0]
    # g.ndata['label'] = g.ndata['label'].to(torch.int64)
    n_feat = g.ndata['feat'].shape[1]
    print("train_mask shape = {}".format(g.ndata['train_mask'].shape))
    print("label shape = {}".format(g.ndata['label'].shape))
    
    if g.ndata['label'].dim() == 1:
    
        n_class = int(torch.max(torch.unique(g.ndata['label'][torch.logical_not(torch.isnan(g.ndata['label']))])).item()) + 1 # g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]


    print(g, n_class, 'hi')
    train_ids = torch.nonzero(g.ndata['train_mask']).reshape(-1)
    valid_ids = torch.nonzero(g.ndata['val_mask']).reshape(-1)
    test_ids = torch.nonzero(g.ndata['test_mask']).reshape(-1)
    print(len(test_ids))
    data = (
            n_class, train_ids, valid_ids, test_ids
            )


    return g, n_feat, n_class

 

class OrkutDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="orkut")

    def process(self):
        root = data_path
        edges_data = pd.read_csv(root + "/orkut/orkut/orkut_edges.csv")
        node_labels = pd.read_csv(root + "/orkut/orkut/orkut_labels.csv")


        node_features = torch.load(root + '/orkut/orkut_features.pt')
        # print(f"node_features = {node_features}")

        node_labels = torch.from_numpy(
            node_labels.astype("category").to_numpy()
        ).view(-1)
        # print(f"node_labels = {node_labels}")

        self.num_classes = (node_labels.max() + 1).item()
        # edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())
        # print(f"node_features.shape = {node_features.shape}")
        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=node_features.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        # self.graph.edata["weight"] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

        self.train_idx = self.graph.ndata["train_mask"].nonzero().view(-1)
        self.val_idx = self.graph.ndata["val_mask"].nonzero().view(-1)
        self.test_idx = self.graph.ndata["test_mask"].nonzero().view(-1)


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def load_orkut():
    dataset = OrkutDataset()
    return dataset

def load_reddit():
    root = data_path
    dataset = AsNodePredDataset(dgl.data.RedditDataset(raw_dir=root))
    # graph = dataset[0]
    # graph = dgl.add_reverse_edges(graph)
    # train_nids = np.where(graph.ndata['train_mask'])[0]
    # valid_nids = np.where(graph.ndata['val_mask'])[0]
    # test_nids = np.where(graph.ndata['test_mask'])[0]
    # node_features = graph.ndata['feat']
    # in_feats = node_features.shape[1]
    # n_classes = dataset.num_classes
    
    # g.edata.clear()
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    return dataset


def load_pubmed():
    root = data_path
    dataset = AsNodePredDataset(dgl.data.PubmedGraphDataset(raw_dir=root))
    return dataset


def load_ogb_dataset(name):
    root = data_path
    print(f"root = {root}")
    print("/work/pi_mserafini_umass_edu/hojae/vldb/dgl/examples/pytorch/cluster_gcn/dataset")
    dataset = AsNodePredDataset(DglNodePropPredDataset(name=name, root=root))

    return dataset
    

def ns_evaluate(model, g, n_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=n_classes,
    )

 
def train_evaluate(model, g, num_classes, dataloader):
    model.eval()
    with torch.no_grad():
        train_preds = []
        train_labels = []
        for it, sg in enumerate(dataloader):
            # print(sg.ndata['_ID'])
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m_train = sg.ndata["train_mask"].bool()
            y_hat = model(sg, x)
            train_preds.append(y_hat[m_train])
            train_labels.append(y[m_train])
        train_preds = torch.cat(train_preds, 0)
        train_labels = torch.cat(train_labels, 0)

        if len(train_preds) == 0:
            return torch.zeros(1)

        train_acc = MF.accuracy(
            train_preds,
            train_labels,
            task="multiclass",
            num_classes=num_classes,
        )
        return train_acc

def evaluate(model, g, num_classes, dataloader):
    model.eval()
    with torch.no_grad():
        val_preds, test_preds = [], []
        val_labels, test_labels = [], []
        for it, sg in enumerate(dataloader):
            # print(sg.ndata['_ID'])
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m_val = sg.ndata["val_mask"].bool()
            m_test = sg.ndata["test_mask"].bool()
            y_hat = model(sg, x)
            val_preds.append(y_hat[m_val])
            val_labels.append(y[m_val])
            test_preds.append(y_hat[m_test])
            test_labels.append(y[m_test])
        val_preds = torch.cat(val_preds, 0)
        val_labels = torch.cat(val_labels, 0)
        test_preds = torch.cat(test_preds, 0)
        test_labels = torch.cat(test_labels, 0)
        val_acc = MF.accuracy(
            val_preds,
            val_labels,
            task="multiclass",
            num_classes=num_classes,
        )
        test_acc = MF.accuracy(
            test_preds,
            test_labels,
            task="multiclass",
            num_classes=num_classes,
        )
        return val_acc, test_acc
