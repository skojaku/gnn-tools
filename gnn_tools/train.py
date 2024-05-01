# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-10 04:51:58
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-20 05:52:09
# %%
import numpy as np
from scipy import sparse
from tqdm import tqdm
import igraph as ig
import copy
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Data, ClusterData
from torch_geometric.utils import index_sort, narrow, select, sort_edge_index
from torch_geometric.utils.sparse import index2ptr, ptr2index
from torch_geometric.data import Data
from torch_geometric.loader import ClusterLoader
import GPUtil
from adabelief_pytorch import AdaBelief
from gnn_tools.LinkPredictionDataset import LinkPredictionDataset


# ================================
# Trainer for link prediction task
# ================================
def link_prediction_task(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    net: sparse.spmatrix,
    device: str,
    epochs: int,
    feature_vec_dim: int = 64,
    negative_edge_sampler=None,
    batch_size: int = 2500,
    resolution=2.0,
    lr=1e-3,
) -> torch.nn.Module:
    """
    Train a PyTorch model on a given graph dataset using minibatch stochastic gradient descent with negative sampling.

    Parameters
    ----------
    model : nn.Module
        A PyTorch module representing the model to be trained.
    feature_vec : np.ndarray
        A numpy array of shape (num_nodes, num_features) containing the node feature vectors for the graph.
    net : sp.spmatrix
        A scipy sparse matrix representing the adjacency matrix of the graph.
    device : str
        The device to use for training the model.
    epochs : int
        The number of epochs to train the model for.
    negative_edge_sampler : str
        Name of the negative edge sampler to use. Options are "degreeBiased" and "uniform", and "randomWalk".
    batch_size : int, optional
        The number of nodes in each minibatch.
    resolution : float, optional
        The resolution parameter for the modularity clustering algorithm.

    Returns
    -------
    nn.Module
        The trained model.
    """
    n_nodes = net.shape[0]

    # Convert sparse adjacency matrix to edge list format
    r, c, _ = sparse.find(net)
    edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))

    if feature_vec is None:
        feature_vec = generate_base_embedding(net, feature_vec_dim)
        feature_vec = torch.FloatTensor(feature_vec)

    # Create PyTorch data object with features and edge list
    data = Data(edge_index=edge_index, x=feature_vec)

    # Set up minibatching for the data using a clustering algorithm
    num_sub_batches = 5
    batch_size = np.minimum(n_nodes, batch_size)
    cluster_data = ModularityClusterData(
        data, resolution=resolution
    )  # 1. Create subgraphs.
    train_loader = ClusterLoader(
        cluster_data, batch_size=num_sub_batches, shuffle=False
    )  # 2. Stochastic partioning scheme.

    # Use default negative sampling function if none is specified
    if negative_edge_sampler is None:
        negative_edge_sampler = "uniform"

    # Set the model in training mode and initialize optimizer
    model.to(device)
    model.train()

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = AdaBelief(
        model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )

    # Train the model for the specified number of epochs
    pbar = tqdm(total=epochs)

    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        # Iterate over minibatches of the data
        ave_loss = 0
        n_iter = 0
        for sub_data in train_loader:
            optimizer.zero_grad()

            # Sample negative edges using specified or default sampler
            _edge_index = sub_data.edge_index  # positive edges
            x = sub_data.x

            # To scipy sparse matrix
            _n_nodes = x.size()[0]
            _net = sparse.csr_matrix(
                (
                    np.ones_like(_edge_index[0]),
                    (_edge_index[0].numpy(), _edge_index[1].numpy()),
                ),
                shape=(_n_nodes, _n_nodes),
            )

            sampled = False
            for testEdgeFraction in [0.15, 0.10, 0.05]:
                try:
                    test_net, test_edges = (
                        LinkPredictionDataset(
                            testEdgeFraction=testEdgeFraction,
                            negative_edge_sampler=negative_edge_sampler,
                            duplicated_negative_edges=True,
                        )
                        .fit(_net)
                        .transform()
                    )
                    sampled = True
                except:
                    continue
            if sampled == False:
                continue

            src, trg, _ = sparse.find(test_net)
            _train_edge_index = torch.LongTensor(np.vstack([src, trg])).to(device)
            src_test, trg_test, y = (
                test_edges["src"],
                test_edges["trg"],
                test_edges["isPositiveEdge"],
            )

            x = x.to(device)
            # neg_edge_index = neg_edge_index.to(device)

            # Negative edge injection
            z = model(x, _train_edge_index)
            # "            neg_edge_index = negative_edge_sampler(
            # "                edge_index=_edge_index,
            # "                num_nodes=x.shape[0],
            # "                num_neg_samples=2,
            # "            )

            # Zero-out gradient, compute embeddings and logits, and calculate loss
            score = (z[src_test, :] * z[trg_test, :]).sum(dim=1)
            loss = criterion(score, torch.FloatTensor(y).to(device))

            # Compute gradients and update parameters of the model
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ave_loss += loss.item()
                n_iter += 1
        pbar.update(1)
        ave_loss /= n_iter
        pbar.set_description(f"loss={ave_loss:.3f} iter/epoch={n_iter}")

    # Set the model in evaluation mode and return
    model.eval()
    emb = model(feature_vec.to(device), edge_index.to(device))
    emb = emb.detach().cpu().numpy()
    return model, emb


#
# Node pair samplers
#
# def degreeBiasedNegativeEdgeSampling(edge_index, num_nodes, num_neg_samples):
#    deg = np.bincount(edge_index.reshape(-1).cpu(), minlength=num_nodes).astype(float)
#    deg /= np.sum(deg)
#    t = np.random.choice(
#        num_nodes, p=deg, size=num_neg_samples * edge_index.size()[1]
#    ).reshape((num_neg_samples, edge_index.size()[1]))
#    return torch.LongTensor(t)
#
#
# def negative_uniform(edge_index, num_nodes, num_neg_samples):
#    t = np.random.randint(
#        0, num_nodes, size=num_neg_samples * edge_index.size()[1]
#    ).reshape((num_neg_samples, edge_index.size()[1]))
#    return torch.LongTensor(t)
#
#
# NegativeEdgeSampler = {
#    "degreeBiased": degreeBiasedNegativeEdgeSampling,
#    "uniform": negative_uniform,
# }


# ====================================
# Trainer for community detection task
# ====================================
def community_detection_task(
    model: torch.nn.Module,
    feature_vec: np.ndarray,
    net: sparse.spmatrix,
    memberships: np.ndarray,
    device: str,
    epochs: int,
    feature_vec_dim: int = 64,
    negative_edge_sampler=None,
    batch_size: int = 2500,
    resolution=2.0,
    lr=1e-3,
) -> torch.nn.Module:
    n_nodes = net.shape[0]

    # Convert sparse adjacency matrix to edge list format
    r, c, _ = sparse.find(net)
    edge_index = torch.LongTensor(np.array([r.astype(int), c.astype(int)]))

    if feature_vec is None:
        feature_vec = generate_base_embedding(net, feature_vec_dim)
        feature_vec = torch.FloatTensor(feature_vec)

    # Create PyTorch data object with features and edge list
    memberships = torch.LongTensor(memberships)
    data = Data(edge_index=edge_index, x=feature_vec, membership=memberships)

    # Set up minibatching for the data using a clustering algorithm
    num_sub_batches = 5
    batch_size = np.minimum(n_nodes, batch_size)
    num_parts = np.maximum(2, int(n_nodes / batch_size))
    # cluster_data = ClusterData(data, num_parts=num_parts)  # 1. Create subgraphs.
    cluster_data = ModularityClusterData(
        data, resolution=resolution
    )  # 1. Create subgraphs.
    train_loader = ClusterLoader(
        cluster_data, batch_size=num_sub_batches, shuffle=False
    )  # 2. Stochastic partioning scheme.

    # Use default negative sampling function if none is specified
    if negative_edge_sampler is None:
        negative_edge_sampler = negative_uniform

    # Set the model in training mode and initialize optimizer
    model.to(device)
    model.train()

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = AdaBelief(
        model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )

    # Train the model for the specified number of epochs
    pbar = tqdm(total=epochs)
    logsigmoid = torch.nn.LogSigmoid()
    min_size = 5
    for epoch in range(epochs):
        # Iterate over minibatches of the data
        ave_loss = 0
        n_iter = 0
        for sub_data in train_loader:
            # Sample negative edges using specified or default sampler
            _edge_index = sub_data.edge_index  # positive edges
            x = sub_data.x
            node_label = sub_data.membership
            src_pos, trg_pos, src_neg, trg_neg = sample_community_membership_pairs(
                membership=node_label, n_samples=_edge_index.size()[1]
            )
            x = x.to(device)
            _edge_index = _edge_index.to(device)
            src_pos = torch.LongTensor(src_pos).to(device)
            trg_pos = torch.LongTensor(trg_pos).to(device)
            src_neg = torch.LongTensor(src_neg).to(device)
            trg_neg = torch.LongTensor(trg_neg).to(device)

            if len(trg_neg) <= min_size:
                continue
            if len(trg_pos) <= min_size:
                continue

            # Zero-out gradient, compute embeddings and logits, and calculate loss
            optimizer.zero_grad()

            # Negative edge injection
            z = model(x, _edge_index)

            pos = (z[src_pos, :] * z[trg_pos, :]).sum(dim=1)
            ploss = -pos.mean()
            ploss = -logsigmoid(pos).mean()
            neg = (z[src_neg, :] * z[trg_neg, :]).sum(dim=1)
            nloss = -logsigmoid(neg.neg()).mean()
            if nloss.isnan():
                print("sada", nloss, neg.neg())
            loss = ploss + nloss

            # Compute gradients and update parameters of the model
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ave_loss += loss.item()
                n_iter += 1
        pbar.update(1)
        ave_loss /= n_iter
        pbar.set_description(f"loss={ave_loss:.3f} iter/epoch={n_iter}")

    # Set the model in evaluation mode and return
    model.eval()
    emb = model(feature_vec.to(device), edge_index.to(device))
    emb = emb.detach().cpu().numpy()
    return model, emb


def sample_community_membership_pairs(membership, n_samples):

    coms, membership = np.unique(membership, return_inverse=True)
    n_nodes = len(membership)
    n_coms = len(coms)

    n_samples_com = np.bincount(np.random.randint(0, n_coms, n_samples))

    pos_pairs = set()
    for i in range(n_coms):
        if n_samples_com[i] == 0:
            continue
        nodes = np.where(membership == i)[0]
        src, trg = tuple(
            np.random.choice(nodes, size=n_samples_com[i] * 2).reshape((-1, 2)).T
        )
        pos_pairs.update(set(list(src + trg * 1j)))

    neg_pairs = set()
    n_iter = 0
    n_max_iter = 10
    while len(neg_pairs) < n_samples and n_iter < n_max_iter:
        _target_n_samples = n_samples - len(neg_pairs)
        src = np.random.randint(0, n_nodes, size=_target_n_samples)
        trg = np.random.randint(0, n_nodes, size=_target_n_samples)
        s = membership[src] != membership[trg]
        n_iter += 1
        src, trg = src[s], trg[s]
        if len(src) == 0:
            continue
        neg_pairs.update(set(list(src + trg * 1j)))

    pos_pairs = list(pos_pairs)
    neg_pairs = list(neg_pairs)
    src_pos, trg_pos = np.real(pos_pairs), np.imag(pos_pairs)
    src_neg, trg_neg = np.real(neg_pairs), np.imag(neg_pairs)
    src_pos, trg_pos, src_neg, trg_neg = (
        src_pos.astype(int),
        trg_pos.astype(int),
        src_neg.astype(int),
        trg_neg.astype(int),
    )

    assert np.all(membership[src_pos] == membership[trg_pos])
    assert np.all(membership[src_neg] != membership[trg_neg])

    return src_pos, trg_pos, src_neg, trg_neg


# ====================
# Utils
# ====================
def generate_embedding(model, net, feature_vec=None, device="cpu"):
    """Generate embeddings using a specified network model.

    Parameters
    ----------
        net (sparse matrix): The input sparse matrix.
        feature_vec (ndarray, optional): The initial feature vector. If not provided,
            the base embedding will be used instead. Defaults to None.
        device (str, optional): The device to use for computations. Defaults to "cpu".

    Returns
    -------
        ndarray: The resulting embeddings as a numpy array on the CPU.
    """
    rows, cols, _ = sparse.find(
        net
    )  # Find the row and column indices of non-zero elements in the sparse matrix
    edge_index = torch.LongTensor(np.array([rows.astype(int), cols.astype(int)])).to(
        device
    )  # Convert the indices to a tensor and move it to the specified device

    if feature_vec is None:
        feature_vec = generate_base_embedding(net, dim=model.in_channels)

    embeddings = model.forward(
        torch.FloatTensor(feature_vec).to(device), edge_index
    )  # Generate embeddings using the model
    return (
        embeddings.detach().cpu().numpy()
    )  # Detach the embeddings from the computation graph and convert it to a numpy array on the CPU


def generate_base_embedding(A, dim):
    """
    Compute the base embedding using the input adjacency matrix.

    Parameters
    ----------
    A (numpy.ndarray): Input adjacency matrix

    Returns
    -------
    numpy.ndarray: Base embedding computed using normalized laplacian matrix
    """
    # svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
    # base_emb = svd.fit_transform(A)
    a = np.sqrt(6.0 / (2 * dim))
    base_emb = np.random.uniform(-a, a, size=(A.shape[0], dim))
    return base_emb


def get_gpu_id(excludeID=[]):
    device = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=1,
        maxMemory=0.3,
        attempts=99999,
        interval=60 * 1,
        verbose=False,
        excludeID=excludeID,
    )[0]
    device = f"cuda:{device}"
    return device


# ================================
# Dataset
# ================================
@dataclass
class Partition:
    indptr: Tensor
    index: Tensor
    partptr: Tensor
    node_perm: Tensor
    edge_perm: Tensor
    sparse_format: Literal["csr", "csc"]


class ModularityClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    .. note::
        The underlying METIS algorithm requires undirected graphs as input.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (str, optional): If set, will save the partitioned data to the
            :obj:`save_dir` directory for faster re-use. (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
        keep_inter_cluster_edges (bool, optional): If set to :obj:`True`,
            will keep inter-cluster edge connections. (default: :obj:`False`)
        sparse_format (str, optional): The sparse format to use for computing
            partitions. (default: :obj:`"csr"`)
    """

    def __init__(
        self,
        data,
        resolution: float = 2.0,
        recursive: bool = False,
        save_dir: Optional[str] = None,
        log: bool = False,
        keep_inter_cluster_edges: bool = False,
        sparse_format: Literal["csr", "csc"] = "csr",
    ):
        assert data.edge_index is not None
        assert sparse_format in ["csr", "csc"]

        self.recursive = recursive
        self.keep_inter_cluster_edges = keep_inter_cluster_edges
        self.sparse_format = sparse_format

        recursive_str = "_recursive" if recursive else ""
        filename = f"metis_{recursive_str}.pt"
        path = osp.join(save_dir or "", filename)

        if save_dir is not None and osp.exists(path):
            self.partition = torch.load(path)
        else:
            if log:  # pragma: no cover
                print("Computing Modularity partitioning...", file=sys.stderr)

            cluster = self._modularity(data.edge_index, data.num_nodes, resolution)
            self.num_parts = torch.max(cluster) + 1
            self.partition = self._partition(data.edge_index, cluster)

            if save_dir is not None:
                torch.save(self.partition, path)

            if log:  # pragma: no cover
                print("Done!", file=sys.stderr)

        self.data = self._permute_data(data, self.partition)

    def _modularity(
        self, edge_index: Tensor, num_nodes: int, resolution: float
    ) -> Tensor:
        # Computes a node-level partition assignment vector via METIS.
        if self.sparse_format == "csr":  # Calculate CSR representation:
            row, index = sort_edge_index(edge_index, num_nodes=num_nodes)
            indptr = index2ptr(row, size=num_nodes)
        else:  # Calculate CSC representation:
            index, col = sort_edge_index(
                edge_index, num_nodes=num_nodes, sort_by_row=False
            )
            indptr = index2ptr(col, size=num_nodes)
        src, trg = edge_index[0], edge_index[1]
        num_nodes = (
            torch.maximum(torch.max(edge_index[0]), torch.max(edge_index[1])) + 1
        )
        g = ig.Graph(zip(src.tolist(), trg.tolist()))
        memberships = g.community_leiden("modularity", resolution=resolution).membership
        cluster = torch.tensor(memberships)
        return cluster

    def _partition(self, edge_index: Tensor, cluster: Tensor) -> Partition:
        # Computes node-level and edge-level permutations and permutes the edge
        # connectivity accordingly:

        # Sort `cluster` and compute boundaries `partptr`:
        cluster, node_perm = index_sort(cluster, max_value=self.num_parts)
        partptr = index2ptr(cluster, size=self.num_parts)

        # Permute `edge_index` based on node permutation:
        edge_perm = torch.arange(edge_index.size(1), device=edge_index.device)
        arange = torch.empty_like(node_perm)
        arange[node_perm] = torch.arange(cluster.numel(), device=cluster.device)
        edge_index = arange[edge_index]

        # Compute final CSR representation:
        (row, col), edge_perm = sort_edge_index(
            edge_index,
            edge_attr=edge_perm,
            num_nodes=cluster.numel(),
            sort_by_row=self.sparse_format == "csr",
        )
        if self.sparse_format == "csr":
            indptr, index = index2ptr(row, size=cluster.numel()), col
        else:
            indptr, index = index2ptr(col, size=cluster.numel()), row

        return Partition(
            indptr, index, partptr, node_perm, edge_perm, self.sparse_format
        )

    def _permute_data(self, data: Data, partition: Partition) -> Data:
        # Permute node-level and edge-level attributes according to the
        # calculated permutations in `Partition`:
        out = copy.copy(data)
        for key, value in data.items():
            if key == "edge_index":
                continue
            elif data.is_node_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, partition.node_perm, dim=cat_dim)
            elif data.is_edge_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, partition.edge_perm, dim=cat_dim)
        out.edge_index = None

        return out

    def __len__(self) -> int:
        return self.partition.partptr.numel() - 1

    def __getitem__(self, idx: int) -> Data:
        node_start = int(self.partition.partptr[idx])
        node_end = int(self.partition.partptr[idx + 1])
        node_length = node_end - node_start

        indptr = self.partition.indptr[node_start : node_end + 1]
        edge_start = int(indptr[0])
        edge_end = int(indptr[-1])
        edge_length = edge_end - edge_start
        indptr = indptr - edge_start

        if self.sparse_format == "csr":
            row = ptr2index(indptr)
            col = self.partition.index[edge_start:edge_end]
            if not self.keep_inter_cluster_edges:
                edge_mask = (col >= node_start) & (col < node_end)
                row = row[edge_mask]
                col = col[edge_mask] - node_start
        else:
            col = ptr2index(indptr)
            row = self.partition.index[edge_start:edge_end]
            if not self.keep_inter_cluster_edges:
                edge_mask = (row >= node_start) & (row < node_end)
                col = col[edge_mask]
                row = row[edge_mask] - node_start

        out = copy.copy(self.data)

        for key, value in self.data.items():
            if key == "num_nodes":
                out.num_nodes = node_length
            elif self.data.is_node_attr(key):
                cat_dim = self.data.__cat_dim__(key, value)
                out[key] = narrow(value, cat_dim, node_start, node_length)
            elif self.data.is_edge_attr(key):
                cat_dim = self.data.__cat_dim__(key, value)
                out[key] = narrow(value, cat_dim, edge_start, edge_length)
                if not self.keep_inter_cluster_edges:
                    out[key] = out[key][edge_mask]

        out.edge_index = torch.stack([row, col], dim=0)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num_parts})"
