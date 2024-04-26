# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-29 06:06:37
# %%
from sklearn.decomposition import PCA
import gnn_tools
import torch
import numpy as np
import torch_geometric
import torch
import torch
import numpy as np

#
# Embedding models
#
embedding_models = {}
embedding_model = lambda f: embedding_models.setdefault(f.__name__, f)


class LinkPredictor(torch.nn.Module):
    def predict(network, src, trg):
        raise NotImplementedError()

    def train(network, **params):
        raise NotImplementedError()

    def state_dict(self):
        d = super().state_dict()
        d.update(self.params)
        d["model"] = self.model
        return d

    def load(self, filename):
        raise NotImplementedError()


class EmbeddingLinkPredictor(LinkPredictor):
    def __init__(self, model, **params):
        super().__init__()
        self.model = model
        self.params = params
        self.embedding_models = embedding_models

    def train(self, network, **params):
        emb_func = embedding_models[self.model]
        emb = emb_func(network=network, **self.params)
        self.emb = torch.nn.Parameter(torch.FloatTensor(emb), requires_grad=False)

    def predict(self, network, src, trg, **params):
        return torch.sum(self.emb[src, :] * self.emb[trg, :], axis=1).reshape(-1)

    def load(self, filename):
        d = torch.load(filename)
        self.model = d["model"]
        self.emb = d["emb"]


# ==============================
# Graph embeddings
# ==============================


@embedding_model
def line(network, dim, num_walks=40, **params):
    model = gnn_tools.embeddings.Node2Vec(window_length=1, num_walks=num_walks)
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def node2vec(network, dim, window_length=10, num_walks=40, **params):
    model = gnn_tools.embeddings.Node2Vec(
        window_length=window_length, num_walks=num_walks
    )
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def deepwalk(network, dim, window_length=10, num_walks=40, **params):
    model = gnn_tools.embeddings.DeepWalk(
        window_length=window_length, num_walks=num_walks
    )
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def leigenmap(network, dim, **params):
    model = gnn_tools.embeddings.LaplacianEigenMap()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def modspec(network, dim, **params):
    model = gnn_tools.embeddings.ModularitySpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def nonbacktracking(network, dim, **params):
    model = gnn_tools.embeddings.NonBacktrackingSpectralEmbedding()
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def fastrp(network, dim, window_length=5, inner_dim=2048, **params):
    model = gnn_tools.embeddings.FastRP(window_size=window_length)
    model.fit(network)
    emb = model.transform(dim=inner_dim)
    return PCA(n_components=dim).fit_transform(emb)


@embedding_model
def SGTLaplacianExp(network, dim, **params):
    model = gnn_tools.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="laplacian"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTLaplacianNeumann(network, dim, **params):
    model = gnn_tools.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="laplacian"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTAdjacencyExp(network, dim, **params):
    model = gnn_tools.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTAdjacencyNeumann(network, dim, **params):
    model = gnn_tools.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTNormAdjacencyExp(network, dim, **params):
    model = gnn_tools.embeddings.SpectralGraphTransformation(
        kernel_func="exp", kernel_matrix="normalized_A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SGTNormAdjacencyNeumann(network, dim, **params):
    model = gnn_tools.embeddings.SpectralGraphTransformation(
        kernel_func="neu", kernel_matrix="normalized_A"
    )
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def dcSBM(network, dim, **params):
    model = gnn_tools.embeddings.SBMEmbedding()
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


# ==============================
# Generic graph neural networks
# ==============================
def gnn_embedding(
    model,
    network,
    in_channels,
    memberships=None,
    device=None,
    epochs=50,
    negative_edge_sampler=None,
    **params
):
    if device is None:
        device = gnn_tools.train.get_gpu_id()

    if memberships is None:
        model, emb = gnn_tools.train.link_prediction_task(
            model=model,
            feature_vec=None,
            feature_vec_dim=in_channels,
            net=network,
            negative_edge_sampler=negative_edge_sampler,
            device=device,
            epochs=epochs,
            **params,
        )
    else:
        model, emb = gnn_tools.train.community_detection_task(
            model=model,
            feature_vec=None,
            feature_vec_dim=in_channels,
            memberships=memberships,
            net=network,
            negative_edge_sampler=negative_edge_sampler,
            device=device,
            epochs=epochs,
            **params,
        )
    return emb


@embedding_model
def GCN(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=64,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GCN(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["uniform"],
        epochs=epochs,
        device=device,
        memberships=memberships,
        **params,
    )


@embedding_model
def GIN(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def PNA(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.PNA(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            aggregators=["sum", "mean", "min", "max", "max", "var", "std"],
            scalers=[
                "identity",
                "amplification",
                "attenuation",
                "linear",
                "inverse_linear",
            ],
            deg=torch.FloatTensor(
                np.bincount(np.array(network.sum(axis=0)).reshape(-1))
            ),
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def EdgeCNN(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def GraphSAGE(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphSAGE(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["uniform"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def GAT(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=64,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GAT(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        device=device,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["uniform"],
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def dcGCN(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=64,
    epochs=100,
    dropout=0.2,
    memberships=None,
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GCN(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["degreeBiased"],
        epochs=epochs,
        device=device,
        memberships=memberships,
        **params,
    )


@embedding_model
def dcGIN(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GIN(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        network=network,
        in_channels=dim,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["degreeBiased"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def dcEdgeCNN(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["degreeBiased"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def dcGraphSAGE(
    network,
    dim,
    device=None,
    dim_h=64,
    num_layers=2,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphSAGE(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["degreeBiased"],
        device=device,
        epochs=epochs,
        memberships=memberships,
        **params,
    )


@embedding_model
def dcGAT(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=64,
    epochs=100,
    dropout=0.2,
    memberships=None,
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GAT(
            in_channels=dim,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=gnn_tools.train.NegativeEdgeSampler["degreeBiased"],
        epochs=epochs,
        device=device,
        memberships=memberships,
        **params,
    )
