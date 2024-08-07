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
def node2vec(network, dim, window_length=10, num_walks=10, **params):
    model = gnn_tools.embeddings.Node2Vec(
        window_length=window_length, num_walks=num_walks
    )
    model.fit(network)
    return model.transform(dim=dim)


@embedding_model
def deepwalk(network, dim, window_length=10, num_walks=10, **params):
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
    model = gnn_tools.embeddings.SBMEmbedding(degreeCorrected=True)
    model.fit(network)
    emb = model.transform(dim=dim)
    return emb


@embedding_model
def SBM(network, dim, **params):
    model = gnn_tools.embeddings.SBMEmbedding(degreeCorrected=False)
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
    epochs=None,
    negative_edge_sampler=None,
    clustering="modularity",
    batch_size=None,
    **params
):
    if device is None:
        device = gnn_tools.train.get_gpu_id()
    
    n_edges = len(network.data)/2
    if (n_edges > 300000) and (epochs is None):
        epochs = int(np.maximum(100 * 300000 / n_edges, 3))
    elif epochs is None:
        epochs = 500
    
    if (n_edges > 300000) and (batch_size is None):
        batch_size= 5000 * 3
    elif batch_size is None:
        batch_size = 5000

    if memberships is None:
        model, emb = gnn_tools.train.link_prediction_task(
            model=model,
            feature_vec=None,
            feature_vec_dim=in_channels,
            net=network,
            negative_edge_sampler=negative_edge_sampler,
            device=device,
            epochs=epochs,
            clustering=clustering,
            batch_size=batch_size,
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
            clustering=clustering,
            batch_size=batch_size,
            **params,
        )
    return emb


@embedding_model
def GCN(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=128,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):

    return gnn_embedding(
        model=torch_geometric.nn.models.GCN(
            in_channels=-1,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=negative_edge_sampler,
        epochs=epochs,
        device=device,
        memberships=memberships,
        clustering="modularity",
        **params,
    )


@embedding_model
def GIN(
    network,
    dim,
    device=None,
    dim_h=128,
    num_layers=2,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
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
        negative_edge_sampler=negative_edge_sampler,
        device=device,
        epochs=epochs,
        memberships=memberships,
        clustering=clustering,
        **params,
    )


# @embedding_model
# def PNA(
#    network,
#    dim,
#    device=None,
#    dim_h=128,
#    num_layers=2,
#    epochs=None,
#    dropout=0.2,
#    memberships=None,
#    negative_edge_sampler="uniform",
#    **params
# ):
#    return gnn_embedding(
#        model=torch_geometric.nn.models.PNA(
#            in_channels=dim,
#            hidden_channels=dim_h,
#            num_layers=num_layers,
#            out_channels=dim,
#            aggregators=["sum", "mean", "min", "max", "max", "var", "std"],
#            scalers=[
#                "identity",
#                "amplification",
#                "attenuation",
#                "linear",
#                "inverse_linear",
#            ],
#            deg=torch.FloatTensor(
#                np.bincount(np.array(network.sum(axis=0)).reshape(-1).astype(int))
#            ),
#            dropout=dropout,
#        ),
#        in_channels=dim,
#        network=network,
#        negative_edge_sampler=negative_edge_sampler,
#        device=device,
#        epochs=epochs,
#        memberships=memberships,
#        **params,
#    )


@embedding_model
def EdgeCNN(
    network,
    dim,
    device=None,
    dim_h=128,
    num_layers=2,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.EdgeCNN(
            in_channels=-1,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=negative_edge_sampler,
        device=device,
        epochs=epochs,
        memberships=memberships,
        clustering=clustering,
        **params,
    )


@embedding_model
def GraphSAGE(
    network,
    dim,
    device=None,
    dim_h=128,
    num_layers=2,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GraphSAGE(
            in_channels=-1,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        negative_edge_sampler=negative_edge_sampler,
        device=device,
        epochs=epochs,
        memberships=memberships,
        clustering=clustering,
        **params,
    )


@embedding_model
def GAT(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=128,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):
    return gnn_embedding(
        model=torch_geometric.nn.models.GAT(
            in_channels=-1,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=dim,
            dropout=dropout,
        ),
        in_channels=dim,
        network=network,
        device=device,
        negative_edge_sampler=negative_edge_sampler,
        epochs=epochs,
        memberships=memberships,
        clustering=clustering,
        **params,
    )


@embedding_model
def dcGCN(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=128,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="degreeBiased",
    clustering="modularity",
    **params
):
    return GCN(
        network=network,
        dim=dim,
        num_layers=num_layers,
        device=device,
        dim_h=dim_h,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def dcGIN(
    network,
    dim,
    device=None,
    dim_h=128,
    num_layers=2,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="degreeBiased",
    clustering="modularity",
    **params
):
    return GIN(
        network=network,
        dim=dim,
        device=device,
        dim_h=dim_h,
        num_layers=num_layers,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def dcEdgeCNN(
    network,
    dim,
    device=None,
    dim_h=128,
    num_layers=2,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="degreeBiased",
    clustering="modularity",
    **params
):
    return EdgeCNN(
        network=network,
        dim=dim,
        device=device,
        dim_h=dim_h,
        num_layers=num_layers,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def dcGraphSAGE(
    network,
    dim,
    device=None,
    dim_h=128,
    num_layers=2,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="degreeBiased",
    clustering="modularity",
    **params
):
    return GraphSAGE(
        network=network,
        dim=dim,
        device=device,
        dim_h=dim_h,
        num_layers=num_layers,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def dcGAT(
    network,
    dim,
    num_layers=2,
    device=None,
    dim_h=128,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="degreeBiased",
    clustering="modularity",
    **params
):
    
    return GAT(
        network=network,
        dim=dim,
        num_layers=num_layers,
        device=device,
        dim_h=dim_h,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )
