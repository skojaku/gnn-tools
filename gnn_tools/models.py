# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-07-29 06:06:37
# %%
from itertools import product
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

    n_edges = len(network.data) / 2
    if (n_edges > 300000) and (epochs is None):
        epochs = int(np.maximum(100 * 300000 / n_edges, 3))
    elif epochs is None:
        epochs = 500

    if (n_edges > 300000) and (batch_size is None):
        batch_size = 5000 * 3
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


#
# Hyperparameter tuning
#
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
import torch
from itertools import product
from dclinkpred import LinkPredictionDataset
import copy


def tune_hyperparameters(
    network: sparse.spmatrix,
    dim: int,
    model_class: str,
    feature_vec: Optional[np.ndarray] = None,
    device: str = None,
    val_ratio: float = 0.1,
    negative_edge_sampler: str = "uniform",
    early_stopping: bool = True,
    patience: int = 5,
    **base_params
) -> Dict:
    """
    Tune hyperparameters for GNN models using grid search over all parameter combinations.

    Parameters
    ----------
    network : sparse matrix
        Input network
    dim : int
        Output dimension
    model_class : str
        Name of the GNN model class (e.g., "GCN", "GAT")
    feature_vec : ndarray, optional
        Initial node features
    device : str, optional
        Device to use for training
    val_ratio : float, optional
        Ratio of edges to use for validation
    negative_edge_sampler : str, optional
        Type of negative sampling ("uniform" or "degreeBiased")
    early_stopping : bool, optional
        Whether to use early stopping during training
    patience : int, optional
        Number of epochs to wait before early stopping
    base_params : dict
        Base parameters for the model

    Returns
    -------
    dict
        Best hyperparameters found and their performances
    """
    if device is None:
        device = gnn_tools.train.get_gpu_id()

    # Use LinkPredictionDataset for train/validation split and negative sampling
    lp_dataset = LinkPredictionDataset(
        testEdgeFraction=val_ratio,
        degree_correction=negative_edge_sampler == "degreeBiased",
        negatives_per_positive=1,
        allow_duplicated_negatives=False,
    )
    lp_dataset.fit(network)

    # Get training network and validation edges
    train_network, val_src, val_trg, val_labels = lp_dataset.transform()

    # Define parameter grid
    param_ranges = {
        "num_layers": [1,2],
        "dim_h": [64, 128, 256],
        "dropout": [0.2],
    }

    # Generate all parameter combinations
    param_combinations = list(
        product(
            param_ranges["num_layers"], param_ranges["dim_h"], param_ranges["dropout"]
        )
    )

    # Store results for all combinations
    results = []

    # Try each parameter combination
    for num_layers, dim_h, dropout in param_combinations:
        params = {"num_layers": num_layers, "dim_h": dim_h, "dropout": dropout}

        # Create model with current parameters
        if model_class == "GCN":
            model = torch_geometric.nn.models.GCN(
                in_channels=dim,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=dim,
                dropout=dropout,
            )
        elif model_class == "GAT":
            model = torch_geometric.nn.models.GAT(
                in_channels=-1,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=dim,
                dropout=dropout,
            )
        elif model_class == "GraphSAGE":
            model = torch_geometric.nn.models.GraphSAGE(
                in_channels=-1,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=dim,
                dropout=dropout,
            )

        # Train model with early stopping if enabled
        if early_stopping:
            model, emb, training_history = train_with_early_stopping(
                model=model,
                feature_vec=feature_vec,
                dim = dim,
                train_network=train_network,
                val_src=val_src,
                val_trg=val_trg,
                val_labels=val_labels,
                device=device,
                patience=patience,
                negative_edge_sampler=negative_edge_sampler,
                **base_params,
            )
            best_val_score = max(training_history["val_scores"])
            best_epoch = training_history["val_scores"].index(best_val_score)
        else:
            model, emb = gnn_tools.train.link_prediction_task(
                model=model,
                feature_vec=feature_vec,
                net=train_network,
                device=device,
                epochs=base_params.get("epochs", 100),
                feature_vec_dim=dim,
                negative_edge_sampler=negative_edge_sampler,
                **base_params,
            )
            # Evaluate on validation set
            best_val_score = evaluate_link_prediction(
                embeddings=emb, val_src=val_src, val_trg=val_trg, val_labels=val_labels
            )
            best_epoch = base_params.get("epochs", 100)

        # Store results
        results.append(
            {"params": params, "val_score": best_val_score, "best_epoch": best_epoch}
        )

    # Sort results by validation score
    results.sort(key=lambda x: x["val_score"], reverse=True)

    return results


def train_with_early_stopping(
    model,
    feature_vec,
    dim,
    train_network,
    val_src,
    val_trg,
    val_labels,
    device,
    patience,
    **training_params
) -> Tuple[torch.nn.Module, np.ndarray, Dict]:
    """
    Train model with early stopping based on validation performance.

    Parameters
    ----------
    model : torch.nn.Module
        The GNN model to train
    feature_vec : np.ndarray
        Node features
    train_network : sparse matrix
        Training network
    val_src, val_trg : np.ndarray
        Validation edge endpoints
    val_labels : np.ndarray
        Validation edge labels
    device : str
        Device to use for training
    patience : int
        Number of epochs to wait before early stopping
    training_params : dict
        Additional training parameters

    Returns
    -------
    tuple
        (best_model, best_embeddings, training_history)
    """
    best_val_score = float("-inf")
    best_model = None
    best_embeddings = None
    epochs_without_improvement = 0
    training_history = {"val_scores": []}
    n_edges = len(train_network.data) / 2
    if (n_edges > 300000) and (training_params.get("epochs", 100) is None):
        epochs = int(np.maximum(100 * 300000 / n_edges, 3))
    elif training_params.get("epochs", 100) is None:
        epochs = 500
    else:
        epochs = training_params.get("epochs", 3)
    # Remove "epochs" key from training_params if it exists
    training_params.pop("epochs", None)

    for epoch in range(epochs):
        # Train for one epoch
        model, embeddings = gnn_tools.train.link_prediction_task(
            model=model,
            feature_vec=feature_vec,
            feature_vec_dim=dim,
            net=train_network,
            device=device,
            epochs=1,
            **training_params,
        )

        # Evaluate on validation set
        val_score = evaluate_link_prediction(
            embeddings=embeddings,
            val_src=val_src,
            val_trg=val_trg,
            val_labels=val_labels,
        )
        training_history["val_scores"].append(val_score)

        # Check if this is the best model so far
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)
            best_embeddings = embeddings.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= patience:
            break

    return best_model, best_embeddings, training_history


def evaluate_link_prediction(
    embeddings: np.ndarray,
    val_src: np.ndarray,
    val_trg: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """
    Evaluate link prediction performance on validation set.
    """
    scores = np.sum(embeddings[val_src] * embeddings[val_trg], axis=1)
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(val_labels, scores)


#
# Fine-tuned models
#
@embedding_model
def fineTunedGCN(
    network,
    dim,
    device=None,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):
    tuning_results = tune_hyperparameters(
        network=network,
        dim=dim,
        model_class="GCN",
        device=device,
        epochs=epochs,
        negative_edge_sampler=negative_edge_sampler,
        **params,
    )
    best_params = tuning_results[0]["params"]
    return GCN(
        network=network,
        dim=dim,
        device=device,
        dim_h=best_params["dim_h"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        epochs=epochs,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def fineTunedGAT(
    network,
    dim,
    device=None,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):
    tuning_results = tune_hyperparameters(
        network=network,
        dim=dim,
        model_class="GAT",
        device=device,
        epochs=epochs,
        negative_edge_sampler=negative_edge_sampler,
        **params,
    )
    best_params = tuning_results[0]["params"]
    return GAT(
        network=network,
        dim=dim,
        device=device,
        dim_h=best_params["dim_h"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        epochs=epochs,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def fineTunedGraphSAGE(
    network,
    dim,
    device=None,
    epochs=None,
    dropout=0.2,
    memberships=None,
    negative_edge_sampler="uniform",
    clustering="modularity",
    **params
):
    tuning_results = tune_hyperparameters(
        network=network,
        dim=dim,
        model_class="GraphSAGE",
        device=device,
        epochs=epochs,
        negative_edge_sampler=negative_edge_sampler,
        **params,
    )
    best_params = tuning_results[0]["params"]
    return GraphSAGE(
        network=network,
        dim=dim,
        device=device,
        dim_h=best_params["dim_h"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        epochs=epochs,
        memberships=memberships,
        negative_edge_sampler=negative_edge_sampler,
        clustering=clustering,
        **params,
    )


@embedding_model
def dcFineTunedGCN(
    network,
    dim,
    device=None,
    epochs=None,
    dropout=0.2,
    memberships=None,
    clustering="modularity",
    **params
):
    return fineTunedGCN(
        network=network,
        dim=dim,
        device=device,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler="degreeBiased",
        clustering=clustering,
        **params,
    )


@embedding_model
def dcFineTunedGAT(
    network,
    dim,
    device=None,
    epochs=None,
    dropout=0.2,
    memberships=None,
    clustering="modularity",
    **params
):
    return fineTunedGAT(
        network=network,
        dim=dim,
        device=device,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler="degreeBiased",
        clustering=clustering,
        **params,
    )


@embedding_model
def dcFineTunedGraphSAGE(
    network,
    dim,
    device=None,
    epochs=None,
    dropout=0.2,
    memberships=None,
    clustering="modularity",
    **params
):
    return fineTunedGraphSAGE(
        network=network,
        dim=dim,
        device=device,
        epochs=epochs,
        dropout=dropout,
        memberships=memberships,
        negative_edge_sampler="degreeBiased",
        clustering=clustering,
        **params,
    )
