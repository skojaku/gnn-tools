# gnn-tools

Repository for GNN tools for link prediction and community detection.

### Install

To install the `gnn-tools` package, you can use pip directly from the GitHub repository. Run the following command in your terminal:
```
pip install git+https://github.com/skojaku/gnn-tools.git
```

Additionally, install the following packages:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [igraph](https://python.igraph.org/en/stable/)
- [GPUtil](https://pypi.org/project/GPUtil/)
- [tqdm](https://pypi.org/project/tqdm/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [networkx](https://networkx.org/documentation/stable/install.html)
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [numpy](https://numpy.org/install/)
- [scipy](https://scipy.org/install/)

### Example

- [notebook/example_01.ipynb](./notebook/example_01.ipynb)

### Usage

The package implements two training tasks, link prediction and community detection.

#### Link prediction

```python
# Karate club network
import networkx as nx
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

# Train GAT on the karate club network
emb = gnn_tools.models.GAT(A, dim=32)
```

#### Community detection

```python
# Karate club network
import networkx as nx
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

# Ground truth community membership
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]

# Train GAT on the karate club network
emb = gnn_tools.models.GAT(A, dim=32, memberships=labels) # pass ground truth community membership
```

### Model

### Available Models

Below is a table of the available models in the `gnn-tools` package, along with a brief description of each:

| Model Name         | Description                                           |
|--------------------|-------------------------------------------------------|
| `line`             | LINE model for network embedding.                     |
| `node2vec`         | Node2Vec model for network embedding.                 |
| `deepwalk`         | DeepWalk model for network embedding.                 |
| `leigenmap`        | Laplacian Eigenmap for network embedding.             |
| `modspec`          | Modularity Spectral Embedding for network embedding.  |
| `nonbacktracking` | Non-Backtracking Spectral Embedding.                  |
| `fastrp`           | Fast Random Projection (FastRP) embedding.            |
| `SGTLaplacianExp`  | Spectral Graph Transformation with exponential kernel using Laplacian matrix. |
| `SGTLaplacianNeumann` | Spectral Graph Transformation with Neumann kernel using Laplacian matrix. |
| `SGTAdjacencyExp`  | Spectral Graph Transformation with exponential kernel using adjacency matrix. |
| `SGTAdjacencyNeumann` | Spectral Graph Transformation with Neumann kernel using adjacency matrix. |
| `SGTNormAdjacencyExp` | Spectral Graph Transformation with exponential kernel using normalized adjacency matrix. |
| `SGTNormAdjacencyNeumann` | Spectral Graph Transformation with Neumann kernel using normalized adjacency matrix. |
| `dcSBM`            | Degree-corrected Stochastic Block Model embedding.    |
| `GCN`              | Graph Convolutional Network embedding.                |
| `GIN`              | Graph Isomorphism Network embedding.                  |
| `PNA`              | Principal Neighbors Aggregation embedding.            |
| `EdgeCNN`          | Edge Convolutional Neural Network embedding.          |
| `GraphSAGE`        | Graph Sample and Aggregate embedding.                 |
| `GAT`              | Graph Attention Network embedding.                    |
| `dcGCN`            | Degree-corrected Graph Convolutional Network embedding. |
| `dcGIN`            | Degree-corrected Graph Isomorphism Network embedding. |
| `dcEdgeCNN`        | Degree-corrected Edge Convolutional Neural Network embedding. |
| `dcGraphSAGE`      | Degree-corrected Graph Sample and Aggregate embedding. |
| `dcGAT`            | Degree-corrected Graph Attention Network embedding.   |

These models can be used for various graph-based tasks such as link prediction, community detection, and node classification.
