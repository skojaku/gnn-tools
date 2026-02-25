# gnn-tools

Repository for GNN tools for link prediction and community detection.

### Install

#### Using mamba + uv (recommended)

[mamba](https://mamba.readthedocs.io/) handles the conda packages that need a specific CUDA build (PyTorch, PyG). [uv](https://docs.astral.sh/uv/) then installs the remaining pip dependencies and the package itself.

**1. Install mamba and uv** (skip if already installed)

```bash
# mamba — fast conda-compatible package manager
conda install -n base -c conda-forge mamba

# uv — fast Python package installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Create the conda environment**

Linux (CUDA 12.1, default):
```bash
mamba env create -f environment.yml
```

macOS (Apple Silicon, CPU):
```bash
mamba env create -f environment-mac.yml
```

> To use a different CUDA version on Linux, edit the `cuda-version` and `pytorch-cuda` lines in `environment.yml` before running.

```bash
mamba activate gnn-tools
```

**3. Install gnn-tools and its pip dependencies**

From GitHub:
```bash
uv pip install git+https://github.com/skojaku/gnn-tools.git
```

Or in editable mode from a local clone:
```bash
git clone https://github.com/skojaku/gnn-tools.git
cd gnn-tools
uv pip install -e .
```

---

#### pip only (no GPU / quick start)

```bash
pip install git+https://github.com/skojaku/gnn-tools.git
```

### Example

- [notebooks/example_01.ipynb](./notebooks/example_01.ipynb)

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


### Implementation details

**Mini-batch Training**: `gnn-tools` utilizes a mini-batch training technique for training graph neural networks to handle large-scale graphs that cannot fit entirely in memory. The mini-batch technique divides the graph into smaller subgraphs or batches, allowing the model to update its parameters incrementally. Specifically, we use the *neighborhood sampling* proposed in [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), which forms a subgraph by randomly sampling the neighbors of a small number of focal nodes.
  - `for the link prediction tasks`, we use the edge-level neighborhood sampling. Namely, a batch is a set of edges, and we sample the neighborhoods of the the end nodes of each edge.
  - `for the community detection tasks`, we use the node-level neighborhood sampling.
