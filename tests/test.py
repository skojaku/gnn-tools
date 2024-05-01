import unittest
import gnn_tools
from scipy import sparse
import numpy as np
from sklearn.metrics import roc_auc_score


class TestCommunityDetection(unittest.TestCase):

    def setUp(self) -> None:
        import networkx as nx

        G = nx.karate_club_graph()
        A = nx.adjacency_matrix(G)
        self.A = A
        self.labels = np.unique(
            [d[1]["club"] for d in G.nodes(data=True)], return_inverse=True
        )[1]

    def test_linkprediction(self):
        emb = gnn_tools.models.GAT(self.A, dim=32)

        S = emb @ emb.T
        U = sparse.csr_matrix(
            (np.ones_like(self.labels), (np.arange(len(self.labels)), self.labels)),
            shape=(len(self.labels), len(set(self.labels))),
        )
        Sy = (U @ U.T).toarray()

        score = roc_auc_score(Sy.reshape(-1), S.reshape(-1))
        print(f"SCore = {score}")
        assert score > 0.7, f"Test failed with ROC AUC score: {score}"



# %%
# import networkx as nx
# import gnn_tools
# import numpy as np
# from scipy import sparse
# from sklearn.metrics import roc_auc_score
#
# G = nx.karate_club_graph()
# A = nx.adjacency_matrix(G)
# A = A
# labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
#
# emb = gnn_tools.models.GAT(A, dim=32, memberships=labels, lr=4e-4)
#
# S = emb @ emb.T
# U = sparse.csr_matrix(
#    (np.ones_like(labels), (np.arange(len(labels)), labels)),
#    shape=(len(labels), len(set(labels))),
# )
# Sy = (U @ U.T).toarray()
#
# score = roc_auc_score(Sy.reshape(-1), S.reshape(-1))
# print(score)
## %%
# import seaborn as sns
# from sklearn.decomposition import PCA
#
# xy = PCA(n_components=2).fit_transform(emb)
# sns.scatterplot(xy[:, 0], xy[:, 1], hue=labels)
#
## %%
# sns.heatmap(S)
## %%
#
