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

    def test_community_detection(self):
        model_names = ["GCN", "GAT", "GraphSAGE"]
        for model_name in model_names:
            emb = gnn_tools.embedding_models[model_name](
                self.A, dim=16, memberships=self.labels
            )

            S = emb @ emb.T
            U = sparse.csr_matrix(
                (np.ones_like(self.labels), (np.arange(len(self.labels)), self.labels)),
                shape=(len(self.labels), len(set(self.labels))),
            )
            Sy = (U @ U.T).toarray()

            score = roc_auc_score(Sy.reshape(-1), S.reshape(-1))
            print(f"{model_name}: {score}")
            # assert score > 0.5, f"test failed for {model_name}. ROC AUC score must be >0.5: {score}"

    def test_embedding_model(self):
        for model_name in gnn_tools.embedding_models.keys():
            emb = gnn_tools.embedding_models[model_name](self.A, dim=16)

            S = emb @ emb.T
            U = sparse.csr_matrix(
                (np.ones_like(self.labels), (np.arange(len(self.labels)), self.labels)),
                shape=(len(self.labels), len(set(self.labels))),
            )
            Sy = (U @ U.T).toarray()

            score = roc_auc_score(Sy.reshape(-1), S.reshape(-1))
            print(f"{model_name}: {score}")
            # assert score > 0.5, f"test failed for {model_name}. ROC AUC score must be >0.5: {score}"

    def test_link_prediction(self):
        dataset = gnn_tools.LinkPredictionDataset(
            testEdgeFraction=0.25, negative_edge_sampler="degreeBiased"
        )

        dataset.fit(self.A)
        train_net, test_edge_table = dataset.transform()

        # Test
        positive_edge_table = test_edge_table[test_edge_table["isPositiveEdge"] == 1]
        negative_edge_table = test_edge_table[test_edge_table["isPositiveEdge"] == 0]

        test_net = sparse.csr_matrix(
            (
                np.ones(positive_edge_table.shape[0]),
                (positive_edge_table["src"], positive_edge_table["trg"]),
            ),
            shape=self.A.shape,
        )
        test_net = test_net + test_net.T
        test_net.data = np.ones_like(test_net.data)

        neg_net = sparse.csr_matrix(
            (
                np.ones(negative_edge_table.shape[0]),
                (negative_edge_table["src"], negative_edge_table["trg"]),
            ),
            shape=self.A.shape,
        )
        neg_net = neg_net + neg_net.T
        neg_net.data = np.ones_like(neg_net.data)

        # positive edges and negative edges must be disjoint
        assert np.all((test_net.multiply(neg_net)).data == 0)

        # test_net and train_net must be disjoint
        assert np.all((test_net.multiply(train_net)).data == 0)
