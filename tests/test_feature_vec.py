"""Quick smoke test for the feature_vec parameter in GNN models."""
import numpy as np
import networkx as nx
import gnn_tools


def make_karate():
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    labels = np.unique(
        [d[1]["club"] for d in G.nodes(data=True)], return_inverse=True
    )[1]
    return A, n, labels


def test_feature_vec_link_prediction():
    A, n, _ = make_karate()
    feat = np.random.randn(n, 16).astype(np.float32)
    emb = gnn_tools.models.GAT(A, dim=8, feature_vec=feat, epochs=3)
    assert emb.shape == (n, 8), f"Expected ({n}, 8), got {emb.shape}"


def test_feature_vec_community_detection():
    A, n, labels = make_karate()
    feat = np.random.randn(n, 16).astype(np.float32)
    emb = gnn_tools.models.GAT(A, dim=8, feature_vec=feat, memberships=labels, epochs=3)
    assert emb.shape == (n, 8), f"Expected ({n}, 8), got {emb.shape}"


def test_no_feature_vec_still_works():
    A, n, _ = make_karate()
    emb = gnn_tools.models.GAT(A, dim=8, epochs=3)
    assert emb.shape == (n, 8), f"Expected ({n}, 8), got {emb.shape}"


if __name__ == "__main__":
    test_feature_vec_link_prediction()
    print("test_feature_vec_link_prediction passed")
    test_feature_vec_community_detection()
    print("test_feature_vec_community_detection passed")
    test_no_feature_vec_still_works()
    print("test_no_feature_vec_still_works passed")
