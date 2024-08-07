# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-08-26 09:51:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-16 15:49:52
"""Module for embedding."""
# %%
import graph_tool.all as gt
import gensim
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from gnn_tools import samplers, utils
from scipy.sparse import csgraph
from scipy.optimize import minimize


# Base class
class NodeEmbeddings:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass


class Node2Vec(NodeEmbeddings):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        window_length=10,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.sampler = samplers.Node2VecWalkSampler(
            num_walks=num_walks,
            walk_length=walk_length,
            p=p,
            q=q,
        )

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": 1,
            "workers": 4,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.window_length

        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(
            sentences=self.sampler.walks, **self.w2vparams
        )

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv[i]
            self.out_vec[i, :] = self.model.syn1neg[self.model.wv.key_to_index[i]]


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams = {
            "sg": 1,
            "hs": 1,
            "min_count": 0,
            "workers": 16,
        }


class LaplacianEigenMap(NodeEmbeddings):
    def __init__(self, p=100, q=40, reconstruction_vector=False):
        self.in_vec = None
        self.L = None
        self.deg = None
        self.p = p
        self.q = q
        self.reconstruction_vector = reconstruction_vector

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        if self.reconstruction_vector:
            u, s = truncated_eigs(self.L, dim)
            order = np.argsort(s)[::-1]
            self.in_vec = u[:, order]
            self.out_vec = u[:, order]
        else:
            u, s = truncated_eigs(self.L, dim + 1)
            s, u = np.real(s), np.real(u)
            order = np.argsort(-s)[1:]
            s, u = s[order], u[:, order]
            Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
            self.in_vec = Dsqrt @ u @ sparse.diags(np.sqrt(np.abs(s)))
            self.out_vec = u


class AdjacencySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):
        u, s = truncated_eigs(self.A, dim)
        self.in_vec = u @ sparse.diags(s)


class ModularitySpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, reconstruction_vector=False, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.reconstruction_vector = reconstruction_vector
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        s, u = self.powerIteratedModularityEmbedding(self.A, k=dim)
        s, u = np.real(s), np.real(u)
        if self.reconstruction_vector:
            is_positive = s > 0
            u[:, ~is_positive] = 0
            s[~is_positive] = 0
            self.in_vec = u @ sparse.diags(np.sqrt(s))
        else:
            self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = u

    def shave_embedding(self, emb, K):
        # Column normalize emb
        # nemb = emb / np.maximum(1e-12, np.linalg.norm(emb, axis=0))
        nemb = emb / np.maximum(
            1e-12, np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
        )

        # Compute the modularity eigenvalues
        vals = np.diag(nemb.T @ self.A @ nemb) - np.sum(
            np.power(nemb.T @ self.deg, 2)
        ) / np.sum(self.deg)

        # Shave
        order = np.argsort(-vals)
        shaved_emb = emb[:, order[: (K - 1)]]
        return shaved_emb

    def powerIteratedModularityEmbedding(self, net, k, n_iter=100, eps=1e-12):
        deg = np.array(net.sum(axis=1)).flatten()

        double_m = np.sum(deg)

        n_nodes = len(deg)
        eigenvecs = np.zeros((n_nodes, k))
        eigenvals = np.zeros(k)

        for ell in range(k):
            b = np.random.randn(len(deg))
            b /= np.linalg.norm(b)
            b_prev = b.copy()
            for it in range(n_iter):
                b = (
                    net @ b / double_m
                    - deg * (deg.T @ b) / double_m**2
                    - np.einsum("ij,j->ij", eigenvecs, eigenvals) @ (eigenvecs.T @ b)
                )
                b /= np.linalg.norm(b)
                if (
                    np.mean(np.abs(b - b_prev) / np.maximum(1e-12, np.abs(b_prev)))
                    < eps
                ):
                    break
                b_prev = b.copy()
            eigenvecs[:, ell] = b.copy()
            eigenvals[ell] = b.T @ net @ b / double_m - (b.T @ deg) ** 2 / double_m**2
        return eigenvals, eigenvecs


class LinearizedNode2Vec(NodeEmbeddings):
    def __init__(self, verbose=False, window_length=10, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        # Calculate the normalized transition matrix
        Dinvsqrt = sparse.diags(1 / np.sqrt(np.maximum(1, self.deg)))
        Psym = Dinvsqrt @ self.A @ Dinvsqrt

        u, s = truncated_eigs(Psym, dim)
        order = np.argsort(-s)
        s, u = s[order], u[:, order]

        # u, s, v = rsvd.rSVD(Psym, dim=dim + 1, p=self.p, q=self.q)
        # sign = np.sign(np.diag(v @ u))

        s = np.abs(s)
        mask = s < np.max(s)
        u = u[:, mask]
        s = s[mask]

        if self.window_length > 1:
            s = (s * (1 - s**self.window_length)) / (self.window_length * (1 - s))

        self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = None


class NonBacktrackingSpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, auto_dim=False):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.auto_dim = auto_dim
        self.C = 10

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.A = A
        return self

    def update_embedding(self, dim):
        N = self.A.shape[0]
        Z = sparse.csr_matrix((N, N))
        I = sparse.identity(N, format="csr")
        D = sparse.diags(self.deg)
        B = sparse.bmat([[Z, D - I], [-I, self.A]], format="csr")

        if self.auto_dim is False:
            v, s = truncated_eigs(B, dim)
            s, v = np.real(s), np.real(v)
            order = np.argsort(-np.abs(s))
            s, v = s[order], v[:, order]
            v = v[N:, :]

            # Normalize the eigenvectors because we cut half the vec
            # and omit the imaginary part.
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

            # Weight the dimension by the eigenvalues
            v = v @ np.diag(np.sqrt(np.abs(s)))
        else:
            dim = int(self.C * np.sqrt(N))
            dim = np.minimum(dim, N - 1)

            v, s = truncated_eigs(B, dim + 1)

            c = int(self.A.sum() / N)
            s, v = s[np.abs(s) > c], v[:, np.abs(s) > c]

            order = np.argsort(s)
            s, v = s[order], v[:, order]
            s, v = s[1:], v[:, 1:]
            v = v[N:, :]
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

        self.in_vec = v


class DegreeEmbedding:
    def __init__(self, **params):
        return

    def fit(self, net):
        self.degree = np.array(net.sum(axis=0)).reshape(-1)

    def transform(self, dim):
        emb = np.zeros((len(self.degree), dim))
        emb[:, 0] = self.degree
        return emb


class FastRP:
    """
    Fast Random Projection embedding
    See https://arxiv.org/abs/1908.11512.
    """

    def __init__(self, window_size, beta=-1, s=1.0):
        self.window_size = window_size
        self.beta = beta
        self.s = s

    def fit(self, net):
        self.net = net

    def transform(self, dim):
        n_nodes = self.net.shape[0]
        # Generate random matrix for random projection
        if np.isclose(self.s, 1):
            X = np.random.randn(n_nodes, dim)
        else:
            X = sparse.random(
                n_nodes,
                dim,
                density=1 / self.s,
                data_rvs=lambda x: np.sqrt(self.s)
                * (2 * np.random.randint(2, size=x) - 1),
            ).toarray()

        emb = self._fastRP(self.net, dim, self.window_size, beta=self.beta, X=X.copy())

        return emb

    def _fastRP(self, net, dim, window_size, X, beta=-1):
        # Get stats
        n_nodes = net.shape[0]
        outdeg = np.array(net.sum(axis=1)).reshape(-1)
        indeg = np.array(net.sum(axis=0)).reshape(-1)

        # Construct the transition matrix
        P = sparse.diags(1 / np.maximum(1, outdeg)) @ net  # Transition matrix
        L = sparse.diags(np.power(np.maximum(indeg.astype(float), 1.0), beta))

        # First random projection
        X0 = (P @ L) @ X.copy()  # to include the self-loops

        # h is an array for normalization
        h = np.ones((n_nodes, 1))
        h0 = h.copy()

        # Iterative projection
        for _ in range(window_size):
            X = P @ X + X0
            h = P @ h + h0

        # Normalization
        X = sparse.diags(1.0 / np.maximum(np.array(h).reshape(-1), 1e-8)) @ X
        return X


class SpectralGraphTransformation(NodeEmbeddings):
    """
    Spectral graph transformation

    Fits and transforms input graph using spectral graph transformation method.

    Parameters
    ----------
    NodeEmbeddings : _type_
        _description_

    Attributes
    ----------
    kernel_matrix : str
        String indicating kernel matrix to use (options: "A", "normalized_A", "laplacian")
    kernel_func : callable
        Function defining the kernel function used in the transformation
    in_vec, out_vec : ndarray or None
        Input and output node embeddings

    Methods
    -------
    fit(net)
        Fits the model on the input graph
    update_embedding(dim)
        Transforms the fitted graph into low-dimensional space
    get_kernel_matrix(A)
        Computes the kernel matrix from an adjacency matrix
    train_test_edge_split(A, fraction)
        Splits edge set into training and testing sets

    References
    ----------
    - https://dl.acm.org/doi/abs/10.1145/1553374.1553447
    """

    def __init__(self, kernel_func="exp", kernel_matrix="A"):
        """
        Initialize SpectralGraphTransformation with given kernel function and
        kernel matrix options.

        Parameters
        ----------
        kernel_func : str or callable, optional (default="exp")
            The kernel function used for the spectral graph transformation.
            Can be a string representing one of two built-in kernel functions ("exp" or "neu"),
            or a user-defined function taking two input values as follows:
            `kernel_func(x, a) -> y`, where x is an array of eigenvalues, a is a scalar parameter,
            and y is the transformed array of eigenvalues.
        kernel_matrix : str, optional (default="A")
            The kernel matrix used in the transformation. Can be one of three options:
            "A" for adjacency matrix, "normalized_A" for normalized adjacency matrix,
            or "laplacian" for Laplacian matrix.
        """
        self.kernel_matrix = kernel_matrix

        if kernel_func == "exp":
            self.kernel_func = lambda x, a: np.exp(-a * x)
        elif kernel_func == "neu":
            self.kernel_func = lambda x, a: 1.0 / (1 - a * x)
        else:
            self.kernel_func = kernel_func
        self.in_vec = None
        self.out_vec = None
        self.reg = 1e-2

    def fit(self, net):
        """
        Fit the spectral graph transformation model on the input graph.

        Parameters
        ----------
        net : array-like or sparse matrix
            Input graph as an adjacency matrix or equivalent sparse matrix format.

        Returns
        -------
        self : object
            Returns self.
        """

        A = utils.to_adjacency_matrix(net)
        n_nodes = A.shape[0]
        train_edges, test_edges = self.train_test_edge_split(A, 1 / 3)
        train_net = utils.edge2network(train_edges[0], train_edges[1], n_nodes=n_nodes)
        test_net = utils.edge2network(test_edges[0], test_edges[1], n_nodes=n_nodes)

        self.A = A
        self.train_net = train_net
        self.test_net = test_net
        self.Gkernel = self.get_kernel_matrix(A)
        self.Gkernel_train = self.get_kernel_matrix(train_net)

        return self

    def update_embedding(self, dim):
        """
        Update the input network embedding and transform it into low-dimensional space.

        Parameters
        ----------
        dim : int
            Dimensions of the output embedding.

        Returns
        -------
        None
        """

        if self.kernel_matrix == "laplacian":
            # Find the largest eigenvalue
            # Flip the eigenvalues by max(lam) - lam. Since the laplacian has eigenvalues that are all positive, this will make the the largest eivalues the smallest and vise versa.
            # This allows us to use the fast singular value decomposition to identify the eigenvectors
            # associated with the smallest eigenvalues of the laplacian matrix efficiently.
            s_largest, _ =  sparse.linalg.eigs(self.Gkernel_train, k=1, which="LR")
            s_largest = np.real(s_largest)
            svd = TruncatedSVD(n_components=dim+1, n_iter=7, random_state=42)
            svd.fit(sparse.diags(s_largest * np.ones(self.Gkernel_train.shape[0])) - self.Gkernel_train)
            u = svd.components_.T
            s_train = svd.singular_values_
            u = u[:, 1:]
            s_train = s_train[1:]
        else:
            svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
            svd.fit(self.Gkernel_train)
            s_train = svd.singular_values_
            u = svd.components_.T
        s_test = np.diag(u.T @ self.Gkernel @ u)

        #s_train, u = sparse.linalg.eigs(self.Gkernel_train, k=dim, which=which)
        #s_test = np.diag(u.T @ self.Gkernel @ u)
        s_test, u, s_train = np.real(s_test), np.real(u), np.real(s_train)
        s_train = s_train / np.max(np.abs(s_train))
        s_test = s_test / np.max(np.abs(s_test))

        def cost(params):  # simply use globally defined x and y
            alpha = params[0]
            return (
                np.mean((self.kernel_func(s_train, alpha) - s_test) ** 2)
                + self.reg * alpha**2
            )  # quadratic cost function

        res = minimize(cost, [1e-1])
        alpha = res.x[0]
        if (np.isnan(alpha)) or (np.isinf(alpha)):
            spred = s_test
        else:
            spred = self.kernel_func(s_test, alpha)
        self.in_vec = u @ np.diag(np.sqrt(np.abs(spred)))
        self.out_vec = self.in_vec
        return self

    def get_kernel_matrix(self, A):
        deg = np.array(A.sum(axis=1)).reshape(-1)
        if self.kernel_matrix == "A":
            M = A
        elif self.kernel_matrix == "normalized_A":
            M = sparse.diags(1 / np.sqrt(deg)) @ A @ sparse.diags(1 / np.sqrt(deg))
        elif self.kernel_matrix == "laplacian":
            M = sparse.diags(deg) - A
        return sparse.csr_matrix(M)

    def train_test_edge_split(self, A, fraction):
        r, c, _ = sparse.find(A)
        edges = np.unique(utils.pairing(r, c))

        MST = csgraph.minimum_spanning_tree(A + A.T)
        r, c, _ = sparse.find(MST)
        mst_edges = np.unique(utils.pairing(r, c))
        remained_edge_set = np.array(
            list(set(list(edges)).difference(set(list(mst_edges))))
        )
        max_fraction = len(remained_edge_set) / len(edges)
        if fraction > max_fraction:
            fraction = max_fraction * 0.5
        n_edge_removal = int(fraction * len(remained_edge_set))
        test_edge_set = np.random.choice(
            remained_edge_set, n_edge_removal, replace=False
        )

        train_edge_set = np.array(
            list(set(list(edges)).difference(set(list(test_edge_set))))
        )

        test_edges_ = utils.depairing(test_edge_set)
        train_edges_ = utils.depairing(train_edge_set)
        return train_edges_, test_edges_


from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


class SBMEmbedding(NodeEmbeddings):
    def __init__(self, min_com_size=5, degreeCorrected=True):
        self.min_com_size = min_com_size
        self.degreeCorrected = degreeCorrected
        self.in_vec = None
        self.out_vec = None

    def fit(self, net):
        self.net = net
        return self

    def update_embedding(self, dim):
        r, c, v = sparse.find(self.net)
        g = gt.Graph(directed=False)
        g.add_edge_list(np.vstack([r, c]).T)

        n_nodes = self.net.shape[0]
        K = np.minimum(dim, int(n_nodes / self.min_com_size))

        if self.degreeCorrected:
            state = gt.minimize_blockmodel_dl(
                g,
                state_args={"B_min": K, "B_max": K},
                multilevel_mcmc_args={"B_max": K, "B_min": K},
            )
        else:
            state = gt.minimize_blockmodel_dl(
                g,
                state_args={"B_min": K, "B_max": K, "deg_corr": False},
                multilevel_mcmc_args={"B_max": K, "B_min": K},
            )
        b = state.get_blocks()
        ucids, cids = np.unique(np.array(b.a), return_inverse=True)
        n_nodes = len(cids)
        K = len(ucids)
        U = sparse.csr_matrix(
            (np.ones_like(cids), (np.arange(len(cids)), cids)), shape=(n_nodes, K)
        )
        outdeg = np.array(self.net.sum(axis=1)).reshape(-1)
        indeg = np.array(self.net.sum(axis=0)).reshape(-1)
        Din = np.array(U.T @ indeg).reshape(-1)
        Dout = np.array(U.T @ outdeg).reshape(-1)
        Lsbm = (
            np.diag(1 / np.maximum(1e-32, Dout))
            @ (U.T @ self.net @ U)
            @ np.diag(1 / np.maximum(1e-32, Din))
        )

        s, u = np.linalg.eig(Lsbm)
        u = np.einsum("ij,j->ij", u, np.sqrt(np.maximum(s, 0)))
        u = U @ u
        self.in_vec = np.einsum("ij,i->ij", u, outdeg)
        self.out_vec = np.einsum("ij,i->ij", u, indeg)


#    def update_embedding(self, dim):
#        r, c, v = sparse.find(self.net)
#        g = gt.Graph(directed=False)
#        g.add_edge_list(np.vstack([r, c]).T)
#
#        n_nodes = self.net.shape[0]
#        K = np.minimum(dim, int(n_nodes / self.min_com_size))
#        cids = self.find_blocks_by_sbm(self.net, K)
#        cids = np.unique(np.array(cids), return_inverse=True)[1]
#        n_nodes = len(cids)
#        U = sparse.csr_matrix(
#            (np.ones_like(cids), (np.arange(len(cids)), cids)), shape=(n_nodes, K)
#        )
#        outdeg = np.array(self.net.sum(axis=1)).reshape(-1)
#        indeg = np.array(self.net.sum(axis=0)).reshape(-1)
#        Din = np.array(U.T @ indeg).reshape(-1)
#        Dout = np.array(U.T @ outdeg).reshape(-1)
#        Lsbm = (
#            np.diag(1 / np.maximum(1e-32, Dout))
#            @ (U.T @ self.net @ U)
#            @ np.diag(1 / np.maximum(1e-32, Din))
#        )
#
#        s, u = np.linalg.eig(Lsbm)
#        u = np.einsum("ij,j->ij", u, np.sqrt(np.maximum(s, 0)))
#        u = U @ u
#        self.in_vec = np.einsum("ij,i->ij", u, outdeg)
#        self.out_vec = np.einsum("ij,i->ij", u, indeg)

#    def find_blocks_by_sbm(self, A, K, directed=False):
#        """Jiashun Jin. Fast community detection by SCORE.
#
#        :param A: scipy sparse matrix
#        :type A: sparse.csr_matrix
#        :param K: number of communities
#        :type K: int
#        :param directed: whether to cluster directed or undirected, defaults to False
#        :type directed: bool, optional
#        :return: [description]
#        :rtype: [type]
#        """
#
#        if K >= (A.shape[0] - 1):
#            cids = np.arange(A.shape[0])
#            return cids
#
#        svd = TruncatedSVD(n_components=K).fit(A)
#        u = svd.components_.T
#        v = svd.transform(A)
#        v = np.einsum("ij,j->ij", v, 1.0/np.linalg.norm(v, axis = 0))
#
#        #u, s, v = utils.rSVD(A, dim=K)
#        u = np.ascontiguousarray(u, dtype=np.float32)
#        if directed:
#            v = np.ascontiguousarray(v.T, dtype=np.float32)
#            u = np.hstack([u, v])
#        norm = np.linalg.norm(u, axis=1)
#        denom = 1 / np.maximum(norm, 1e-5)
#        denom[np.isclose(norm, 0)] = 0
#
#        u = np.einsum("ij,i->ij", u, denom)
#
#        if (u.shape[0] / K) < 10:
#            niter = 1
#        else:
#            niter = 10
#        km = KMeans(n_clusters = K)
#        km.fit(u)
#        cids = km.labels_
#
#        return np.array(cids).reshape(-1)
#    def find_blocks_by_sbm(self, A, K, directed=False):


def truncated_eigs(M, dim):
    svd = TruncatedSVD(n_components=dim)
    svd = svd.fit(M)
    u = svd.components_.T
    s = np.array(np.sum((M @ u) * u, axis=0)).reshape(-1)
    s, u = np.real(s), np.real(u)
    return u, s
