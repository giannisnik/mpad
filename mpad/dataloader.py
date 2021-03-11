import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import lil_matrix
from torch.utils.data import (
    Dataset,
    DataLoader,
)


def doc2graph(doc, word2idx, window_size, directed, to_normalize, use_master_node):

    # First construct the nodes that live inside the document

    nodes = list(set(doc))  # todo: how to handle OOV words
    # Give each word/node a unique id for bookkeeping of edges
    node2idx = {node: ix for ix, node in enumerate(nodes)}
    node_features = [word2idx[node] for node in nodes]

    if use_master_node:
        master_node_idx = len(node2idx)
        nodes.append(master_node_idx)
        node2idx["master_node"] = master_node_idx

    node_features = np.array(node_features, dtype=np.int32)

    # Then construct the edges by taking a sliding window over the document
    edges = {}
    for i, w in enumerate(doc):
        src_ix = node2idx[w]
        for j in range(
            i + 1, i + window_size
        ):  # TODO: check if window size is applied correctly
            if j < len(doc):
                tgt_ix = node2idx[doc[j]]
                if (src_ix, tgt_ix) in edges:
                    edges[(src_ix, tgt_ix)] += 1.0 / (j - i)
                    if not directed:
                        edges[(tgt_ix, src_ix)] += 1.0 / (j - i)
                else:
                    edges[(src_ix, tgt_ix)] = 1.0 / (j - i)
                    if not directed:
                        edges[(tgt_ix, src_ix)] = 1.0 / (j - i)
        if use_master_node:
            edges[(src_ix, master_node_idx)] = 1.0
            edges[(master_node_idx, src_ix)] = 1.0

    # Construct a sparse matrix from the edges
    edge_s = list()
    edge_t = list()
    val = list()
    for edge in edges:
        edge_s.append(edge[0])
        edge_t.append(edge[1])
        val.append(edges[edge])
    # Construct the sparse adjacency matrix
    A = sp.csr_matrix((val, (edge_s, edge_t)), shape=(len(node2idx), len(node2idx)))
    if len(edges) == 0:
        A = sp.csr_matrix(([0], ([0], [0])), shape=(1, 1))
        node_features = np.zeros(1, dtype=np.int32)

    # commented for now: waiting for explanation of original authors why the transpose is there
    # if directed:
    #     A = A.transpose()
    if to_normalize and A.size > 1:
        A = normalize(A)

    return A, node_features


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def collate_fn_no_master_node(batch):
    """
    Default collate fn does not support batching differently sized sparse matrices
    """
    batch_A, batch_nodes, batch_y = zip(*batch)

    n_graphs = len(batch_nodes)
    max_n_nodes = max([nodes.shape[0] for nodes in batch_nodes])

    n_nodes = n_graphs * max_n_nodes

    adj_batch = lil_matrix((n_nodes, n_nodes))
    batch_features = np.zeros(n_nodes)
    for i, (A, features) in enumerate(zip(batch_A, batch_nodes)):
        start_ix = i * max_n_nodes
        adj_batch[
            start_ix : start_ix + A.shape[0], start_ix : start_ix + A.shape[0]
        ] = A
        batch_features[start_ix : start_ix + features.shape[0]] = features

    adj_batch = adj_batch.tocsr()
    batch_A = adj_batch
    batch_A = sparse_mx_to_torch_sparse_tensor(batch_A)

    # concatenate all features and labels to one long vector
    batch_nodes = torch.cat(batch_nodes, dim=0)
    batch_y = torch.cat(batch_y)

    return batch_A, batch_nodes, batch_y, torch.LongTensor([n_graphs])


def collate_fn_w_master_node(batch):
    """
    Default collate fn does not support batching differently sized sparse matrices
    """
    batch_A, batch_nodes, batch_y = zip(*batch)

    n_graphs = len(batch_nodes)
    max_n_nodes = max([nodes.shape[0] for nodes in batch_nodes])

    n_nodes = n_graphs * max_n_nodes

    adj_batch = lil_matrix((n_nodes, n_nodes))
    batch_features = np.zeros(n_nodes)
    for i, (A, features) in enumerate(zip(batch_A, batch_nodes)):
        start_ix = i * max_n_nodes
        # Word-word edges
        adj_batch[
            start_ix : start_ix + A.shape[0] - 1, start_ix : start_ix + A.shape[0] - 1
        ] = A[:-1, :-1]
        # Edges to the master node
        adj_batch[
            start_ix : start_ix + A.shape[0] - 1, start_ix : start_ix + max_n_nodes - 1
        ] = A[:-1, -1]
        # Edges from the master node
        adj_batch[start_ix + max_n_nodes - 1, start_ix : start_ix + A.shape[0] - 1] = A[
            -1, :-1
        ]

        # Set features in padded manner
        batch_features[start_ix : start_ix + features.shape[0]] = features

    adj_batch = adj_batch.tocsr()
    batch_A = adj_batch
    batch_A = sparse_mx_to_torch_sparse_tensor(batch_A)

    # concatenate all features and labels to 1 long vector
    batch_features = torch.LongTensor(batch_features)
    batch_y = torch.cat(batch_y)

    return batch_A, batch_features, batch_y, torch.LongTensor([n_graphs])


class DocumentGraphDataset(Dataset):
    def __init__(
        self,
        docs,
        labels,
        word2idx,
        window_size,
        use_master_node,
        normalize_edges,
        use_directed_edges,
    ):

        """
        Dataset inheriting from PyTorch's Dataset class.
        Constructs a graph of words of each document based
        """
        self.use_master_node = use_master_node
        self.labels = torch.LongTensor(labels)
        self.graphs = [
            doc2graph(
                doc,
                word2idx=word2idx,
                window_size=window_size,
                directed=use_directed_edges,
                to_normalize=normalize_edges,
                use_master_node=use_master_node,
            )
            for doc in docs
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        graphs = self.graphs[idx]
        A, nodes = graphs
        y = self.labels[idx]

        return (A, nodes, y)

    def to_dataloader(self, batch_size, shuffle, drop_last):

        collate_fn = (
            collate_fn_w_master_node
            if self.use_master_node
            else collate_fn_no_master_node
        )
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
