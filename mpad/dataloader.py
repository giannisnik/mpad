import scipy.sparse as sp
from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from torch.utils.data import (
	Dataset,
	DataLoader,
	TensorDataset,
	RandomSampler,
	DistributedSampler,
	SequentialSampler,
)
import tqdm
import concurrent.futures
import pickle
import torch
import glob
from collections import defaultdict



def doc2graph(doc, word2idx, window_size, directed, to_normalize, use_master_node):

	# First construct the nodes that live inside the document

	nodes = list(set(doc))#todo: how to handle OOV words
	# Give each word/node a unique id for bookkeeping of edges
	node2idx = {node:ix for ix,node in enumerate(nodes)}
	node_features = [word2idx[node] for node in nodes]

	if use_master_node:
		master_node_idx = len(node2idx)
		nodes.append(master_node_idx)
		node2idx['master_node'] = master_node_idx

	node_features = np.array(node_features, dtype=np.int32)

	# idx = dict()
	# l_terms = list()
	# for i in range(len(doc)):
	# 	if doc[i] not in idx:
	# 		l_terms.append(doc[i])
	# 		idx[doc[i]] = len(idx)


	# Then construct the edges by taking a sliding window over the document
	edges = {}
	for i, w in enumerate(doc):
		src_ix = node2idx[w]
		for j in range(i + 1, i + window_size):  # TODO: check if window size is applied correctly
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
	A = sp.csr_matrix((val, (edge_s, edge_t)), shape=(len(node2idx), len(node2idx)))
	if len(edges) == 0:
		A = sp.csr_matrix(([0], ([0], [0])), shape=(1, 1))
		node_features = np.zeros(1, dtype=np.int32)

	if directed:
		A = A.transpose()
	if to_normalize and A.size > 1:
		A = normalize(A)

	# Convert to torch tensors
	# A = sparse_mx_to_torch_sparse_tensor(A)
	# node_features = torch.LongTensor(node_features)

	return A, node_features


def create_gows(docs, vocab, window_size, directed, to_normalize, use_master_node):
	adj = list()
	features = list()
	idx2term = list()

	for doc in docs:
		edges = dict()

		idx = dict()
		l_terms = list()
		for i in range(len(doc)):
			if doc[i] not in idx:
				l_terms.append(doc[i])
				idx[doc[i]] = len(idx)
		idx2term.append(l_terms)
		if use_master_node:
			idx["master_node"] = len(idx)
		X = np.zeros(len(idx), dtype=np.int32)
		for w in idx:
			if w != "master_node":
				X[idx[w]] = vocab[w]
			else:
				X[idx[w]] = len(vocab)
		for i in range(len(doc)):
			for j in range(i + 1, i + window_size): #TODO: check if window size is applied correctly
				if j < len(doc):
					if (doc[i], doc[j]) in edges:
						edges[(doc[i], doc[j])] += 1.0 / (j - i)
						if not directed:
							edges[(doc[j], doc[i])] += 1.0 / (j - i)
					else:
						edges[(doc[i], doc[j])] = 1.0 / (j - i)
						if not directed:
							edges[(doc[j], doc[i])] = 1.0 / (j - i)
			if use_master_node:
				edges[(doc[i], "master_node")] = 1.0
				edges[("master_node", doc[i])] = 1.0

		edge_s = list()
		edge_t = list()
		val = list()
		for edge in edges:
			edge_s.append(idx[edge[0]])
			edge_t.append(idx[edge[1]])
			val.append(edges[edge])
		A = sp.csr_matrix((val, (edge_s, edge_t)), shape=(len(idx), len(idx)))
		if len(edges) == 0:
			A = sp.csr_matrix(([0], ([0], [0])), shape=(1, 1))
			X = np.zeros(1, dtype=np.int32)

		if directed:
			A = A.transpose()
		if to_normalize and A.size > 1:
			A = normalize(A)
		adj.append(A)
		features.append(X)


	return adj, features, idx2term


def normalize(mx):
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(np.vstack((sparse_mx.row,
										  sparse_mx.col))).long()
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches(adj, features, y, batch_size, use_master_node, shuffle=False):
	n = len(y)
	if shuffle:
		index = np.random.permutation(n)
	else:
		index = np.array(range(n), dtype=np.int32)

	n_batches = ceil(n / batch_size)

	adj_l = list()
	features_l = list()
	batch_n_graphs_l = list()
	y_l = list()

	for i in range(0, n, batch_size):
		if n > i + batch_size:
			up = i + batch_size
		else:
			up = n

		n_graphs = 0
		max_n_nodes = 0
		for j in range(i, up):
			n_graphs += 1
			if adj[index[j]].shape[0] > max_n_nodes:
				max_n_nodes = adj[index[j]].shape[0]

		n_nodes = n_graphs * max_n_nodes

		adj_batch = lil_matrix((n_nodes, n_nodes))
		features_batch = np.zeros(n_nodes)
		y_batch = np.zeros(n_graphs)

		for j in range(i, up):
			idx = (j - i) * max_n_nodes
			if max_n_nodes >= adj[index[j]].shape[0]:
				if use_master_node:
					adj_batch[idx:idx + adj[index[j]].shape[0] - 1, idx:idx + adj[index[j]].shape[0] - 1] = adj[index[
						j]][:-1, :-1]
					adj_batch[idx:idx + adj[index[j]].shape[0] - 1, idx + max_n_nodes - 1] = adj[index[j]][:-1, -1]
					adj_batch[idx + max_n_nodes - 1, idx:idx + adj[index[j]].shape[0] - 1] = adj[index[j]][-1, :-1]
				else:
					adj_batch[idx:idx + adj[index[j]].shape[0], idx:idx + adj[index[j]].shape[0]] = adj[index[j]]

				features_batch[idx:idx + adj[index[j]].shape[0] - 1] = features[index[j]][:-1]
			else:
				if use_master_node:
					adj_batch[idx:idx + max_n_nodes - 1, idx:idx + max_n_nodes - 1] = adj[index[j]][:max_n_nodes - 1,
																					  :max_n_nodes - 1]
					adj_batch[idx:idx + max_n_nodes - 1, idx + max_n_nodes - 1] = adj[index[j]][:max_n_nodes - 1, -1]
					adj_batch[idx + max_n_nodes - 1, idx:idx + max_n_nodes - 1] = adj[index[j]][-1, :max_n_nodes - 1]
				else:
					adj_batch[idx:idx + max_n_nodes, idx:idx + max_n_nodes] = adj[index[j]][:max_n_nodes, :max_n_nodes]

				features_batch[idx:idx + max_n_nodes - 1] = features[index[j]][:max_n_nodes - 1]

			y_batch[j - i] = y[index[j]]

		adj_batch = adj_batch.tocsr()

		adj_l.append(sparse_mx_to_torch_sparse_tensor(adj_batch))
		features_l.append(torch.LongTensor(features_batch))
		batch_n_graphs_l.append(torch.LongTensor(np.array([n_graphs], dtype=np.int64)))
		y_l.append(torch.LongTensor(y_batch))

	return adj_l, features_l, batch_n_graphs_l, y_l


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
		start_ix = i*max_n_nodes
		adj_batch[start_ix:start_ix+A.shape[0], start_ix:start_ix+A.shape[0]] = A
		batch_features[start_ix:start_ix+features.shape[0]] = features

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
		adj_batch[start_ix:start_ix + A.shape[0]-1, start_ix:start_ix + A.shape[0]-1] = A[:-1,:-1]
		# Edges to the master node
		adj_batch[start_ix:start_ix + A.shape[0] - 1, start_ix:start_ix + max_n_nodes - 1] = A[:-1, -1]
		# Edges from the master node
		adj_batch[start_ix+max_n_nodes-1, start_ix:start_ix + A.shape[0]-1] = A[-1, :-1]

		# Set features in padded manner
		batch_features[start_ix:start_ix + features.shape[0]] = features

	adj_batch = adj_batch.tocsr()
	batch_A = adj_batch
	batch_A = sparse_mx_to_torch_sparse_tensor(batch_A)

	# concatenate all features and labels to 1 long vector
	batch_features = torch.LongTensor(batch_features)
	batch_y = torch.cat(batch_y)

	return batch_A, batch_features, batch_y, torch.LongTensor([n_graphs])


class DocumentGraphDataset(Dataset):
	def __init__(self, docs, labels, word2idx, window_size, use_master_node, normalize_edges, use_directed_edges):

		"""
		Dataset inheriting from PyTorch's Dataset class.
		Constructs a graph of words of each document based
		"""
		self.use_master_node = use_master_node
		self.labels = torch.LongTensor(labels)
		self.graphs = [doc2graph(
		 doc,
		 word2idx=word2idx,
		 window_size=window_size,
		 directed=use_directed_edges,
		 to_normalize=normalize_edges,
		 use_master_node=use_master_node

		) for doc in docs]

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

		collate_fn = collate_fn_w_master_node if self.use_master_node else collate_fn_no_master_node
		return DataLoader(
			self,
			batch_size=batch_size,
			shuffle=shuffle,
			drop_last=drop_last,
			collate_fn=collate_fn

		)