import numpy as np
import scipy.sparse as sp
import re
from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
import torch
from gensim.models.keyedvectors import KeyedVectors


def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split('\t')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  

  
def load_embeddings(fname, vocab):
    word_vecs = np.zeros((len(vocab)+1, 300))
    unknown_words = set()
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in vocab:
        if word in model:
            word_vecs[vocab[word],:] = model[word]
        else:
            unknown_words.add(word)
            word_vecs[vocab[word],:] = np.random.uniform(-0.25, 0.25, 300)
    print("Existing vectors:", len(vocab)-len(unknown_words))
    return word_vecs


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0

    for doc in docs:
        preprocessed_docs.append(clean_str(doc))
    
    return preprocessed_docs
    
    
def get_vocab(docs):
    vocab = dict()
    
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)+1

    print("Vocabulary size: ", len(vocab))
        
    return vocab


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
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    if (doc[i], doc[j]) in edges:
                        edges[(doc[i], doc[j])] += 1.0/(j-i)
                        if not directed:
                            edges[(doc[j], doc[i])] += 1.0/(j-i)
                    else:
                        edges[(doc[i], doc[j])] = 1.0/(j-i)
                        if not directed:
                            edges[(doc[j], doc[i])] = 1.0/(j-i)
            if use_master_node:
                edges[(doc[i],"master_node")] = 1.0
                edges[("master_node",doc[i])] = 1.0

        edge_s = list()
        edge_t = list()
        val = list()
        for edge in edges:
            edge_s.append(idx[edge[0]])
            edge_t.append(idx[edge[1]])
            val.append(edges[edge])
        A = sp.csr_matrix((val,(edge_s, edge_t)), shape=(len(idx), len(idx)))
        if len(edges) == 0:
            A = sp.csr_matrix(([0],([0], [0])), shape=(1, 1))
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

    n_batches = ceil(n/batch_size)

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

        n_nodes = n_graphs*max_n_nodes

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)

        for j in range(i, up):
            idx = (j-i)*max_n_nodes
            if max_n_nodes >= adj[index[j]].shape[0]:
                if use_master_node:
                    adj_batch[idx:idx+adj[index[j]].shape[0]-1, idx:idx+adj[index[j]].shape[0]-1] = adj[index[j]][:-1,:-1]
                    adj_batch[idx:idx+adj[index[j]].shape[0]-1, idx+max_n_nodes-1] = adj[index[j]][:-1,-1]
                    adj_batch[idx+max_n_nodes-1, idx:idx+adj[index[j]].shape[0]-1] = adj[index[j]][-1,:-1]
                else:
                    adj_batch[idx:idx+adj[index[j]].shape[0], idx:idx+adj[index[j]].shape[0]] = adj[index[j]]
                    
                features_batch[idx:idx+adj[index[j]].shape[0]-1] = features[index[j]][:-1]
            else:
                if use_master_node:
                    adj_batch[idx:idx+max_n_nodes-1, idx:idx+max_n_nodes-1] = adj[index[j]][:max_n_nodes-1,:max_n_nodes-1]
                    adj_batch[idx:idx+max_n_nodes-1, idx+max_n_nodes-1] = adj[index[j]][:max_n_nodes-1,-1]
                    adj_batch[idx+max_n_nodes-1, idx:idx+max_n_nodes-1] = adj[index[j]][-1,:max_n_nodes-1]
                else:
                    adj_batch[idx:idx+max_n_nodes, idx:idx+max_n_nodes] = adj[index[j]][:max_n_nodes,:max_n_nodes]
                
                features_batch[idx:idx+max_n_nodes-1] = features[index[j]][:max_n_nodes-1]

            y_batch[j-i] = y[index[j]]

        adj_batch = adj_batch.tocsr()
        
        adj_l.append(sparse_mx_to_torch_sparse_tensor(adj_batch))
        features_l.append(torch.LongTensor(features_batch))
        batch_n_graphs_l.append(torch.LongTensor(np.array([n_graphs], dtype=np.int64)))
        y_l.append(torch.LongTensor(y_batch))

    return adj_l, features_l, batch_n_graphs_l, y_l


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count