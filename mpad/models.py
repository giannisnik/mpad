import torch
import torch.nn as nn
from layers import MessagePassing, Attention


class MPAD(nn.Module):
    def __init__(
        self,
        n_feat,
        n_message_passing,
        n_hid,
        n_penultimate,
        n_class,
        dropout,
        embeddings,
        use_master_node,
    ):
        super(MPAD, self).__init__()
        self.n_message_passing = n_message_passing
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = True

        self.mps = torch.nn.ModuleList()
        self.atts = torch.nn.ModuleList()
        for i in range(n_message_passing):
            if i == 0:
                self.mps.append(MessagePassing(n_feat, n_hid))
            else:
                self.mps.append(MessagePassing(n_hid, n_hid))
            self.atts.append(Attention(n_hid, n_hid, use_master_node))

        if use_master_node:
            self.bn = nn.BatchNorm1d(2 * n_message_passing * n_hid)
            self.fc1 = nn.Linear(2 * n_message_passing * n_hid, n_penultimate)
        else:
            self.bn = nn.BatchNorm1d(n_message_passing * n_hid)
            self.fc1 = nn.Linear(n_message_passing * n_hid, n_penultimate)

        self.fc2 = nn.Linear(n_penultimate, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj, n_graphs):
        x = self.embedding(x)
        x = self.dropout(x)
        lst = list()
        for i in range(self.n_message_passing):
            # Get incoming messages for each node as function of neighbours
            x = self.mps[i](x, adj)
            # Resize to [B, max_n_nodes, emb_size]
            t = x.view(n_graphs[0], -1, x.size()[1])
            # Apply self-attention over each nodes to obtain graph representation of layer L
            t = self.atts[i](t)
            lst.append(t)

        # Concatenate representations of all layers
        x = torch.cat(lst, 1)
        # Batch norm
        x = self.bn(x)
        # Feed-forward network as classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
