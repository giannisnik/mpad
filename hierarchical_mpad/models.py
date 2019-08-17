import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MessagePassing, Attention

class MPAD(nn.Module):
    def __init__(self, n_feat, n_message_passing, n_hid, n_penultimate, n_class, dropout, embeddings, use_master_node, graph_of_sentences):
        super(MPAD, self).__init__()
        self.graph_of_sentences = graph_of_sentences
        self. n_message_passing =  n_message_passing
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = False

        self.mps1 = torch.nn.ModuleList()
        self.atts1 = torch.nn.ModuleList()
        for i in range(n_message_passing):
            if i == 0:
                self.mps1.append(MessagePassing(n_feat, n_hid))
            else:
                self.mps1.append(MessagePassing(n_hid, n_hid))
            self.atts1.append(Attention(n_hid, n_hid, use_master_node))

        if use_master_node:
            self.bn = nn.BatchNorm1d(2*n_message_passing*n_hid, n_hid)
            self.fc1 = nn.Linear(2*n_message_passing*n_hid, n_hid)
        else:
            self.bn = nn.BatchNorm1d(n_message_passing*n_hid, n_hid)
            self.fc1 = nn.Linear(n_message_passing*n_hid, n_hid)

        if graph_of_sentences == 'sentence_att':
            self.att = Attention(n_hid, n_hid, False)
            self.fc2 = nn.Linear(n_hid, n_penultimate)
        else:
            self.fc2 = nn.Linear(n_message_passing*n_hid, n_penultimate)
            self.mps2 = torch.nn.ModuleList()
            self.atts2 = torch.nn.ModuleList()
            for i in range(n_message_passing):
                self.mps2.append(MessagePassing(n_hid, n_hid))
                self.atts2.append(Attention(n_hid, n_hid, False))
         
        self.fc3 = nn.Linear(n_penultimate, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj, adj_s, shapes):
        x = self.embedding(x)
        x = self.dropout(x)
        lst = list()
        for i in range(self.n_message_passing):
            x = self.mps1[i](x, adj)
            t = x.view(-1, shapes[0], x.size()[1])
            t = self.atts1[i](t)
            lst.append(t)
        x = torch.cat(lst, 1)
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        if self.graph_of_sentences == 'sentence_att':
            x = x.view(-1, shapes[1], x.size()[1])
            x = self.att(x)
        else:
            lst = list()
            for i in range(self.n_message_passing):
                x = self.mps2[i](x, adj_s)
                t = x.view(-1, shapes[1], x.size()[1])
                t = self.atts2[i](t)
                lst.append(t)
            x = torch.cat(lst, 1)  
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
