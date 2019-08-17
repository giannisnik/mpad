import math
import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

from mlp import MLP

class MessagePassing(Module):
    """
    Simple Message Passing layer
    """

    def __init__(self, in_features, out_features):
        super(MessagePassing, self).__init__()
        self.mlp1 = MLP(2, in_features, out_features, out_features)
        self.mlp2 = MLP(2, out_features, out_features, out_features)

        self.fc1_update = nn.Linear(out_features, out_features)
        self.fc2_update = nn.Linear(out_features, out_features)
        self.fc1_reset = nn.Linear(out_features, out_features)
        self.fc2_reset = nn.Linear(out_features, out_features)
        self.fc1 = nn.Linear(out_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x_in, adj):
        x = self.mlp1(x_in)
        out = torch.spmm(adj, x)
        out = self.mlp2(out)


        z = torch.sigmoid(self.fc1_update(out) + self.fc2_update(x))
        r = torch.sigmoid(self.fc1_reset(out) + self.fc2_reset(x))
        out = torch.tanh(self.fc1(out) + self.fc2(r*x))
        out = (1-z)*x + z*out
        return out


class Attention(Module):
    """
    Simple Attention layer
    """

    def __init__(self, in_features, nhid, master_node):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.nhid = nhid
        self.master_node = master_node
        self.fc1 = nn.Linear(in_features, nhid)
        self.fc2 = nn.Linear(nhid, 1, bias=False)
        self.fc3 = nn.Linear(nhid, nhid)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def forward(self, x_in):
        x = torch.tanh(self.fc1(x_in))
        x = torch.tanh(self.fc2(x))
        if self.master_node:
            t = self.softmax(x[:,:-1,:])
            t = t.unsqueeze(3)
            x = x_in[:,:-1,:].repeat(1, 1, 1)
            x = x.view(x.size()[0],x.size()[1], 1, self.nhid)
            t = t.repeat(1, 1, 1, x_in.size()[2])*x
            t = t.view(t.size()[0], t.size()[1], -1)
            t = t.sum(1)
            t = self.relu(self.fc3(t))
            out = torch.cat([t, x_in[:,-1,:].squeeze()], 1)
        else:
            t = self.softmax(x)
            t = t.unsqueeze(3)
            x = x_in.repeat(1, 1, 1)
            x = x.view(x.size()[0],x.size()[1], 1, self.nhid)
            t = t.repeat(1, 1, 1, x_in.size()[2])*x
            t = t.view(t.size()[0], t.size()[1], -1)
            t = t.sum(1)
            out = self.relu(self.fc3(t))
            
        return out
