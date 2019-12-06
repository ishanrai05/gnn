import numpy as np
import torch

class gcn_layer(torch.nn.Module):
    """
    layer http://tkipf.github.io/graph-convolutional-networks/
    """

    def __init__(self, indim, outdim):
        super(gcn_layer, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.indim, self.outdim))
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.outdim))
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / np.sqrt(self.outdim)
        torch.nn.init.uniform_(self.weight.data, stdv, stdv)
        torch.nn.init.uniform_(self.bias.data, stdv, stdv)
        # self.weight.data.uniform_(-stdv, stdv)
        # self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

    def __repr__(self):
        return ['layer',self._get_name(),"from %s to %s"%(self.indim, self.outdim)]