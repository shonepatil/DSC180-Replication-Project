import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, 'src/model')
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)