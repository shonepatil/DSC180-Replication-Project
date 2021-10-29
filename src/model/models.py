import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, 'src/model')
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # Problems with training loss not decreasing
        # Model lacks capacity to learn(more units)
        # not enough parameters to learn
        # Combined GCN with dense linear layers
        # one approach: start with more layers and work backwards to avoid overfitting
        # add batch normlaizing layers
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nhid, nhid)
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc5 = nn.Linear(nhid, nhid)
        self.fc6 = nn.Linear(nhid, nhid)
        self.fc7 = nn.Linear(nhid, nhid)
        self.fc8 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):# Add linear layers before and after
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        x = self.fc4(F.relu(x))
        x = self.gc1(F.relu(x), adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.fc5(F.relu(x))
        x = self.fc6(F.relu(x))
        x = self.fc7(F.relu(x))
        x = self.fc8(F.relu(x))
        return F.sigmoid(x)

class FCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(FCN, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)