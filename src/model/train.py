import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

import sys
sys.path.insert(0, 'src/model')
from models import GCN, FCN
from utils import accuracy, accuracy_binary

def train_test(A, X, y, idx_train, idx_val, idx_test, 
    no_cuda, seed, epochs, learning_rate, weight_decay, hidden_units, dropout, type):

    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = not no_cuda and torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type == "GCN":
        # create a model from GCN class
        # load it to the specified device, either gpu or cpu
        model = GCN(nfeat=X.shape[1],
                    nhid=hidden_units,
                    nclass=y.max().item(),
                    dropout=dropout).to(device)
    elif type == "FCN":
        # create a model from FCN class
        # load it to the specified device, either gpu or cpu
        model = FCN(nfeat=X.shape[1],
                    nhid=hidden_units,
                    nclass=y.max().item(),
                    dropout=dropout).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if cuda:
        model.cuda()
        X = X.cuda()
        A = A.cuda()
        y = y.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Function to train
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(X, A)
        target = y[idx_train].float()
        print(output[idx_train])
        print(target)
        loss_train = F.binary_cross_entropy(output[idx_train].cuda(), target.unsqueeze(1).cuda())
        acc_train = accuracy_binary(output[idx_train], target)
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(X, A)

        target = y[idx_val].float()
        loss_val = F.binary_cross_entropy(output[idx_val].cuda(), target.unsqueeze(1).cuda())
        acc_val = accuracy_binary(output[idx_val], target)
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))

    def test():
        model.eval()
        output = model(X, A)
        target = y[idx_test].float()
        loss_test = F.binary_cross_entropy(output[idx_test].cuda(), target.unsqueeze(1).cuda())
        acc_test = accuracy_binary(output[idx_test], target)
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))

    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)
        break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()