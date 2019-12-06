import time
import torch
import torch.nn.functional as F
from gcn.layers import gcn_layer


class GCN(torch.nn.Module):
    def __init__(self, indim, hiddim, classdim, dropout, lr, weight_decay):
        super(GCN, self).__init__()

        self.gc1 = gcn_layer(indim, hiddim)
        self.gc2 = gcn_layer(hiddim, classdim)
        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def accuracy(self, output, labels):
        prediction = torch.argmax(output, dim=1)
        correct = prediction.eq(labels).float()
        correct = torch.sum(correct)/len(labels)
        return correct

    def train2(self, features, labels, adj, idx_train, idx_val, idx_test, epoch, verbose):
        t = time.time()
        self.train()
        self.optimizer.zero_grad()
        output = self.forward(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = self.accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        if verbose:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.eval()
            output = self.forward(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = self.accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def test(self, features, labels, adj, idx_train, idx_val, idx_test, epoch, verbose):
        self.eval()
        output = self.forward(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = self.accuracy(output[idx_test], labels[idx_test])
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))


