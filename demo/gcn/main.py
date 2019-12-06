import time
import numpy as np
import argparse
from gcn.utils import *
from gcn.model import *

# Overall settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_false', default=True,
                    help='CUDA training.')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training settings
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


if  __name__ == "__main__":

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    print(adj.shape, features.shape, labels.shape)

    # Model and optimizer
    model = GCN(indim = features.shape[1],
                hiddim = args.hidden,
                classdim = np.max(labels.numpy(), axis=0) + 1,
                dropout = args.dropout,
                lr = args.lr,
                weight_decay = args.weight_decay)

    # Set cuda
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Init time
    t_total = time.time()

    # Train
    for epoch in range(args.epochs):
        model.train2(features, labels, adj, idx_train, idx_val, idx_test, epoch, verbose=True)
    print("Epochs done~~")
    print("Time: {:.4f}s".format(time.time() - t_total))

    # Test
    model.test(features, labels, adj, idx_train, idx_val, idx_test, epoch, verbose=True)
