import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading %s dataset...'%dataset)

    # set feature,labels
    idx_features_labels = np.genfromtxt("%s%s.content"%(path, dataset),
                                        dtype=np.dtype(str))
    features = np.array([list(map(eval,idx_features_labels[:, 1:-1][i])) for i in range(idx_features_labels.shape[0])])
    labels = encode_onehot(idx_features_labels[:, -1])
    features = normalize_feature(features)

    # set adj
    edges_unordered = np.genfromtxt("%s%s.cites"%(path, dataset),
                                    dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = np.array(sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32).toarray())
    adj = adj + np.where(adj.transpose()>adj, 1, 0)
    adj = normalize_adj(adj)

    # set training data
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # turn to tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(np.sum(mx, axis=1).astype(np.float32))
    r_temp = np.power(rowsum, -1)
    r_mat_temp = np.diag(r_temp)
    mx = r_mat_temp.dot(mx)
    return mx

def normalize_adj(mx):
    """Lapacian normalization"""
    mx = mx + np.identity(mx.shape[0])
    rowsum = np.array(np.sum(mx, axis=1).astype(np.float32))
    r_temp = np.power(rowsum, -0.5)
    r_mat_temp = np.diag(r_temp)
    mx = mx.dot(r_mat_temp).transpose().dot(r_mat_temp)
    return mx




