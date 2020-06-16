import os
import pickle

import numpy as np
import torch
from filelock import FileLock
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from DataSet import DataSet


def svd_factorization(x, edge_index):
    rows = edge_index[:, 0]
    cols = edge_index[:, 1]
    values = np.ones_like(rows)
    sp = coo_matrix((values, (rows, cols)), shape=(np.size(x, 0), np.size(x, 0)), dtype='f')
    # u, s, vt = svds((sp + sp.dot(sp)) / 2, k=64, which='LM', return_singular_vectors=True)
    u, s, vt = svds(sp, k=64, which='LM', return_singular_vectors=True)
    res = np.dot(u, np.diag(s))
    return res


def generate_data(data, LOGGER):
    LOGGER.info("Start generate data")

    # 特征处理
    x = data['fea_table']
    x = x.ix[:, (x != x.ix[0]).any()]
    df = data['edge_file']
    edge_index = df[['src_idx', 'dst_idx']].to_numpy()
    edge_weight = df['edge_weight'].to_numpy()

    if x.shape[1] == 1:
        LOGGER.info("Without original features")
        x = np.zeros(shape=(len(x), len(x)))
        for i in range(len(edge_index)):
            x[edge_index[i][0]][edge_index[i][1]] = edge_weight[i]
        # x = x.to_numpy()
        # x = x.reshape(x.shape[0])
        # x = np.array(pd.get_dummies(x))
        original_features = False
    else:
        LOGGER.info("With original features")
        x = x.drop('node_index', axis=1).to_numpy()
        original_features = True

    num_nodes = len(x)
    num_edges = len(edge_index)
    LOGGER.info("Num of nodes: {0}".format(num_nodes))
    LOGGER.info("Num of edges: {0}".format(num_edges))

    # svd 分解
    res = svd_factorization(x, edge_index)
    x = np.concatenate((x, res), 1)
    x = torch.tensor(x, dtype=torch.float)

    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    LOGGER.info("Finish transform x, edge_index, edge_weight")

    y = torch.zeros(num_nodes, dtype=torch.long)
    index = data['train_label'][['node_index']].to_numpy()
    train_y = data['train_label'][['label']].to_numpy()
    y[index] = torch.tensor(train_y, dtype=torch.long)
    LOGGER.info("Finish get y")

    train_valid_indices = data['train_indices']
    test_indices = data['test_indices']
    LOGGER.info("Finish get train_valid_indices, test_indices")

    mask_train_valid = torch.zeros(num_nodes, dtype=torch.bool)
    mask_train_valid[train_valid_indices] = 1
    LOGGER.info("Finish get mask_train_valid")

    mask_test = torch.zeros(num_nodes, dtype=torch.bool)
    mask_test[test_indices] = 1
    LOGGER.info("Finish get mask_test")

    LOGGER.info("Finish generate data")
    data = DataSet(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_valid_indices=train_valid_indices,
        test_indices=test_indices,
        mask_train_valid=mask_train_valid,
        mask_test=mask_test,
        original_features=original_features
    )
    LOGGER.info("Finish: data -> DataSet")
    # data.add_node_embedding(main_node2vec(data=data))
    # LOGGER.info("Finish: Node embedding")

    with FileLock(file_path("AOE.ready")):
        save_data(file_path("AOE.data"), data)
    LOGGER.info("Finish save new data")

    return data


def save_data(file_name, data):
    f = open(file_name, 'wb+')
    pickle.dump(data, f)
    f.close()


def load_data(file_name):
    f = open(file_name, 'rb+')
    data = pickle.load(f)
    f.close()
    return data


def file_path(file_name):
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(here, file_name)


def is_subprocess_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True
