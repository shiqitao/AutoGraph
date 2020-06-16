import random

import torch


class DataSet:

    def __init__(
            self,
            x,
            y,
            edge_index,
            edge_weight,
            train_valid_indices,
            test_indices,
            mask_train_valid,
            mask_test,
            original_features
    ):
        self.__x = x
        self.__y = y
        self.__edge_index = edge_index
        self.__edge_weight = edge_weight
        self.__train_valid_indices = train_valid_indices
        self.__test_indices = test_indices
        self.__mask_train_valid = mask_train_valid
        self.__mask_test = mask_test
        self.__original_features = original_features
        self.__mask_train = None
        self.__mask_valid = None
        self.__node_embedding = None

    def remove_features(self):
        self.__x = None

    def add_node_embedding(self, node_embedding):
        self.__node_embedding = node_embedding
        if self.original_features:
            self.__x = torch.cat((self.node_embedding, self.x), 1)
        else:
            self.__x = self.node_embedding

    def to(self, device):
        self.__x = self.__x.to(device)
        self.__y = self.__y.to(device)
        self.__edge_index = self.__edge_index.to(device)
        self.__edge_weight = self.__edge_weight.to(device)
        self.__mask_train_valid = self.__mask_train_valid.to(device)
        self.__mask_test = self.__mask_test.to(device)
        if self.mask_train is not None:
            self.__mask_train = self.__mask_train.to(device)
        if self.mask_valid is not None:
            self.__mask_valid = self.__mask_valid.to(device)
        if self.node_embedding is not None:
            self.__node_embedding = self.__node_embedding.to(device)
        return self

    def split_train_valid(self):
        train_valid_indices = self.train_valid_indices
        train_indices = []
        valid_indices = []
        for i in range(len(train_valid_indices)):
            if random.random() < 0.9:
                train_indices.append(train_valid_indices[i])
            else:
                valid_indices.append(train_valid_indices[i])
        self.__mask_train = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.__mask_train[train_indices] = 1
        self.__mask_valid = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.__mask_valid[valid_indices] = 1

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def edge_index(self):
        return self.__edge_index

    @property
    def edge_weight(self):
        return self.__edge_weight

    @property
    def num_nodes(self):
        return self.x.size(0)

    @property
    def num_features(self):
        return self.x.size(1)

    @property
    def num_class(self):
        return int(max(self.y)) + 1

    @property
    def train_valid_indices(self):
        return self.__train_valid_indices

    @property
    def test_indices(self):
        return self.__test_indices

    @property
    def mask_train_valid(self):
        return self.__mask_train_valid

    @property
    def mask_test(self):
        return self.__mask_test

    @property
    def original_features(self):
        return self.__original_features

    @property
    def mask_train(self):
        return self.__mask_train

    @property
    def mask_valid(self):
        return self.__mask_valid

    @property
    def node_embedding(self):
        return self.__node_embedding
