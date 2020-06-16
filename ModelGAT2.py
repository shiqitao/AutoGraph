import torch
from sklearn.metrics import accuracy_score
from torch.nn import Linear
from torch.nn.functional import relu, dropout, log_softmax, nll_loss, leaky_relu
from torch_geometric.nn import GATConv
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import coalesce

from Result import Result


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index, edge_attr=None, p=0.5, force_undirected=False,
                num_nodes=None, training=True):
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0),), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


class ModelGAT(torch.nn.Module):

    def __init__(self, num_layers, hidden_list, activation, data):
        super(ModelGAT, self).__init__()
        assert len(hidden_list) == num_layers + 1
        self.linear_1 = Linear(data.num_features, hidden_list[0])
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GATConv(hidden_list[i], hidden_list[i + 1]))
        self.linear_2 = Linear(hidden_list[-1], data.num_class)
        if activation == "relu":
            self.activation = relu
        elif activation == "leaky_relu":
            self.activation = leaky_relu

    def reset_parameters(self):
        self.linear_1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.linear_2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = dropout_adj(edge_index, p=0.8, training=self.training)
        x = self.linear_1(x)
        x = self.activation(x)
        x = dropout(x, p=0.5, training=self.training)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if i != len(self.convs) - 1:
                x = self.activation(x)
        x = dropout(x, p=0.5, training=self.training)
        x = self.linear_2(x)
        return log_softmax(x, dim=-1)


def main_model_gat(data, num_layers, hidden_list, activation, if_all=False):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelGAT(
        num_layers=num_layers,
        hidden_list=hidden_list,
        activation=activation,
        data=data,
    )

    data.split_train_valid()
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    epoch = 1
    loss_train = float("inf")
    loss_valid = float("inf")
    best_loss_train = float("inf")
    best_loss_valid = float("inf")
    best_epoch = 0
    while best_epoch + 10 >= epoch:
        model.train()
        optimizer.zero_grad()
        predict = model(data)
        loss_train = nll_loss(predict[data.mask_train], data.y[data.mask_train])
        loss_valid = nll_loss(predict[data.mask_valid], data.y[data.mask_valid])
        loss_train.backward()
        optimizer.step()
        if loss_valid < best_loss_valid:
            best_loss_train = loss_train
            best_loss_valid = loss_valid
            best_epoch = epoch
        epoch += 1

    model.eval()
    with torch.no_grad():
        result = model(data)
    if if_all:
        return Result(
            result=result.cpu(),
            loss_train=loss_train.cpu(),
            loss_valid=loss_valid.cpu(),
            acc_train=accuracy_score(data.y[data.mask_train].cpu().numpy().flatten(),
                                     result[data.mask_train].max(1)[1].cpu().numpy().flatten()),
            acc_valid=accuracy_score(data.y[data.mask_valid].cpu().numpy().flatten(),
                                     result[data.mask_valid].max(1)[1].cpu().numpy().flatten()),
            epoch=epoch - 1,
        )
    else:
        return Result(
            result=result[data.mask_test].max(1)[1].cpu().numpy().flatten(),
            loss_train=loss_train.cpu(),
            loss_valid=loss_valid.cpu(),
            acc_train=accuracy_score(data.y[data.mask_train].cpu().numpy().flatten(),
                                     result[data.mask_train].max(1)[1].cpu().numpy().flatten()),
            acc_valid=accuracy_score(data.y[data.mask_valid].cpu().numpy().flatten(),
                                     result[data.mask_valid].max(1)[1].cpu().numpy().flatten()),
            epoch=epoch - 1,
        )
