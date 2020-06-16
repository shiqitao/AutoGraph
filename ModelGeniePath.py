import torch
from sklearn.metrics import accuracy_score
from torch.nn.functional import nll_loss, log_softmax
from torch_geometric.nn import GATConv

from Result import Result


class Breadth(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):
    def __init__(self, dim, lstm_hidden, in_dim):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, dim)
        self.depth_func = Depth(dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


class GeniePath(torch.nn.Module):
    def __init__(self, num_layers, dim, lstm_hidden, data):
        super(GeniePath, self).__init__()
        self.dim = dim
        self.lstm_hidden = lstm_hidden
        self.lin1 = torch.nn.Linear(data.num_features, dim)
        self.gplayers = torch.nn.ModuleList(
            [GeniePathLayer(self.dim, self.lstm_hidden, dim) for _ in range(num_layers)])
        self.lin2 = torch.nn.Linear(dim, data.num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
        embed = x
        x = self.lin2(embed)
        return embed, log_softmax(x, dim=-1)


def main_model_geniepath(data, num_layers, dim, lstm_hidden, if_all=False):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GeniePath(
        num_layers=num_layers,
        dim=dim,
        lstm_hidden=lstm_hidden,
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
        _, predict = model(data)
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
        embed, result = model(data)
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
