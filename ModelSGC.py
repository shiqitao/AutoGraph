import torch
from sklearn.metrics import accuracy_score
from torch.nn import Linear
from torch.nn.functional import relu, dropout, log_softmax, nll_loss, leaky_relu
from torch_geometric.nn import SGConv

from Result import Result


class ModelSGC(torch.nn.Module):

    def __init__(self, num_layers, hidden, activation, data):
        super(ModelSGC, self).__init__()
        self.linear_1 = Linear(data.num_features, hidden)
        self.conv = SGConv(hidden, hidden, K=num_layers)
        self.linear_2 = Linear(hidden, data.num_class)
        if activation == "relu":
            self.activation = relu
        elif activation == "leaky_relu":
            self.activation = leaky_relu

    def reset_parameters(self):
        self.linear_1.reset_parameters()
        self.conv.reset_parameters()
        self.linear_2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.linear_1(x)
        x = self.activation(x)
        x = dropout(x, p=0.5, training=self.training)
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = dropout(x, p=0.5, training=self.training)
        x = self.linear_2(x)
        return log_softmax(x, dim=-1)


def main_model_sgc(data, num_layers, hidden, activation, if_all=False):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelSGC(
        num_layers=num_layers,
        hidden=hidden,
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
