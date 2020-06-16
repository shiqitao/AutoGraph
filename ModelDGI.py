import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.nn import PReLU
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from Result import Result


class Encoder(torch.nn.Module):

    def __init__(self, hidden, data):
        super(Encoder, self).__init__()
        self.conv = GCNConv(data.num_features, hidden, cached=True)
        self.prelu = PReLU(hidden)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return self.prelu(x)


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


def main_model_dgi(data, hidden, if_all=False):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepGraphInfomax(
        hidden_channels=hidden,
        encoder=Encoder(hidden, data),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption)

    data.split_train_valid()
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_acc_valid = 0
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)

        lr = LogisticRegression().fit(pos_z[data.mask_train].detach().cpu().numpy().reshape(-1, hidden),
                                      data.y[data.mask_train].cpu().numpy())

        valid_pred = lr.predict(pos_z[data.mask_valid].detach().cpu().numpy().reshape(-1, hidden))
        acc_valid = accuracy_score(data.y[data.mask_valid].cpu().numpy(),
                                   valid_pred)

        if acc_valid > best_acc_valid:
            best_acc_valid = acc_valid
            result = pos_z

        loss = model.loss(pos_z.to(device), neg_z.to(device), summary.to(device))
        loss.backward()
        optimizer.step()

    lr = LogisticRegression().fit(result[data.mask_train].detach().cpu().numpy().reshape(-1, hidden),
                                  data.y[data.mask_train].cpu().numpy())

    train_pred = lr.predict(result[data.mask_train].detach().cpu().numpy().reshape(-1, hidden))
    all_pred = lr.predict(result.detach().cpu().numpy().reshape(-1, hidden))

    if if_all:
        return Result(
            result=torch.tensor(np.eye(data.num_class)[all_pred]).float().cpu(),
            loss_train=-1,
            loss_valid=-1,
            acc_train=accuracy_score(data.y[data.mask_train].cpu().numpy(),
                                     train_pred),
            acc_valid=best_acc_valid,
            epoch=10,
        )
    else:
        return Result(
            result=all_pred[data.mask_test],
            loss_train=-1,
            loss_valid=-1,
            acc_train=accuracy_score(data.y[data.mask_train].cpu().numpy(),
                                     train_pred),
            acc_valid=best_acc_valid,
            epoch=10,
        )
