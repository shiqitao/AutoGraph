import argparse
import os
import time

import torch
from filelock import FileLock
# from torch.nn import Linear
from torch.nn.functional import log_softmax, nll_loss
from torch_geometric.nn import JumpingKnowledge

from tools import file_path
from tools import save_data, load_data


class ModelEnsemble(torch.nn.Module):

    def __init__(self, num_features, num_class):
        super(ModelEnsemble, self).__init__()
        self.JK = JumpingKnowledge(mode='lstm', channels=num_class, num_layers=1)
        # self.linear = Linear(num_features, num_class)

    def reset_parameters(self):
        self.JK.reset_parameters()

    def forward(self, x):
        x = self.JK(x)
        return log_softmax(x, dim=-1)


def main_ensemble(data, num_features, num_class):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = load_data(file_path("AOE_ENSEMBLE.data"))
    model = ModelEnsemble(
        num_features=num_features,
        num_class=num_class
    )

    data.split_train_valid()
    model = model.to(device)
    mask_train, mask_valid, mask_test, y = data.mask_train, data.mask_valid, data.mask_test, data.y
    mask_train = mask_train.to(device)
    mask_valid = mask_valid.to(device)
    mask_test = mask_test.to(device)
    y = y.to(device)
    for i in range(len(x)):
        x[i] = x[i].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    epoch = 1
    loss_train = float("inf")
    loss_valid = float("inf")
    best_loss_train = float("inf")
    best_loss_valid = float("inf")
    best_result = None
    best_epoch = 0
    while best_epoch + 10 >= epoch:
        model.train()
        optimizer.zero_grad()
        predict = model(x)
        loss_train = nll_loss(predict[mask_train], y[mask_train])
        loss_valid = nll_loss(predict[mask_valid], y[mask_valid])
        loss_train.backward()
        optimizer.step()
        if loss_valid < best_loss_valid:
            best_loss_train = loss_train
            best_loss_valid = loss_valid
            best_result = predict
            best_epoch = epoch
        epoch += 1

    return best_result[mask_test].max(1)[1].cpu().numpy().flatten()


if __name__ == "__main__":
    start_time = time.time()
    time_budget = float("inf")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    parser.add_argument("--file_param", type=str)
    parser.add_argument("--file_ready", type=str)
    parser.add_argument("--file_result", type=str)
    parser.add_argument("--file_lock", type=str)
    args = parser.parse_args()
    aoe_data = None
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    with FileLock(args.file_lock):
        save_data(args.file_ready, os.getpid())
    while True:
        if aoe_data is None and os.path.exists(file_path("AOE.data")):
            with FileLock(file_path("AOE.ready")):
                aoe_data = load_data(file_path("AOE.data"))
        if os.path.exists(args.file_param):
            if aoe_data is None and os.path.exists(file_path("AOE.data")):
                with FileLock(file_path("AOE.ready")):
                    aoe_data = load_data(file_path("AOE.data"))
            start_time = time.time()  # 重置开始时间
            with FileLock(args.file_lock):
                param = load_data(args.file_param)
            time_budget = param.time_budget  # 重置时间限制
            assert param.model == "ModelEnsemble"
            result = main_ensemble(
                data=aoe_data,
                num_features=param.num_features,
                num_class=param.num_class
            )
            with FileLock(args.file_lock):
                os.remove(args.file_param)
                save_data(args.file_result, result)
            break

        if time.time() - start_time >= time_budget:
            break
