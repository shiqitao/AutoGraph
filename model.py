import os
import time

import torch
from common import get_logger
from filelock import FileLock
from torch.nn.functional import nll_loss

from AutoGraphEnsemble import ModelEnsemble
from Param import Param
from tools import file_path, save_data, load_data, generate_data, is_subprocess_alive

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

for root, dirs, files in os.walk(os.path.dirname(os.path.realpath(__file__))):
    for file in files:
        if file.startswith("AOE"):
            os.remove(file_path(file))
os.system("kill -9 `ps -ef | grep AutoGraphModel.py | awk '{print $2}' `")

max_num_parallel = 4
pid_model = []
pid_ensemble = None
for k in range(max_num_parallel):
    os.system("python {0} {1} {2} {3} {4} {5} {6} &".format(
        file_path("AutoGraphModel.py"),
        "--index {0}".format(k),
        "--file_param {0}".format(file_path("AOE_MODEL_{0}.param".format(k))),
        "--file_ready {0}".format(file_path("AOE_MODEL_{0}.ready".format(k))),
        "--file_result {0}".format(file_path("AOE_MODEL_{0}.result".format(k))),
        "--file_lock {0}".format(file_path("AOE_MODEL_{0}.lock".format(k))),
        "--if_kill {0}".format(1 if k != 0 else 0),
    ))
if torch.cuda.is_available():
    torch.zeros(1).cuda()
for k in range(max_num_parallel):
    while not os.path.exists(file_path("AOE_MODEL_{0}.ready".format(k))):
        pass
    with FileLock(file_path("AOE_MODEL_{0}.lock".format(k))):
        pid_model.append(load_data(file_path("AOE_MODEL_{0}.ready".format(k))))
        LOGGER.info("pid_model_{0} {1}".format(k, pid_model[-1]))


class Model:

    @staticmethod
    def train_predict(data, time_budget, n_class, schema):
        start_time = time.time()
        LOGGER.info("Start!")
        LOGGER.info("time_budget: {0}".format(time_budget))

        data = generate_data(data, LOGGER)
        LOGGER.info("Num of features: {0}".format(data.num_features))
        LOGGER.info("Num of classes: {0}".format(data.num_class))
        params = [
            Param("ModelGCN", [1, [16, 16], "leaky_relu"]),
            Param("ModelGCN", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT4", [1, [32, 32], "leaky_relu"]),
            Param("ModelGCN", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT4", [1, [64, 64], "leaky_relu"]),
            Param("ModelGCN", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT4", [1, [128, 128], "leaky_relu"]),
            Param("ModelGCN", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT4", [1, [256, 256], "leaky_relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [1, [16, 16], "relu"]),
            Param("ModelGCN", [1, [32, 32], "relu"]),
            Param("ModelGAT", [1, [32, 32], "relu"]),
            Param("ModelGAT4", [1, [32, 32], "relu"]),
            Param("ModelGCN", [1, [64, 64], "relu"]),
            Param("ModelGAT", [1, [64, 64], "relu"]),
            Param("ModelGAT4", [1, [64, 64], "relu"]),
            Param("ModelGCN", [1, [128, 128], "relu"]),
            Param("ModelGAT", [1, [128, 128], "relu"]),
            Param("ModelGAT4", [1, [128, 128], "relu"]),
            Param("ModelGCN", [1, [256, 256], "relu"]),
            Param("ModelGAT", [1, [256, 256], "relu"]),
            Param("ModelGAT4", [1, [256, 256], "relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "relu"]),
            Param("ModelGCN", [1, [16, 16], "leaky_relu"]),
            Param("ModelGCN", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT4", [1, [32, 32], "leaky_relu"]),
            Param("ModelGCN", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT4", [1, [64, 64], "leaky_relu"]),
            Param("ModelGCN", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT4", [1, [128, 128], "leaky_relu"]),
            Param("ModelGCN", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT4", [1, [256, 256], "leaky_relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [1, [16, 16], "relu"]),
            Param("ModelGCN", [1, [32, 32], "relu"]),
            Param("ModelGAT", [1, [32, 32], "relu"]),
            Param("ModelGAT4", [1, [32, 32], "relu"]),
            Param("ModelGCN", [1, [64, 64], "relu"]),
            Param("ModelGAT", [1, [64, 64], "relu"]),
            Param("ModelGAT4", [1, [64, 64], "relu"]),
            Param("ModelGCN", [1, [128, 128], "relu"]),
            Param("ModelGAT", [1, [128, 128], "relu"]),
            Param("ModelGAT4", [1, [128, 128], "relu"]),
            Param("ModelGCN", [1, [256, 256], "relu"]),
            Param("ModelGAT", [1, [256, 256], "relu"]),
            Param("ModelGAT4", [1, [256, 256], "relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "relu"]),
            Param("ModelGCN", [1, [16, 16], "leaky_relu"]),
            Param("ModelGCN", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT4", [1, [32, 32], "leaky_relu"]),
            Param("ModelGCN", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT4", [1, [64, 64], "leaky_relu"]),
            Param("ModelGCN", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT4", [1, [128, 128], "leaky_relu"]),
            Param("ModelGCN", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT4", [1, [256, 256], "leaky_relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [1, [16, 16], "relu"]),
            Param("ModelGCN", [1, [32, 32], "relu"]),
            Param("ModelGAT", [1, [32, 32], "relu"]),
            Param("ModelGAT4", [1, [32, 32], "relu"]),
            Param("ModelGCN", [1, [64, 64], "relu"]),
            Param("ModelGAT", [1, [64, 64], "relu"]),
            Param("ModelGAT4", [1, [64, 64], "relu"]),
            Param("ModelGCN", [1, [128, 128], "relu"]),
            Param("ModelGAT", [1, [128, 128], "relu"]),
            Param("ModelGAT4", [1, [128, 128], "relu"]),
            Param("ModelGCN", [1, [256, 256], "relu"]),
            Param("ModelGAT", [1, [256, 256], "relu"]),
            Param("ModelGAT4", [1, [256, 256], "relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "relu"]),
            Param("ModelGCN", [1, [16, 16], "leaky_relu"]),
            Param("ModelGCN", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT", [1, [32, 32], "leaky_relu"]),
            Param("ModelGAT4", [1, [32, 32], "leaky_relu"]),
            Param("ModelGCN", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT", [1, [64, 64], "leaky_relu"]),
            Param("ModelGAT4", [1, [64, 64], "leaky_relu"]),
            Param("ModelGCN", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT", [1, [128, 128], "leaky_relu"]),
            Param("ModelGAT4", [1, [128, 128], "leaky_relu"]),
            Param("ModelGCN", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT", [1, [256, 256], "leaky_relu"]),
            Param("ModelGAT4", [1, [256, 256], "leaky_relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "leaky_relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "leaky_relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "leaky_relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "leaky_relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "leaky_relu"]),
            Param("ModelGCN", [1, [16, 16], "relu"]),
            Param("ModelGCN", [1, [32, 32], "relu"]),
            Param("ModelGAT", [1, [32, 32], "relu"]),
            Param("ModelGAT4", [1, [32, 32], "relu"]),
            Param("ModelGCN", [1, [64, 64], "relu"]),
            Param("ModelGAT", [1, [64, 64], "relu"]),
            Param("ModelGAT4", [1, [64, 64], "relu"]),
            Param("ModelGCN", [1, [128, 128], "relu"]),
            Param("ModelGAT", [1, [128, 128], "relu"]),
            Param("ModelGAT4", [1, [128, 128], "relu"]),
            Param("ModelGCN", [1, [256, 256], "relu"]),
            Param("ModelGAT", [1, [256, 256], "relu"]),
            Param("ModelGAT4", [1, [256, 256], "relu"]),
            Param("ModelGCN", [2, [16, 16, 16], "relu"]),
            Param("ModelGCN", [2, [32, 32, 32], "relu"]),
            Param("ModelGCN", [2, [64, 64, 64], "relu"]),
            Param("ModelGCN", [2, [128, 128, 128], "relu"]),
            Param("ModelGCN", [2, [256, 256, 256], "relu"]),
            Param("ModelGCN", [3, [16, 16, 16, 16], "relu"]),
            Param("ModelGCN", [3, [32, 32, 32, 32], "relu"]),
            Param("ModelGCN", [3, [64, 64, 64, 64], "relu"]),
            Param("ModelGCN", [3, [128, 128, 128, 128], "relu"]),
            Param("ModelGCN", [3, [256, 256, 256, 256], "relu"]),
        ]

        logger_killed_model_process = [True for _ in range(max_num_parallel)]
        params_running = [None for _ in range(max_num_parallel)]
        while True:
            for i in range(max_num_parallel):
                if time.time() - start_time >= time_budget - 5:
                    break
                if not is_subprocess_alive(pid_model[i]):
                    if logger_killed_model_process[i]:
                        LOGGER.info("Model process {0} has been killed".format(i))
                        if params_running[i]:
                            params_running[i].running = False
                            params_running[i].retry = params_running[i].retry - 1
                        logger_killed_model_process[i] = False
                if os.path.exists(file_path("AOE_MODEL_{0}.result".format(i))):
                    with FileLock(file_path("AOE_MODEL_{0}.lock".format(i))):
                        temp_result = load_data(file_path("AOE_MODEL_{0}.result".format(i)))
                        if temp_result.result is None:
                            params_running[i].running = False
                            params_running[i].retry = params_running[i].retry - 1
                            os.remove(file_path(file_path("AOE_MODEL_{0}.result".format(i))))
                            LOGGER.info("Result of Model {0} is None".format(params_running[i].index))
                        else:
                            params[params_running[i].index].result = temp_result
                            os.remove(file_path(file_path("AOE_MODEL_{0}.result".format(i))))
                            LOGGER.info("Get result of Model {0}, {1}, {2}, {3}, {4}".format(
                                params_running[i].index,
                                "loss_train = {0:.6f}".format(params_running[i].result.loss_train),
                                "loss_valid = {0:.6f}".format(params_running[i].result.loss_valid),
                                "acc_train = {0:.6f}".format(params_running[i].result.acc_train),
                                "acc_valid = {0:.6f}".format(params_running[i].result.acc_valid)
                            ))
                if not os.path.exists(file_path("AOE_MODEL_{0}.param".format(i))) and not os.path.exists(
                        file_path("AOE_MODEL_{0}.result".format(i))):
                    with FileLock(file_path("AOE_MODEL_{0}.lock".format(i))):
                        for params_index in range(len(params)):
                            if not params[params_index].running and params[params_index].retry > 0:
                                params[params_index].index = params_index
                                params[params_index].running = True
                                params[params_index].time_budget = time_budget - (time.time() - start_time)
                                params_running[i] = params[params_index]
                                save_data(file_path("AOE_MODEL_{0}.param".format(i)), params[params_index])
                                LOGGER.info("Start Model {0}".format(params_index))
                                break
            if time.time() - start_time >= time_budget - 5:
                break
            if_continue = False
            for i in range(len(params)):
                if params[i].result is None:
                    if_continue = True
                    break
            if not if_continue:
                break

        os.system("kill -9 `ps -ef | grep AutoGraphModel.py | awk '{print $2}' `")
        LOGGER.info("Start merge the result")
        params_result = []
        for i in range(len(params)):
            if params[i].result is not None:
                params_result.append(params[i])
        LOGGER.info("Num of result: {0}".format(len(params_result)))
        for i in range(len(params_result)):
            for j in range(i + 1, len(params_result)):
                if params_result[i].result.acc_valid > params_result[j].result.acc_valid:
                    params_result[i], params_result[j] = params_result[j], params_result[i]
        params_result = params_result[-4:]
        # 下面这段话？
        # params_result.reverse()
        # for i in range(1, len(params_result)):
        #     if params_result[i].result.acc_valid + 0.01 < params_result[0].result.acc_valid:
        #         params_result = params_result[0:i]
        #         break
        # params_result.reverse()
        # 上面这段话？
        for param in params_result:
            LOGGER.info("Final Model {0} {1}".format(param.index, param.model))
        result = [item.result.result for item in params_result]

        # ensemble
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelEnsemble(num_features=data.num_features, num_class=data.num_class)
        data.split_train_valid()
        model = model.to(device)
        mask_train, mask_valid, mask_test, y = data.mask_train, data.mask_valid, data.mask_test, data.y
        mask_train = mask_train.to(device)
        mask_valid = mask_valid.to(device)
        mask_test = mask_test.to(device)
        y = y.to(device)
        for i in range(len(result)):
            result[i] = result[i].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        epoch = 1
        best_loss_train = float("inf")
        best_loss_valid = float("inf")
        best_result = None
        best_epoch = 0
        while best_epoch + 10 >= epoch:
            model.train()
            optimizer.zero_grad()
            predict = model(result)
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
        LOGGER.info("Finish merge the result")
        return best_result[mask_test].max(1)[1].cpu().numpy().flatten()
