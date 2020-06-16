import argparse
import os
import time

import torch
from filelock import FileLock

from ModelAPPNP import main_model_appnp
from ModelAPPNP2 import main_model_appnp as main_model_appnp_2
from ModelAPPNP3 import main_model_appnp as main_model_appnp_3
from ModelAPPNP4 import main_model_appnp as main_model_appnp_4
from ModelGAT import main_model_gat
from ModelGAT2 import main_model_gat as main_model_gat_2
from ModelGAT3 import main_model_gat as main_model_gat_3
from ModelGAT4 import main_model_gat as main_model_gat_4
from ModelGCN import main_model_gcn
from ModelGCN2 import main_model_gcn as main_model_gcn_2
from ModelGCN3 import main_model_gcn as main_model_gcn_3
from ModelGCN4 import main_model_gcn as main_model_gcn_4
from ModelGCNOld import main_model_gcn_old
from Result import Result
from tools import save_data, load_data, file_path

if __name__ == "__main__":
    start_time = time.time()
    time_budget = float("inf")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    parser.add_argument("--file_param", type=str)
    parser.add_argument("--file_ready", type=str)
    parser.add_argument("--file_result", type=str)
    parser.add_argument("--file_lock", type=str)
    parser.add_argument("--if_kill", type=int)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
    with FileLock(args.file_lock):
        save_data(args.file_ready, os.getpid())
    aoe_data = None
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
            if param.time_budget is not None:
                time_budget = param.time_budget  # 重置时间限制
            try:
                if param.model == "ModelGCNOld":
                    result = main_model_gcn_old(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        # 一定要按这个顺序 save -> remove
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGAT":
                    result = main_model_gat(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        # 一定要按这个顺序 save -> remove
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGCN":
                    result = main_model_gcn(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelAPPNP":
                    result = main_model_appnp(
                        data=aoe_data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGAT2":
                    result = main_model_gat_2(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        # 一定要按这个顺序 save -> remove
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGCN2":
                    result = main_model_gcn_2(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelAPPNP2":
                    result = main_model_appnp_2(
                        data=aoe_data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGAT3":
                    result = main_model_gat_3(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        # 一定要按这个顺序 save -> remove
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGCN3":
                    result = main_model_gcn_3(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelAPPNP3":
                    result = main_model_appnp_3(
                        data=aoe_data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGAT4":
                    result = main_model_gat_4(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        # 一定要按这个顺序 save -> remove
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelGCN4":
                    result = main_model_gcn_4(
                        data=aoe_data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                elif param.model == "ModelAPPNP4":
                    result = main_model_appnp_4(
                        data=aoe_data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=True
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)
                else:
                    raise ValueError("Model name error: {0}".format(param[0]))
            except RuntimeError:
                if args.if_kill == 1:
                    break
                else:
                    result = Result(
                        result=None,
                        loss_train=None,
                        loss_valid=None,
                        acc_train=None,
                        acc_valid=None,
                        epoch=None,
                    )
                    with FileLock(args.file_lock):
                        save_data(args.file_result, result)
                        os.remove(args.file_param)

        if time.time() - start_time >= time_budget:
            break
