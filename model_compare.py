import pandas as pd
from sklearn.metrics import accuracy_score

from ModelAPPNP import main_model_appnp
from ModelAPPNP2 import main_model_appnp as main_model_appnp_2
from ModelAPPNP3 import main_model_appnp as main_model_appnp_3
from ModelAPPNP4 import main_model_appnp as main_model_appnp_4
from ModelDGI import main_model_dgi
from ModelGAT import main_model_gat
from ModelGAT2 import main_model_gat as main_model_gat_2
from ModelGAT3 import main_model_gat as main_model_gat_3
from ModelGAT4 import main_model_gat as main_model_gat_4
from ModelGCN import main_model_gcn
from ModelGCN2 import main_model_gcn as main_model_gcn_2
from ModelGCN3 import main_model_gcn as main_model_gcn_3
from ModelGCN4 import main_model_gcn as main_model_gcn_4
from Param import Param
from tools import file_path, load_data


class Model:

    @staticmethod
    def train_predict():
        data_set = 'c'
        data = load_data(file_path("{0}_AOE.data".format(data_set)))
        params = [
            # Param("ModelAPPNP", [10, 0.15, 16, "relu"]),
            # Param("ModelAPPNP", [10, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP", [10, 0.15, 32, "relu"]),
            # Param("ModelAPPNP", [10, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP", [10, 0.15, 64, "relu"]),
            # Param("ModelAPPNP", [10, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP", [10, 0.15, 128, "relu"]),
            # Param("ModelAPPNP", [10, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP", [20, 0.15, 16, "relu"]),
            # Param("ModelAPPNP", [20, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP", [20, 0.15, 32, "relu"]),
            # Param("ModelAPPNP", [20, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP", [20, 0.15, 64, "relu"]),
            # Param("ModelAPPNP", [20, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP", [20, 0.15, 128, "relu"]),
            # Param("ModelAPPNP", [20, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP", [30, 0.15, 16, "relu"]),
            # Param("ModelAPPNP", [30, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP", [30, 0.15, 32, "relu"]),
            # Param("ModelAPPNP", [30, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP", [30, 0.15, 64, "relu"]),
            # Param("ModelAPPNP", [30, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP", [30, 0.15, 128, "relu"]),
            # Param("ModelAPPNP", [30, 0.15, 128, "leaky_relu"]),
            #
            # Param("ModelAPPNP2", [10, 0.15, 16, "relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 32, "relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 64, "relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 128, "relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 16, "relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 32, "relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 64, "relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 128, "relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 16, "relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 32, "relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 64, "relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 128, "relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 128, "leaky_relu"]),
            #
            # Param("ModelAPPNP3", [10, 0.15, 16, "relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 32, "relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 64, "relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 128, "relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 16, "relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 32, "relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 64, "relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 128, "relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 16, "relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 32, "relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 64, "relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 128, "relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 128, "leaky_relu"]),
            #
            # Param("ModelAPPNP4", [10, 0.15, 16, "relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 32, "relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 64, "relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 128, "relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 16, "relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 32, "relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 64, "relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 128, "relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 128, "leaky_relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 16, "relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 16, "leaky_relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 32, "relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 32, "leaky_relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 64, "relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 64, "leaky_relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 128, "relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 128, "leaky_relu"]),
            #
            # Param("ModelGCN", [1, [16, 16], "relu"]),
            # Param("ModelGCN", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGCN", [1, [32, 32], "relu"]),
            # Param("ModelGCN", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGCN", [1, [64, 64], "relu"]),
            # Param("ModelGCN", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGCN", [1, [128, 128], "relu"]),
            # Param("ModelGCN", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGCN", [2, [16, 16, 16], "relu"]),
            # Param("ModelGCN", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN", [2, [32, 32, 32], "relu"]),
            # Param("ModelGCN", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN", [2, [64, 64, 64], "relu"]),
            # Param("ModelGCN", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN", [2, [128, 128, 128], "relu"]),
            # Param("ModelGCN", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGCN", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGCN", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGCN", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGCN", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGCN", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGCN2", [1, [16, 16], "relu"]),
            # Param("ModelGCN2", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGCN2", [1, [32, 32], "relu"]),
            # Param("ModelGCN2", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGCN2", [1, [64, 64], "relu"]),
            # Param("ModelGCN2", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGCN2", [1, [128, 128], "relu"]),
            # Param("ModelGCN2", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGCN2", [2, [16, 16, 16], "relu"]),
            # Param("ModelGCN2", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN2", [2, [32, 32, 32], "relu"]),
            # Param("ModelGCN2", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN2", [2, [64, 64, 64], "relu"]),
            # Param("ModelGCN2", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN2", [2, [128, 128, 128], "relu"]),
            # Param("ModelGCN2", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGCN2", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGCN2", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN2", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGCN2", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN2", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGCN2", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN2", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGCN2", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGCN3", [1, [16, 16], "relu"]),
            # Param("ModelGCN3", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGCN3", [1, [32, 32], "relu"]),
            # Param("ModelGCN3", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGCN3", [1, [64, 64], "relu"]),
            # Param("ModelGCN3", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGCN3", [1, [128, 128], "relu"]),
            # Param("ModelGCN3", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGCN3", [2, [16, 16, 16], "relu"]),
            # Param("ModelGCN3", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN3", [2, [32, 32, 32], "relu"]),
            # Param("ModelGCN3", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN3", [2, [64, 64, 64], "relu"]),
            # Param("ModelGCN3", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN3", [2, [128, 128, 128], "relu"]),
            # Param("ModelGCN3", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGCN3", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGCN3", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN3", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGCN3", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN3", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGCN3", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN3", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGCN3", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGCN4", [1, [16, 16], "relu"]),
            # Param("ModelGCN4", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGCN4", [1, [32, 32], "relu"]),
            # Param("ModelGCN4", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGCN4", [1, [64, 64], "relu"]),
            # Param("ModelGCN4", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGCN4", [1, [128, 128], "relu"]),
            # Param("ModelGCN4", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGCN4", [2, [16, 16, 16], "relu"]),
            # Param("ModelGCN4", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN4", [2, [32, 32, 32], "relu"]),
            # Param("ModelGCN4", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN4", [2, [64, 64, 64], "relu"]),
            # Param("ModelGCN4", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN4", [2, [128, 128, 128], "relu"]),
            # Param("ModelGCN4", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGCN4", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGCN4", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGCN4", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGCN4", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGCN4", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGCN4", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGCN4", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGCN4", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGAT", [1, [16, 16], "relu"]),
            # Param("ModelGAT", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGAT", [1, [32, 32], "relu"]),
            # Param("ModelGAT", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGAT", [1, [64, 64], "relu"]),
            # Param("ModelGAT", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGAT", [1, [128, 128], "relu"]),
            # Param("ModelGAT", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGAT", [2, [16, 16, 16], "relu"]),
            # Param("ModelGAT", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT", [2, [32, 32, 32], "relu"]),
            # Param("ModelGAT", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT", [2, [64, 64, 64], "relu"]),
            # Param("ModelGAT", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT", [2, [128, 128, 128], "relu"]),
            # Param("ModelGAT", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGAT", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGAT", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGAT", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGAT", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGAT", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGAT2", [1, [16, 16], "relu"]),
            # Param("ModelGAT2", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGAT2", [1, [32, 32], "relu"]),
            # Param("ModelGAT2", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGAT2", [1, [64, 64], "relu"]),
            # Param("ModelGAT2", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGAT2", [1, [128, 128], "relu"]),
            # Param("ModelGAT2", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGAT2", [2, [16, 16, 16], "relu"]),
            # Param("ModelGAT2", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT2", [2, [32, 32, 32], "relu"]),
            # Param("ModelGAT2", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT2", [2, [64, 64, 64], "relu"]),
            # Param("ModelGAT2", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT2", [2, [128, 128, 128], "relu"]),
            # Param("ModelGAT2", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGAT2", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGAT2", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT2", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGAT2", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT2", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGAT2", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT2", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGAT2", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGAT3", [1, [16, 16], "relu"]),
            # Param("ModelGAT3", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGAT3", [1, [32, 32], "relu"]),
            # Param("ModelGAT3", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGAT3", [1, [64, 64], "relu"]),
            # Param("ModelGAT3", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGAT3", [1, [128, 128], "relu"]),
            # Param("ModelGAT3", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGAT3", [2, [16, 16, 16], "relu"]),
            # Param("ModelGAT3", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT3", [2, [32, 32, 32], "relu"]),
            # Param("ModelGAT3", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT3", [2, [64, 64, 64], "relu"]),
            # Param("ModelGAT3", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT3", [2, [128, 128, 128], "relu"]),
            # Param("ModelGAT3", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGAT3", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGAT3", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT3", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGAT3", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT3", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGAT3", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT3", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGAT3", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelGAT4", [1, [16, 16], "relu"]),
            # Param("ModelGAT4", [1, [16, 16], "leaky_relu"]),
            # Param("ModelGAT4", [1, [32, 32], "relu"]),
            # Param("ModelGAT4", [1, [32, 32], "leaky_relu"]),
            # Param("ModelGAT4", [1, [64, 64], "relu"]),
            # Param("ModelGAT4", [1, [64, 64], "leaky_relu"]),
            # Param("ModelGAT4", [1, [128, 128], "relu"]),
            # Param("ModelGAT4", [1, [128, 128], "leaky_relu"]),
            # Param("ModelGAT4", [2, [16, 16, 16], "relu"]),
            # Param("ModelGAT4", [2, [16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT4", [2, [32, 32, 32], "relu"]),
            # Param("ModelGAT4", [2, [32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT4", [2, [64, 64, 64], "relu"]),
            # Param("ModelGAT4", [2, [64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT4", [2, [128, 128, 128], "relu"]),
            # Param("ModelGAT4", [2, [128, 128, 128], "leaky_relu"]),
            # Param("ModelGAT4", [3, [16, 16, 16, 16], "relu"]),
            # Param("ModelGAT4", [3, [16, 16, 16, 16], "leaky_relu"]),
            # Param("ModelGAT4", [3, [32, 32, 32, 32], "relu"]),
            # Param("ModelGAT4", [3, [32, 32, 32, 32], "leaky_relu"]),
            # Param("ModelGAT4", [3, [64, 64, 64, 64], "relu"]),
            # Param("ModelGAT4", [3, [64, 64, 64, 64], "leaky_relu"]),
            # Param("ModelGAT4", [3, [128, 128, 128, 128], "relu"]),
            # Param("ModelGAT4", [3, [128, 128, 128, 128], "leaky_relu"]),
            #
            # Param("ModelAPPNP", [10, 0.15, 256, "relu"]),
            # Param("ModelAPPNP", [10, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP", [20, 0.15, 256, "relu"]),
            # Param("ModelAPPNP", [20, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP", [30, 0.15, 256, "relu"]),
            # Param("ModelAPPNP", [30, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 256, "relu"]),
            # Param("ModelAPPNP2", [10, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 256, "relu"]),
            # Param("ModelAPPNP2", [20, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 256, "relu"]),
            # Param("ModelAPPNP2", [30, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 256, "relu"]),
            # Param("ModelAPPNP3", [10, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 256, "relu"]),
            # Param("ModelAPPNP3", [20, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 256, "relu"]),
            # Param("ModelAPPNP3", [30, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 256, "relu"]),
            # Param("ModelAPPNP4", [10, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 256, "relu"]),
            # Param("ModelAPPNP4", [20, 0.15, 256, "leaky_relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 256, "relu"]),
            # Param("ModelAPPNP4", [30, 0.15, 256, "leaky_relu"]),
            #
            # Param("ModelGCN", [1, [256, 256], "relu"]),
            # Param("ModelGCN", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGCN", [2, [256, 256, 256], "relu"]),
            # Param("ModelGCN", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGCN", [3, [256, 256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN2", [1, [256, 256], "relu"]),
            # Param("ModelGCN2", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGCN2", [2, [256, 256, 256], "relu"]),
            # Param("ModelGCN2", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN2", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGCN2", [3, [256, 256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN3", [1, [256, 256], "relu"]),
            # Param("ModelGCN3", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGCN3", [2, [256, 256, 256], "relu"]),
            # Param("ModelGCN3", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN3", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGCN3", [3, [256, 256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN4", [1, [256, 256], "relu"]),
            # Param("ModelGCN4", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGCN4", [2, [256, 256, 256], "relu"]),
            # Param("ModelGCN4", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGCN4", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGCN4", [3, [256, 256, 256, 256], "leaky_relu"]),
            #
            # Param("ModelGAT", [1, [256, 256], "relu"]),
            # Param("ModelGAT", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGAT", [2, [256, 256, 256], "relu"]),
            # Param("ModelGAT", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGAT", [3, [256, 256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT2", [1, [256, 256], "relu"]),
            # Param("ModelGAT2", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGAT2", [2, [256, 256, 256], "relu"]),
            # Param("ModelGAT2", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT2", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGAT2", [3, [256, 256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT3", [1, [256, 256], "relu"]),
            # Param("ModelGAT3", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGAT3", [2, [256, 256, 256], "relu"]),
            # Param("ModelGAT3", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT3", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGAT3", [3, [256, 256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT4", [1, [256, 256], "relu"]),
            # Param("ModelGAT4", [1, [256, 256], "leaky_relu"]),
            # Param("ModelGAT4", [2, [256, 256, 256], "relu"]),
            # Param("ModelGAT4", [2, [256, 256, 256], "leaky_relu"]),
            # Param("ModelGAT4", [3, [256, 256, 256, 256], "relu"]),
            # Param("ModelGAT4", [3, [256, 256, 256, 256], "leaky_relu"]),
            #
            # Param("ModelDGI", [16]),
            # Param("ModelDGI", [32]),
            # Param("ModelDGI", [64]),
            # Param("ModelDGI", [128]),
            # Param("ModelDGI", [256]),
        ]

        f = open('result.txt', 'w', encoding='utf-8')
        for param in params:
            average_loss_train = 0
            average_loss_valid = 0
            average_accuracy_train = 0
            average_accuracy_valid = 0
            average_accuracy_test = 0
            average_epoch = 0
            for _ in range(5):
                if param.model == "ModelGCN":
                    result = main_model_gcn(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelGAT":
                    result = main_model_gat(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelAPPNP":
                    result = main_model_appnp(
                        data=data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=False
                    )
                elif param.model == "ModelGCN2":
                    result = main_model_gcn_2(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelGAT2":
                    result = main_model_gat_2(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelAPPNP2":
                    result = main_model_appnp_2(
                        data=data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=False
                    )
                elif param.model == "ModelGCN3":
                    result = main_model_gcn_3(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelGAT3":
                    result = main_model_gat_3(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelAPPNP3":
                    result = main_model_appnp_3(
                        data=data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=False
                    )
                elif param.model == "ModelGCN4":
                    result = main_model_gcn_4(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelGAT4":
                    result = main_model_gat_4(
                        data=data,
                        num_layers=param.param[0],
                        hidden_list=param.param[1],
                        activation=param.param[2],
                        if_all=False
                    )
                elif param.model == "ModelAPPNP4":
                    result = main_model_appnp_4(
                        data=data,
                        K=param.param[0],
                        alpha=param.param[1],
                        hidden=param.param[2],
                        activation=param.param[3],
                        if_all=False
                    )
                elif param.model == "ModelDGI":
                    result = main_model_dgi(
                        data=data,
                        hidden=param.param[0],
                        if_all=False
                    )
                else:
                    raise ValueError("Model name error: {0}".format(param[0]))

                solution = pd.read_csv('../data/{0}/test_label.tsv'.format(data_set), sep='\t')['label']
                acc_test = accuracy_score(solution, result.result)
                average_loss_train += result.loss_train / 5.0
                average_loss_valid += result.loss_valid / 5.0
                average_accuracy_train += result.acc_train / 5.0
                average_accuracy_valid += result.acc_valid / 5.0
                average_accuracy_test += acc_test / 5.0
                average_epoch += result.epoch / 5.0
                f.write("Loss Train 166092: {0:.4f}\n".format(result.loss_train))
                f.write("Loss Valid 166092: {0:.4f}\n".format(result.loss_valid))
                f.write("Acc Train 166092: {0:.4f}\n".format(result.acc_train))
                f.write("Acc Valid 166092: {0:.4f}\n".format(result.acc_valid))
                f.write("Acc Test 166092: {0:.4f}\n".format(acc_test))
                f.write("Epoch 166092: {0:.4f}\n".format(result.epoch))
                f.flush()
            f.write("Average Loss Train 166092: {0:.4f}\n".format(average_loss_train))
            f.write("Average Loss Valid 166092: {0:.4f}\n".format(average_loss_valid))
            f.write("Average Acc Train 166092: {0:.4f}\n".format(average_accuracy_train))
            f.write("Average Acc Valid 166092: {0:.4f}\n".format(average_accuracy_valid))
            f.write("Average Acc Test 166092: {0:.4f}\n".format(average_accuracy_test))
            f.write("Average Epoch 166092: {0:.4f}\n".format(average_epoch))
            f.flush()
        f.close()


Model().train_predict()
