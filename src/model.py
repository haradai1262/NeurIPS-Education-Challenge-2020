import torch
import torch.nn as nn
import numpy as np

from dataset import SimpleDataLoader
from utils_model import DNN, PredictionLayer, get_varlen_pooling_list


class DNN_multitask_v2(nn.Module):
    def __init__(
        self,
        dnn_input, dnn_hidden_units, dnn_hidden_units_task, dnn_dropout,
        activation='relu', use_bn=True, l2_reg=1e-4, init_std=1e-4,
        device='cpu',
        feature_index={},
        embedding_dict={},
        dense_features=[],
        sparse_features=[],
        varlen_sparse_features=[],
    ):
        super().__init__()
        self.device = device
        self.feature_index = feature_index
        self.embedding_dict = embedding_dict
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.varlen_sparse_features = varlen_sparse_features

        self.dnn = DNN(
            dnn_input, dnn_hidden_units,
            activation='relu', l2_reg=l2_reg, dropout_rate=dnn_dropout, use_bn=use_bn,
            init_std=init_std, device=device
        )
        self.dnn_t1 = DNN(
            dnn_hidden_units[-1], dnn_hidden_units_task,
            activation='relu', l2_reg=l2_reg, dropout_rate=dnn_dropout, use_bn=use_bn,
            init_std=init_std, device=device
        )
        self.dnn_t2 = DNN(
            dnn_hidden_units[-1], dnn_hidden_units_task,
            activation='relu', l2_reg=l2_reg, dropout_rate=dnn_dropout, use_bn=use_bn,
            init_std=init_std, device=device
        )
        self.dnn_linear_t1 = nn.Linear(dnn_hidden_units_task[-1], 1, bias=False).to(device)
        self.dnn_linear_t2 = nn.Linear(dnn_hidden_units_task[-1], 4, bias=False).to(device)

        task = 'binary'
        self.out_t1 = PredictionLayer(task,)
        self.out_t2 = nn.Softmax(dim=1)
        self.to(device)

    def forward(self, X):

        dense_value_list = [
            X[:, self.feature_index[feat]: self.feature_index[feat] + 1] for feat in self.dense_features
        ]
        sparse_embedding_list = [
            self.embedding_dict[feat](
                X[:, self.feature_index[feat]].long()
            ) for feat in self.sparse_features
        ]
        varlen_sparse_embedding_list = get_varlen_pooling_list(
            self.embedding_dict, X, self.feature_index, self.varlen_sparse_features, self.device
        )
        sparse_embedding_list = sparse_embedding_list + varlen_sparse_embedding_list

        sparse_dnn_input = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)

        dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
        dnn_hidden = self.dnn(dnn_input)

        dnn_output_t1 = self.dnn_t1(dnn_hidden)
        dnn_logit_t1 = self.dnn_linear_t1(dnn_output_t1)
        y_pred_t1 = self.out_t1(dnn_logit_t1)
        y_pred_t1 = torch.clamp(y_pred_t1, min=1e-8, max=1.0 - 1e-8)

        dnn_output_t2 = self.dnn_t2(dnn_hidden)
        dnn_logit_t2 = self.dnn_linear_t2(dnn_output_t2)
        y_pred_t2 = self.out_t2(dnn_logit_t2)

        return y_pred_t1, y_pred_t2

    def predict(self, x, batch_size=256):

        model = self.eval()
        test_loader = SimpleDataLoader(
            [torch.from_numpy(x.values)],
            batch_size=batch_size,
            shuffle=False
        )

        pred_ans = []
        pred_t2_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred, y_pred_t2 = model(x)
                y_pred = y_pred.squeeze()
                y_pred_t2 = y_pred_t2.squeeze()
                pred_ans.append(y_pred.cpu().detach().numpy())
                pred_t2_ans.append(y_pred_t2.cpu().detach().numpy())

        return np.concatenate(pred_ans), np.concatenate(pred_t2_ans)