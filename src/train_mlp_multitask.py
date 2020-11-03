import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile
import shutil
import logging

import mlflow
from sklearn import metrics
import torch
import torch.nn as nn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from utils import (
    seed_everything,
    Timer,
    reduce_mem_usage,
    load_from_pkl, save_as_pkl
)
from dataset import SimpleDataLoader
from model import (
    DNN_multitask_v2
)

logging.basicConfig(level=logging.INFO)

DEVICE = os.environ.get('DEVICE')

EXP_NAME = os.environ.get('EXP_NAME')
EXP_DIR = os.environ.get('EXP_DIR')

INPUT_DIR = os.environ.get('INPUT_DIR')
FEATURE_DIR = os.environ.get('FEATURE_DIR')
FOLD_DIR = os.environ.get('FOLD_DIR')
SUB_DIR = os.environ.get('SUB_DIR')

sys.path.append(f'{EXP_DIR}/{EXP_NAME}')
import config

VARLEN_MAX_LEN = config.VARLEN_MAX_LEN

FOLD_NAME = config.FOLD_NAME
FOLD_NUM = config.FOLD_NUM
RANDOM_STATE = config.RANDOM_STATE

EPOCH_NUM = config.EPOCH_NUM
BATCH_SIZE = config.BATCH_SIZE
DNN_HIDDEN_UNITS = config.DNN_HIDDEN_UNITS
DNN_HIDDEN_UNITS_EACH_TASK = config.DNN_HIDDEN_UNITS_EACH_TASK
DNN_DROPOUT = config.DNN_DROPOUT
DNN_ACTIVATION = config.DNN_ACTIVATION
L2_REG = config.L2_REG
INIT_STD = config.INIT_STD

SPAESE_EMBEDDING_DIM = config.SPAESE_EMBEDDING_DIM

LR = config.LR
OPTIMIZER = config.OPTIMIZER
LOSS = config.LOSS
OPTIM_TARGET = config.OPTIM_TARGET

dense_features = config.dense_features
sparse_features = config.sparse_features
varlen_sparse_features = config.varlen_sparse_features


def save_mlflow(run_id, cv, fold_best_scores, best_acc, best_th, oof_metric_t2):

    mlflow.log_param("fold_name", FOLD_NAME)
    mlflow.log_param("fold_num", FOLD_NUM)

    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("loss", LOSS)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("learning rate", LR)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.log_param("dnn_hidden_layer", DNN_HIDDEN_UNITS)
    mlflow.log_param("dnn_dropout", DNN_DROPOUT)
    mlflow.log_param("dnn_activation", DNN_ACTIVATION)
    mlflow.log_param("l2_reg", L2_REG)
    mlflow.log_param("init_std", INIT_STD)

    mlflow.log_param("embedding_dim", SPAESE_EMBEDDING_DIM)

    for feat in dense_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__dense__{feat}', 1)
    for feat in sparse_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__sparse__{feat}', 1)
    for feat in varlen_sparse_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__varspa__{feat}', 1)

    mlflow.log_metric("oof acc", best_acc)
    mlflow.log_metric("th", best_th)
    mlflow.log_metric("oof acc t2", oof_metric_t2)
    mlflow.log_metric("cv", cv)
    for fold_idx in range(FOLD_NUM):
        mlflow.log_metric(f'val_metric_t1_{fold_idx}', fold_best_scores[fold_idx][1])
    for fold_idx in range(FOLD_NUM):
        mlflow.log_metric(f'val_metric_t2_{fold_idx}', fold_best_scores[fold_idx][2])

    return


def save_best_score(fold_best_scores, model, val_metric, val_metric_t2, run_id, fold_idx):
    if not os.path.exists(f'../save/{EXP_NAME}'):
        os.mkdir(f'../save/{EXP_NAME}')
        os.mkdir(f'../save/{EXP_NAME}/model_weight')
    weight_path = f'../save/{EXP_NAME}/model_weight/train_weights_mlflow-{run_id}_fold{fold_idx}.h5'
    torch.save(model.state_dict(), weight_path)
    fold_best_scores[fold_idx] = (weight_path, val_metric, val_metric_t2)
    mlflow.log_artifact(weight_path)
    return fold_best_scores


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


def save_as_feather(feat, suffix, X_train, X_test):
    X_train[[feat]].reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_{suffix}_train.feather')
    X_test[[feat]].reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_{suffix}_test.feather')
    return


if __name__ == "__main__":

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer(f'read label'):
        data_path = f'{INPUT_DIR}/train_data/train_task_1_2.csv'
        y_train = pd.read_csv(data_path, usecols=['IsCorrect', 'AnswerValue'])
        y_train_t1 = y_train['IsCorrect'].values
        y_train_t2 = (y_train['AnswerValue'] - 1).values  # starting at zero

    with t.timer(f'apply mms'):
        for feat in dense_features:
            if os.path.exists(f'{FEATURE_DIR}/{feat}_mms_test.feather'):
                continue
            # MMS
            f_train = pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')
            f_test = pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')
            tmp = pd.concat([f_train[feat], f_test[feat]])
            print(tmp)
            max_v, min_v = tmp.max(), tmp.min()
            logging.info(f'{feat} - MMS Scaling > max: {max_v}, min: {min_v}')
            f_train[feat] = (f_train[feat] - min_v) / (max_v - min_v)
            f_test[feat] = (f_test[feat] - min_v) / (max_v - min_v)
            save_as_feather(feat, 'mms', f_train, f_test)

    skip_fr = False
    if skip_fr is False:
        with t.timer(f'read features'):
            unique_num_dic = {}
            feature_index = {}

            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            fidx = 0
            for feat in dense_features:
                logging.info(f'[{feat}] read feature ...')
                feature_index[feat] = fidx
                fidx += 1
                X_train = pd.concat([
                    X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_mms_train.feather')
                ], axis=1)
                X_test = pd.concat([
                    X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_mms_test.feather')
                ], axis=1)
            X_train = reduce_mem_usage(X_train)
            X_test = reduce_mem_usage(X_test)
            for feat in sparse_features:
                logging.info(f'[{feat}] read feature ...')
                feature_index[feat] = fidx
                fidx += 1
                X_train = pd.concat([
                    X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')
                ], axis=1)
                X_test = pd.concat([
                    X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')
                ], axis=1)
                unique_num = pd.concat([
                    X_train[feat], X_test[feat]
                ]).nunique()
                unique_num_dic[feat] = unique_num
            X_train = reduce_mem_usage(X_train)
            X_test = reduce_mem_usage(X_test)

            for feat in varlen_sparse_features:
                logging.info(f'[{feat}] read feature ...')
                feature_index[feat] = (fidx, fidx + VARLEN_MAX_LEN)
                fidx += VARLEN_MAX_LEN

                # train_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather').values
                train_feat = pd.read_parquet(f'{FEATURE_DIR}/{feat}_train.parquet').values
                varlen_list = [i[0] for i in train_feat]
                varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
                X_train = pd.concat([
                    X_train, pd.DataFrame(varlen_list)
                ], axis=1)

                # test_feat = pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather').values
                test_feat = pd.read_parquet(f'{FEATURE_DIR}/{feat}_test.parquet').values
                varlen_list = [i[0] for i in test_feat]
                varlen_list = pad_sequences(varlen_list, maxlen=VARLEN_MAX_LEN, padding='post', )
                X_test = pd.concat([
                    X_test, pd.DataFrame(varlen_list)
                ], axis=1)

                tmp = []
                for i in [i[0] for i in train_feat] + [i[0] for i in test_feat]:
                    tmp.extend(i)
                unique_num = len(set(tmp)) + 1
                unique_num_dic[feat] = unique_num

            print('Unique Num', unique_num_dic)
            print('Feature index', feature_index)

            save_as_pkl(X_train, f'X_train_{EXP_NAME}.pkl')
            save_as_pkl(X_test, f'X_test_{EXP_NAME}.pkl')
            save_as_pkl(unique_num_dic, f'unique_num_dic_{EXP_NAME}.pkl')
            save_as_pkl(feature_index, f'feature_index_{EXP_NAME}.pkl')

    elif skip_fr is True:
        X_train = load_from_pkl(f'X_train_{EXP_NAME}.pkl')
        X_test = load_from_pkl(f'X_test_{EXP_NAME}.pkl')
        unique_num_dic = load_from_pkl(f'unique_num_dic_{EXP_NAME}.pkl')
        feature_index = load_from_pkl(f'feature_index_{EXP_NAME}.pkl')

    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)

    with t.timer(f'load folds: {FOLD_NAME}-{FOLD_NUM}'):
        folds = pd.read_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv')

    mlflow.set_experiment(EXP_NAME)
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

    fold_best_scores = {}  # fold_idx:best_cv_score
    for fold_idx in range(FOLD_NUM):

        trn_idx = folds[folds.kfold != fold_idx].index.tolist()
        val_idx = folds[folds.kfold == fold_idx].index.tolist()

        x_trn = X_train.iloc[trn_idx]
        x_val = X_train.iloc[val_idx]

        # if 
        y_trn = y_train_t1[trn_idx]
        y_val = y_train_t1[val_idx]
        y2_trn = y_train_t2[trn_idx]
        y2_val = y_train_t2[val_idx]

        # if
        train_loader = SimpleDataLoader(
            [torch.from_numpy(x_trn.values), torch.from_numpy(y_trn), torch.from_numpy(y2_trn)],
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        sparse = False
        embedding_dict = nn.ModuleDict(
            {
                feat: nn.Embedding(
                    unique_num_dic[feat], SPAESE_EMBEDDING_DIM, sparse=sparse
                ) for feat in sparse_features + varlen_sparse_features
            }
        )

        dnn_input_len = len(dense_features) + len(sparse_features + varlen_sparse_features) * SPAESE_EMBEDDING_DIM

        # if
        model = DNN_multitask_v2(
            dnn_input=dnn_input_len,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_hidden_units_task=DNN_HIDDEN_UNITS_EACH_TASK,
            dnn_dropout=DNN_DROPOUT,
            activation=DNN_ACTIVATION, use_bn=True, l2_reg=L2_REG, init_std=INIT_STD,
            device=DEVICE,
            feature_index=feature_index,
            embedding_dict=embedding_dict,
            dense_features=dense_features,
            sparse_features=sparse_features,
            varlen_sparse_features=varlen_sparse_features,
        )

        loss_func = nn.BCELoss()
        loss_func_t2 = nn.CrossEntropyLoss()

        optim = torch.optim.Adam(model.parameters(), lr=LR)
        metric_func = metrics.roc_auc_score
        metric_func_t2 = multi_acc

        loss_history = []
        steps_per_epoch = (len(x_trn) - 1) // BATCH_SIZE + 1

        if OPTIM_TARGET in ['AUC_t1', 'ACC_t2']:
            best_score = 0.0
        elif OPTIM_TARGET in ['BCE_t1', 'CE_t2', 'BCE-CE_t12']:
            best_score = 999.9
        for epoch in range(EPOCH_NUM):

            # train
            loss_history_epoch = []
            loss_t2_history_epoch = []
            metric_history_epoch = []
            metric_t2_history_epoch = []

            logging.info(f'[{DEVICE}][FOLD:{fold_idx}] EPOCH - {epoch} / {EPOCH_NUM}')
            model = model.train()
            for bi, (bx, by, by2) in tqdm(enumerate(train_loader), total=steps_per_epoch):

                optim.zero_grad()

                x = bx.to(DEVICE).float()

                # if 
                y = by.to(DEVICE).float().squeeze()
                y2 = by2.to(DEVICE).long().squeeze()
                y_pred, y_pred_t2 = model(x)
                y_pred = y_pred.squeeze()
                y_pred_t2 = y_pred_t2.squeeze()

                # if
                loss_t1 = loss_func(y_pred, y)
                loss_t2 = loss_func_t2(y_pred_t2, y2)

                loss = loss_t1 + loss_t2
                loss.backward()

                optim.step()

                loss_history_epoch.append(loss.item())
                loss_t2_history_epoch.append(loss_t2.item())
                
                # if
                y_pred_np = y_pred.cpu().detach().numpy().reshape(-1, 1)
                y_np = y.cpu().detach().numpy().reshape(-1, 1)
                if len(np.unique(y_np)) == 1:
                    # AUC error (ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.)
                    continue
                else:
                    metric = metric_func(y_np, y_pred_np)
                    metric_history_epoch.append(metric)

                metric_t2 = metric_func_t2(y_pred_t2, y2)
                metric_t2_history_epoch.append(metric_t2)

            # if
            trn_loss_epoch = sum(loss_history_epoch) / len(loss_history_epoch)
            trn_metric_epoch = sum(metric_history_epoch) / len(metric_history_epoch)

            trn_loss_t2_epoch = sum(loss_t2_history_epoch) / len(loss_t2_history_epoch)
            trn_metric_t2_epoch = sum(metric_t2_history_epoch) / len(metric_t2_history_epoch)

            # validation evaluate
            # if 
            preds_val, preds_t2_val = model.predict(x_val, BATCH_SIZE)
            val_loss = loss_func(
                torch.from_numpy(preds_val.reshape(-1, 1)).to(torch.float32), torch.from_numpy(y_val.reshape(-1, 1)).to(torch.float32)
            ).item()
            val_metric = metric_func(y_val, preds_val)

            val_loss_t2 = loss_func_t2(
                torch.from_numpy(preds_t2_val), torch.from_numpy(y2_val).long()
            ).item()
            val_metric_t2 = metric_func_t2(
                torch.from_numpy(preds_t2_val), torch.from_numpy(y2_val)
            ).item()

            # if 
            logging.info(f'Train - Loss(t1): {trn_loss_epoch}, Loss(t2): {trn_loss_t2_epoch}, AUC(t1): {trn_metric_epoch}, ACC(t2): {trn_metric_t2_epoch}')
            logging.info(f'Valid - Loss(t1): {val_loss}, Loss(t2): {val_loss_t2}, AUC(t1): {val_metric}, ACC(t2): {val_metric_t2}')
            loss_history.append([
                epoch, trn_loss_epoch, val_loss, trn_loss_t2_epoch, val_loss_t2
            ])

            if OPTIM_TARGET in ['AUC_t1']:
                score = val_metric
                if score > best_score:
                    best_score = score
                    fold_best_scores = save_best_score(fold_best_scores, model, val_metric, val_metric_t2, run_id, fold_idx)
            elif OPTIM_TARGET in ['BCE_t1']:
                score = val_loss
                if score < best_score:
                    best_score = score
                    fold_best_scores = save_best_score(fold_best_scores, model, val_metric, val_metric_t2, run_id, fold_idx)
            elif OPTIM_TARGET in ['ACC_t2']:
                score = val_metric_t2
                if score > best_score:
                    best_score = score
                    fold_best_scores = save_best_score(fold_best_scores, model, val_metric, val_metric_t2, run_id, fold_idx)
            elif OPTIM_TARGET in ['CE_t2']:
                score = val_loss_t2
                if score < best_score:
                    best_score = score
                    fold_best_scores = save_best_score(fold_best_scores, model, val_metric, val_metric_t2, run_id, fold_idx)
            elif OPTIM_TARGET in ['BCE-CE_t12']:
                score = val_loss + val_loss_t2
                if score < best_score:
                    best_score = score
                    fold_best_scores = save_best_score(fold_best_scores, model, val_metric, val_metric_t2, run_id, fold_idx)

        if not os.path.exists(f'../save/{EXP_NAME}/model_log'):
            os.mkdir(f'../save/{EXP_NAME}/model_log')
        history_path = f'../save/{EXP_NAME}/model_log/loss_history-{run_id}_fold{fold_idx}.csv'
        # if
        pd.DataFrame(loss_history, columns=['epoch', 'trn_loss', 'val_loss', 'trn_loss_t2', 'val_loss_t2']).to_csv(history_path)
        mlflow.log_artifact(history_path)

    cv = 0.0
    cv_t2 = 0.0
    for fold_idx in range(FOLD_NUM):
        weight_path, score, score_t2 = fold_best_scores[fold_idx]
        cv += score / FOLD_NUM
        cv_t2 += score_t2 / FOLD_NUM
    logging.info(f"CV(t1): {cv}, CV(t2): {cv_t2}")

    oof = np.zeros(len(X_train))
    oof_t2 = np.zeros((len(X_train), 4))
    preds_test = np.zeros(len(X_test))
    preds_test_t2 = np.zeros((len(X_test), 4))
    for fold_idx in range(FOLD_NUM):

        val_idx = folds[folds.kfold == fold_idx].index.tolist()
        x_val = X_train.iloc[val_idx]

        model = DNN_multitask_v2(
            dnn_input=dnn_input_len,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_hidden_units_task=DNN_HIDDEN_UNITS_EACH_TASK,
            dnn_dropout=DNN_DROPOUT,
            activation=DNN_ACTIVATION, use_bn=True, l2_reg=L2_REG, init_std=INIT_STD,
            device=DEVICE,
            feature_index=feature_index,
            embedding_dict=embedding_dict,
            dense_features=dense_features,
            sparse_features=sparse_features,
            varlen_sparse_features=varlen_sparse_features,
        )
        weight_path, score, score_t2 = fold_best_scores[fold_idx]
        model.load_state_dict(torch.load(weight_path))

        preds_val_fold, preds_val_t2_fold = model.predict(x_val, BATCH_SIZE)
        oof[val_idx] = preds_val_fold
        oof_t2[val_idx] = preds_val_t2_fold

        preds_test_fold, preds_test_t2_fold = model.predict(X_test, BATCH_SIZE)
        preds_test += preds_test_fold / FOLD_NUM
        preds_test_t2 += preds_test_t2_fold / FOLD_NUM

    if not os.path.exists(f'../save/{EXP_NAME}'):
        os.mkdir(f'../save/{EXP_NAME}')

    pd.DataFrame(oof).to_csv(f'../save/{EXP_NAME}/preds_val_task1_{run_id}.csv', index=False)
    pd.DataFrame(preds_test).to_csv(f'../save/{EXP_NAME}/preds_test_task1_{run_id}.csv', index=False)
    pd.DataFrame(oof_t2).to_csv(f'../save/{EXP_NAME}/preds_val_task2_{run_id}.csv', index=False)
    pd.DataFrame(preds_test_t2).to_csv(f'../save/{EXP_NAME}/preds_test_task2_{run_id}.csv', index=False)

    # oof_t2 = np.argmax(oof_t2, axis=1)
    preds_test_t2 = np.argmax(preds_test_t2, axis=1)

    rows = []
    for th in range(40, 60, 1):
        th = th * 0.01
        preds_th = []
        for i in oof:
            if i > th:
                preds_th.append(1)
            else:
                preds_th.append(0)
        acc = metrics.accuracy_score(y_train_t1, preds_th)
        rows.append([th, acc])
    acc_th = pd.DataFrame(rows, columns=['th', 'acc'])
    tmp = acc_th.sort_values('acc', ascending=False).head(1)
    best_th, best_acc = tmp.values[0]

    oof_metric_t2 = metric_func_t2(
        torch.from_numpy(oof_t2), torch.from_numpy(y_train_t2)
    ).item()

    logging.info(f'OOF ACC(t1): {best_acc}, TH:{best_th}, OOF ACC(t2): {oof_metric_t2}')
    save_mlflow(run_id, cv, fold_best_scores, best_acc, best_th, oof_metric_t2)

    # task1
    preds_test_th = []
    for i in preds_test:
        if i > best_th:
            preds_test_th.append(1)
        else:
            preds_test_th.append(0)

    test_data_path = f'../submission_templates/submission_task_1_2.csv'
    sub = pd.read_csv(test_data_path)
    sub['IsCorrect'] = preds_test_th

    if not os.path.exists(f'{SUB_DIR}/{EXP_NAME}'):
        os.mkdir(f'{SUB_DIR}/{EXP_NAME}')

    sub_name = f'submission_task1__auc{cv}__acc{best_acc}__th{best_th}'
    valid_sub_dir = f'{SUB_DIR}/{EXP_NAME}/{sub_name}'
    if not os.path.exists(valid_sub_dir):
        os.mkdir(valid_sub_dir)

    sub.to_csv(f'{valid_sub_dir}/submission_task_1.csv', index=False)
    with zipfile.ZipFile(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        new_zip.write(f'{valid_sub_dir}/submission_task_1.csv', arcname='submission_task_1.csv')
    shutil.rmtree(valid_sub_dir)
    mlflow.log_artifact(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip')

    # task2
    test_data_path = f'../submission_templates/submission_task_1_2.csv'
    sub = pd.read_csv(test_data_path)
    sub['AnswerValue'] = preds_test_t2
    sub['AnswerValue'] = sub['AnswerValue'] + 1

    if not os.path.exists(f'{SUB_DIR}/{EXP_NAME}'):
        os.mkdir(f'{SUB_DIR}/{EXP_NAME}')

    sub_name = f'submission_task2__acc{oof_metric_t2}'
    valid_sub_dir = f'{SUB_DIR}/{EXP_NAME}/{sub_name}'
    if not os.path.exists(valid_sub_dir):
        os.mkdir(valid_sub_dir)

    sub.to_csv(f'{valid_sub_dir}/submission_task_2.csv', index=False)
    with zipfile.ZipFile(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        new_zip.write(f'{valid_sub_dir}/submission_task_2.csv', arcname='submission_task_2.csv')
    shutil.rmtree(valid_sub_dir)
    mlflow.log_artifact(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip')

    mlflow.end_run()
