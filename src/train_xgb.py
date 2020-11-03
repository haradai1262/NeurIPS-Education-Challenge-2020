import os
import sys
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile
import shutil
import logging

import mlflow
from sklearn import metrics

import xgboost as xgb
print('XGB Version', xgb.__version__)

from utils import (
    seed_everything,
    Timer,
    reduce_mem_usage,
    load_from_pkl, save_as_pkl
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

FOLD_NAME = config.FOLD_NAME
FOLD_NUM = config.FOLD_NUM
RANDOM_STATE = config.RANDOM_STATE

TARGET_TASK = config.TARGET_TASK

XGB_MDOEL_PARAMS = config.XGB_MDOEL_PARAMS
XGB_MDOEL_PARAMS['gpu_id'] = int(DEVICE.split(':')[1])
XGB_TRAIN_PARAMS = config.XGB_TRAIN_PARAMS

dense_features = config.dense_features
sparse_features = config.sparse_features
varlen_sparse_features = config.varlen_sparse_features


def save_mlflow(run_id, cv, acc=None, th=None):

    mlflow.log_param("fold_name", FOLD_NAME)
    mlflow.log_param("fold_num", FOLD_NUM)

    for feat in dense_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__dense__{feat}', 1)
    for feat in sparse_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__sparse__{feat}', 1)
    for feat in varlen_sparse_features:
        feat = feat.replace('#', '')
        mlflow.log_param(f'f__varspa__{feat}', 1)

    mlflow.log_metric("cv", cv)
    if acc is not None:
        mlflow.log_metric("acc", acc)
    if th is not None:
        mlflow.log_metric("th", th)
    return


def train_1fold(preds, preds_test, importance, trn_idx, val_idx, X_train, y_train, d_test):

    x_trn = X_train.iloc[trn_idx]
    y_trn = y_train[trn_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train[val_idx]

    d_train = xgb.DMatrix(data=x_trn, label=y_trn)
    d_val = xgb.DMatrix(data=x_val, label=y_val)

    model = xgb.train(
        XGB_MDOEL_PARAMS,
        dtrain=d_train,
        evals=[(d_train, "train"), (d_val, "val")],
        **XGB_TRAIN_PARAMS
    )
    preds[val_idx] = model.predict(d_val)
    preds_test += model.predict(d_test)

    importance_fold = model.get_score(importance_type='weight')
    sum_imp = sum(importance_fold.values())
    for f, s in importance_fold.items():
        if f not in importance:
            importance[f] = 0
        importance[f] += s / sum_imp
    return preds, preds_test, importance


if __name__ == "__main__":

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer(f'read label'):
        data_path = f'{INPUT_DIR}/train_data/train_task_1_2.csv'
        train = pd.read_csv(data_path)
        if TARGET_TASK == '1':
            y_train = train['IsCorrect'].values
        elif TARGET_TASK == '2':
            y_train = (train['AnswerValue'] - 1).values

    skip_fr = False
    if skip_fr is False:
        with t.timer(f'read features'):

            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            for feat in dense_features:
                logging.info(f'[{feat}] read feature ...')
                X_train = pd.concat([
                    X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')
                ], axis=1)
                X_test = pd.concat([
                    X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')
                ], axis=1)
            X_train = reduce_mem_usage(X_train)
            X_test = reduce_mem_usage(X_test)
            for feat in sparse_features:
                logging.info(f'[{feat}] read feature ...')
                X_train = pd.concat([
                    X_train, pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')
                ], axis=1)
                X_test = pd.concat([
                    X_test, pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')
                ], axis=1)
            X_train = reduce_mem_usage(X_train)
            X_test = reduce_mem_usage(X_test)
            save_as_pkl(X_train, f'X_train_{EXP_NAME}_.pkl')
            save_as_pkl(X_test, f'X_test_{EXP_NAME}_.pkl')
    elif skip_fr is True:
        X_train = load_from_pkl(f'X_train_task1_xgb_fs100_meta_.pkl')
        X_test = load_from_pkl(f'X_test_task1_xgb_fs100_meta_.pkl')
        cat_features_index = []

    mlflow.set_experiment(EXP_NAME)
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

    with t.timer(f'load folds: {FOLD_NAME}-{FOLD_NUM}'):
        folds = pd.read_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv')

    with t.timer(f'train XGB'):

        logging.info(f'Num. of Samples: {len(X_train)}')
        logging.info(f'Num. of Features: {X_train.shape[1]}')

        if TARGET_TASK == '1':
            preds = np.zeros(len(X_train))
            preds_test = np.zeros(len(X_test))
        elif TARGET_TASK == '2':
            preds = np.zeros((len(X_train), XGB_MDOEL_PARAMS['num_class']))
            preds_test = np.zeros((len(X_test), XGB_MDOEL_PARAMS['num_class']))

        d_test = xgb.DMatrix(data=X_test)

        importance = {}
        for fold_idx in range(FOLD_NUM):

            logging.info(f'FOLD:{fold_idx}')

            trn_idx = folds[folds.kfold != fold_idx].index.tolist()
            val_idx = folds[folds.kfold == fold_idx].index.tolist()

            preds, preds_test, importance = train_1fold(
                preds, preds_test, importance,
                trn_idx, val_idx, X_train, y_train, d_test
            )

        preds_test /= FOLD_NUM
        df_score = pd.DataFrame(
            [(i, j) for i, j in importance.items()], columns=['fname', 'importance']
        ).sort_values('importance', ascending=False)

        if not os.path.exists(f'../save/{EXP_NAME}'):
            os.mkdir(f'../save/{EXP_NAME}')

        pd.DataFrame(preds).to_csv(f'../save/{EXP_NAME}/preds_val_task{TARGET_TASK}_{run_id}.csv', index=False)
        pd.DataFrame(preds_test).to_csv(f'../save/{EXP_NAME}/preds_test_task{TARGET_TASK}_{run_id}.csv', index=False)
        df_score.to_csv(f'../save/{EXP_NAME}/xgb_feature_importance_task{TARGET_TASK}_{run_id}.csv', index=False)

    if TARGET_TASK == '1':
        with t.timer(f'postprocess, threshold'):
            rows = []
            for th in tqdm(range(490, 510, 1)):
                th = th * 0.001
                preds_th = []
                for i in preds:
                    if i > th:
                        preds_th.append(1)
                    else:
                        preds_th.append(0)
                acc = metrics.accuracy_score(y_train, preds_th)
                rows.append([th, acc])
            acc_th = pd.DataFrame(rows, columns=['th', 'acc'])

            cv = metrics.roc_auc_score(y_train, preds)
            scores = acc_th.sort_values('acc', ascending=False).head(1).values[0]
            best_th, best_acc = scores[0], scores[1]
            logging.info(f'Val AUC: {cv}',)
            logging.info(f'Val Best Acc: {best_acc} (threshold: {best_th})')

        with t.timer(f'make submission'):
            preds_test_th = []
            for i in preds_test:
                if i > best_th:
                    preds_test_th.append(1)
                else:
                    preds_test_th.append(0)

            test_data_path = f'../submission_templates/submission_task_1_2.csv'
            sub = pd.read_csv(test_data_path)
            sub['IsCorrect'] = preds_test_th

            if not os.path.exists(f'../submission/{EXP_NAME}'):
                os.mkdir(f'../submission/{EXP_NAME}')

            sub_name = f'submission_task1__auc{cv}__acc{best_acc}__th{best_th}'
            valid_sub_dir = f'{SUB_DIR}/{EXP_NAME}/{sub_name}'
            if not os.path.exists(valid_sub_dir):
                os.mkdir(valid_sub_dir)

            sub.to_csv(f'{valid_sub_dir}/submission_task_1.csv', index=False)
            with zipfile.ZipFile(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
                new_zip.write(f'{valid_sub_dir}/submission_task_1.csv', arcname='submission_task_1.csv')
            shutil.rmtree(valid_sub_dir)
            mlflow.log_artifact(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip')

            submission_path = f'../submission/{EXP_NAME}/submission_task1__auc{cv}__acc{best_acc}__th{best_th}.csv'
            sub.to_csv(submission_path, index=False)

    elif TARGET_TASK == '2':

        preds_label = np.argmax(preds, axis=1) + 1
        preds_test_label = np.argmax(preds_test, axis=1) + 1
        cv = metrics.accuracy_score(y_train + 1, preds_label)
        logging.info(f'CV Multiclass Acc: {cv}')

        with t.timer(f'make submission'):
            test_data_path = f'../submission_templates/submission_task_1_2.csv'
            sub = pd.read_csv(test_data_path)
            sub['AnswerValue'] = preds_test_label

            if not os.path.exists(f'{SUB_DIR}/{EXP_NAME}'):
                os.mkdir(f'{SUB_DIR}/{EXP_NAME}')

            sub_name = f'submission_task2__acc{cv}'
            valid_sub_dir = f'{SUB_DIR}/{EXP_NAME}/{sub_name}'
            if not os.path.exists(valid_sub_dir):
                os.mkdir(valid_sub_dir)

            sub.to_csv(f'{valid_sub_dir}/submission_task_2.csv', index=False)
            with zipfile.ZipFile(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
                new_zip.write(f'{valid_sub_dir}/submission_task_2.csv', arcname='submission_task_2.csv')
            shutil.rmtree(valid_sub_dir)
            mlflow.log_artifact(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip')           

    if TARGET_TASK == '1':
        save_mlflow(run_id, cv, best_acc, best_th)
    elif TARGET_TASK == '2':
        save_mlflow(run_id, cv)
    mlflow.end_run()
