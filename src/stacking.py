import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

import os
import zipfile
import shutil

import datetime
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from sklearn import metrics
from sklearn.linear_model import Ridge, RidgeClassifier
from scipy.special import softmax
from sklearn.model_selection import KFold

import cloudpickle

INPUT_DIR = '../data'
SUB_DIR='../submission'
FOLD_DIR = '../folds'
FOLD_NAME = 'mskf_user'
FOLD_NUM = 5
RANDOM_STATE = 46

exp_list_t1 = [
    ('task1_lgbm', 'bbb432be523149369a5f95cdfd63578c'), # lgb, all feature, te_smooth5
    ('task1_xgb_2', '69f2b85366b34198a1ea2dc512e05f3a'), # xgb, all feature, te_smooth5
    ('task1_cat', 'b286e8fb8f824027869bf2ec6247b8ec'), # cat, all feature, te_smooth5
    ('task1_lgbm_fs100', 'bf0bfd4d47664b349c02dd92eaa50fc4'), # lgb, fs100, te_smooth5
    ('task1_xgb_fs100', '2cf16a7701da409daf40cbfe78b3b7bf'), # xgb, fs100, te_smooth5 
    ('task1_mlp_fs100', '92598df31e9f4311adccbc8e267e7c01'), # mlp, fs100, te_smooth5
    ('task12_multitask_mlp_fs100', '38a138a4ad5e49b999ec0eca380be5dc'), # mlp multi, fs100, te_smooth5
    ('task1_cat_fs100', '468d28a067bc479caa65e12153965df4'), # cat depth 8, fs100, te_smooth5
    ('task1_cat_fs100', 'e4950f4d436a4d58a89c5bd90e254428'), # cat depth 10, fs100, te_smooth5
    ('task1_lgbm', 'c4def7cc4f3f4f44971ac474d2993e92'), # lgb, all feature, te_smooth2
#---
    ('task1_xgb_fs50_meta', '0da188f3e0014876919457befc0b67e9'), 
    ('task1_xgb_fs100_meta_2', '915471c8af044216ae069ed8568f7f2a'),
    ('task1_cat_fs50_meta', 'bfeb141ae68044ebafadd0a4e286e6e3'),
    ('task12_multitask_mlp_fs50_meta', '629ddeed745148298c6148d30d7a238f'), 
]

exp_list_t2 = [
    ('task2_lgbm', '90443d723c874103a4b1c68f2830446e'), # lgb, all feature, te_smooth5
    ('task2_xgb_2', '19a82f5124d349ee9550b871a88025d2'), # xgb, all feature, te_smooth5
    ('task2_lgbm_fs100', '751bb6f02e5746798740f2b90f183a92'), # lgb, fs100, te_smooth5
    ('task2_xgb_fs100', '89909e8bf54443328830085dca5f26cc'), # xgb, fs100, te_smooth5 
    ('task2_mlp_fs100', '2f1d7ef010874723a8dd70aa49891f5d'), # mlp, fs100, te_smooth5
    ('task12_multitask_mlp_fs100', '38a138a4ad5e49b999ec0eca380be5dc'), # mlp multi, fs100, te_smooth5
    ('task2_cat_fs100', '5f4c75b620ef4eccbb14581589340db6'), # cat depth 8, fs100, te_smooth5
# ---
    ('task2_xgb_fs50_meta', 'ad5e1413cdfa45198ffd3a8d20f17886'), 
    ('task2_xgb_fs100_meta', 'a704720e1cfa42668da970f50c65af2e'),
    ('task12_multitask_mlp_fs50_meta', '629ddeed745148298c6148d30d7a238f'), 
]

meta_pred_features = []
for ename, run_id in exp_list_t1:
    meta_pred_features.append(f't1_{ename}_{run_id}')
for ename, run_id in exp_list_t2:
    for idx in range(4):
        meta_pred_features.append(f't2_{ename}_{run_id}_{idx}')

meta_agg_t1 = [
    'preds_t1_mean_0',
    'preds_t1_std_0',
    'preds_t1_max_0',
    'preds_t1_min_0',
    'preds_t1_diff_0',
]

meta_agg_t2 = [
    'preds_t2_mean_0',
    'preds_t2_mean_1',
    'preds_t2_mean_2',
    'preds_t2_mean_3',
    'preds_t2_mean_max_0',
    'preds_t2_mean_min_0',
    'preds_t2_std_0',
    'preds_t2_std_1',
    'preds_t2_std_2',
    'preds_t2_std_3',
    'preds_t2_std_sum_0',
    'preds_t2_max_0',
    'preds_t2_max_1',
    'preds_t2_max_2',
    'preds_t2_max_3',
    'preds_t2_min_0',
    'preds_t2_min_1',
    'preds_t2_min_2',
    'preds_t2_min_3',
]

meta_mul = []
for i in range(4):
    meta_mul.append(f'preds_t1_mean_mul_t2_mean_{i}')
    meta_mul.append(f'preds_t1_std_mul_t2_std_{i}')
    meta_mul.append(f'preds_t1_std_mul_t2_mean_{i}')

dense_features = []
dense_features += meta_pred_features
dense_features += meta_agg_t1
dense_features += meta_agg_t2
dense_features += meta_mul

def load_from_pkl(load_path):
    frb = open(load_path , 'rb')
    obj = cloudpickle.loads(frb.read())
    return obj


def save_as_pkl(obj, save_path):
    fwb = open(save_path, 'wb')
    fwb.write( cloudpickle.dumps(obj) )
    return

def add_preds(results, TARGET_TASK, EXP_NAME, run_id):
    oof = pd.read_csv(f'../save/{EXP_NAME}/preds_val_task{TARGET_TASK}_{run_id}.csv')
    test_preds = pd.read_csv(f'../save/{EXP_NAME}/preds_test_task{TARGET_TASK}_{run_id}.csv')
    results[f't{TARGET_TASK}_{EXP_NAME}_{run_id}'] = {}
    results[f't{TARGET_TASK}_{EXP_NAME}_{run_id}']['oof'] = oof
    results[f't{TARGET_TASK}_{EXP_NAME}_{run_id}']['test_preds'] = test_preds
    return results

def make_submission_t1(preds_test, auc, best_acc, best_th, EXP_NAME):
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

    sub_name = f'submission_task1__auc{auc}__acc{best_acc}__th{best_th}'
    valid_sub_dir = f'{SUB_DIR}/{EXP_NAME}/{sub_name}'
    if not os.path.exists(valid_sub_dir):
        os.mkdir(valid_sub_dir)

    sub.to_csv(f'{valid_sub_dir}/submission_task_1.csv', index=False)
    with zipfile.ZipFile(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        new_zip.write(f'{valid_sub_dir}/submission_task_1.csv', arcname='submission_task_1.csv')
    shutil.rmtree(valid_sub_dir)

    submission_path = f'../submission/{EXP_NAME}/submission_task1__auc{auc}__acc{best_acc}__th{best_th}.csv'
    sub.to_csv(submission_path, index=False)
    return

def make_submission_t2(preds_test, multi_acc, EXP_NAME):
    preds_test_label = np.argmax(preds_test, axis=1) + 1

    test_data_path = f'../submission_templates/submission_task_1_2.csv'
    sub = pd.read_csv(test_data_path)
    sub['AnswerValue'] = preds_test_label

    if not os.path.exists(f'{SUB_DIR}/{EXP_NAME}'):
        os.mkdir(f'{SUB_DIR}/{EXP_NAME}')

    sub_name = f'submission_task2__acc{multi_acc}'
    valid_sub_dir = f'{SUB_DIR}/{EXP_NAME}/{sub_name}'
    if not os.path.exists(valid_sub_dir):
        os.mkdir(valid_sub_dir)

    sub.to_csv(f'{valid_sub_dir}/submission_task_2.csv', index=False)
    with zipfile.ZipFile(f'{SUB_DIR}/{EXP_NAME}/{sub_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        new_zip.write(f'{valid_sub_dir}/submission_task_2.csv', arcname='submission_task_2.csv')
    shutil.rmtree(valid_sub_dir)
    return

def predict_cv_t1(model, train_x, train_y, test_x, folds):

    preds = []
    preds_test = []
    va_idxes = []
    
    for fold_idx in tqdm(range(FOLD_NUM)):

        tr_idx = folds[folds.kfold != fold_idx].index.tolist()
        va_idx = folds[folds.kfold == fold_idx].index.tolist()

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y[tr_idx], train_y[va_idx]

        model.fit(tr_x, tr_y)

        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)

        va_idxes.append(va_idx)

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test

def predict_cv_t2(model, train_x, train_y, test_x, folds):

    preds = []
    preds_test = []
    va_idxes = []

    for fold_idx in tqdm(range(FOLD_NUM)):

        tr_idx = folds[folds.kfold != fold_idx].index.tolist()
        va_idx = folds[folds.kfold == fold_idx].index.tolist()
    
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y[tr_idx], train_y[va_idx]

        model.fit(tr_x, tr_y)

        pred = model.decision_function(va_x)
        pred = softmax(pred, axis=1)
        preds.append(pred)
        pred_test = model.decision_function(test_x)
        pred_test = softmax(pred_test, axis=1)
        preds_test.append(pred_test)

        va_idxes.append(va_idx)

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test

if __name__ == "__main__":

    folds = pd.read_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv')

    data_path = f'{INPUT_DIR}/train_data/train_task_1_2.csv'
    answer_path = f'{INPUT_DIR}/metadata/answer_metadata_task_1_2.csv'
    test_data_path = f'../submission_templates/submission_task_1_2.csv'

    train = pd.read_csv(data_path)
    test = pd.read_csv(test_data_path)
    answer = pd.read_csv(answer_path)
    answer['AnswerId'] = answer['AnswerId'].fillna(-1).astype('int32')
    answer['Confidence'] = answer['Confidence'].fillna(-1).astype('int8')
    answer['SchemeOfWorkId'] = answer['SchemeOfWorkId'].fillna(-1).astype('int16')

    y_t1 = train['IsCorrect'].values
    y_t2 = train['AnswerValue'].values

    df = pd.merge(train, answer, on='AnswerId', how='left')
    df_test = pd.merge(test, answer, on='AnswerId', how='left')

    results_t1 = {}
    for ename, run_id in exp_list_t1:
        results_t1 = add_preds(results_t1, TARGET_TASK='1', EXP_NAME=ename, run_id=run_id)

    results_t2 = {}
    for ename, run_id in exp_list_t2:
        results_t2 = add_preds(results_t2, TARGET_TASK='2', EXP_NAME=ename, run_id=run_id)

    rdf_oof = pd.DataFrame()
    rdf_test = pd.DataFrame()
    for ename, run_id in exp_list_t1:
        rdf_oof = pd.concat([rdf_oof, results_t1[f't1_{ename}_{run_id}']['oof']], axis=1)
        rdf_test = pd.concat([rdf_test, results_t1[f't1_{ename}_{run_id}']['test_preds']], axis=1)

    cols = [ename for ename, _ in exp_list_t1]
    rdf_oof.columns = cols
    rdf_test.columns = cols

    for ename, run_id in exp_list_t1:
        df[f't1_{ename}_{run_id}'] = results_t1[f't1_{ename}_{run_id}']['oof']
        df_test[f't1_{ename}_{run_id}'] = results_t1[f't1_{ename}_{run_id}']['test_preds']

    meta_oofs_t1 = pd.DataFrame()
    meta_oofs_t1 = pd.concat([meta_oofs_t1, pd.DataFrame(rdf_oof.mean(axis=1)).add_prefix('preds_t1_mean_')], axis=1)
    meta_oofs_t1 = pd.concat([meta_oofs_t1, pd.DataFrame(rdf_oof.std(axis=1)).add_prefix('preds_t1_std_')], axis=1)
    meta_oofs_t1 = pd.concat([meta_oofs_t1, pd.DataFrame(rdf_oof.max(axis=1)).add_prefix('preds_t1_max_')], axis=1)
    meta_oofs_t1 = pd.concat([meta_oofs_t1, pd.DataFrame(rdf_oof.min(axis=1)).add_prefix('preds_t1_min_')], axis=1)
    meta_oofs_t1 = pd.concat([meta_oofs_t1, pd.DataFrame(rdf_oof.max(axis=1) - rdf_oof.min(axis=1)).add_prefix('preds_t1_diff_')], axis=1)

    meta_tests_t1 = pd.DataFrame()
    meta_tests_t1 = pd.concat([meta_tests_t1, pd.DataFrame(rdf_test.mean(axis=1)).add_prefix('preds_t1_mean_')], axis=1)
    meta_tests_t1 = pd.concat([meta_tests_t1, pd.DataFrame(rdf_test.std(axis=1)).add_prefix('preds_t1_std_')], axis=1)
    meta_tests_t1 = pd.concat([meta_tests_t1, pd.DataFrame(rdf_test.max(axis=1)).add_prefix('preds_t1_max_')], axis=1)
    meta_tests_t1 = pd.concat([meta_tests_t1, pd.DataFrame(rdf_test.min(axis=1)).add_prefix('preds_t1_min_')], axis=1)
    meta_tests_t1 = pd.concat([meta_tests_t1, pd.DataFrame(rdf_test.max(axis=1) - rdf_test.min(axis=1)).add_prefix('preds_t1_diff_')], axis=1)

    df = pd.concat([df, meta_oofs_t1], axis=1)
    df_test = pd.concat([df_test, meta_tests_t1], axis=1)

    oofs = np.zeros((len(rdf_oof), 4, len(exp_list_t2)))
    tests = np.zeros((len(rdf_test), 4, len(exp_list_t2)))
    for eidx, (ename, run_id) in enumerate(exp_list_t2):
        oofs[:, :, eidx] = results_t2[f't2_{ename}_{run_id}']['oof'].values
        tests[:, :, eidx] = results_t2[f't2_{ename}_{run_id}']['test_preds'].values

    for eidx, (ename, run_id) in enumerate(exp_list_t2):
        df = pd.concat([df, pd.DataFrame(oofs[:, :, eidx]).add_prefix(f't2_{ename}_{run_id}_')], axis=1)
        df_test = pd.concat([df_test,pd.DataFrame(tests[:, :, eidx]).add_prefix(f't2_{ename}_{run_id}_')], axis=1)
        
    meta_oofs_t2 = pd.DataFrame()
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.mean(axis=2)).add_prefix('preds_t2_mean_')], axis=1)
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.mean(axis=2).max(axis=1)).add_prefix('preds_t2_mean_max_')], axis=1)
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.mean(axis=2).min(axis=1)).add_prefix('preds_t2_mean_min_')], axis=1)
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.std(axis=2)).add_prefix('preds_t2_std_')], axis=1)
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.std(axis=2).sum(axis=1)).add_prefix('preds_t2_std_sum_')], axis=1)
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.max(axis=2)).add_prefix('preds_t2_max_')], axis=1)
    meta_oofs_t2 = pd.concat([meta_oofs_t2, pd.DataFrame(oofs.min(axis=2)).add_prefix('preds_t2_min_')], axis=1)

    meta_tests_t2 = pd.DataFrame()
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.mean(axis=2)).add_prefix('preds_t2_mean_')], axis=1)
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.mean(axis=2).max(axis=1)).add_prefix('preds_t2_mean_max_')], axis=1)
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.mean(axis=2).min(axis=1)).add_prefix('preds_t2_mean_min_')], axis=1)
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.std(axis=2)).add_prefix('preds_t2_std_')], axis=1)
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.std(axis=2).sum(axis=1)).add_prefix('preds_t2_std_sum_')], axis=1)
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.max(axis=2)).add_prefix('preds_t2_max_')], axis=1)
    meta_tests_t2 = pd.concat([meta_tests_t2, pd.DataFrame(tests.min(axis=2)).add_prefix('preds_t2_min_')], axis=1)

    df = pd.concat([df, meta_oofs_t2], axis=1)
    df_test = pd.concat([df_test, meta_tests_t2], axis=1)

    for i in range(4):
        df[f'preds_t1_mean_mul_t2_mean_{i}'] = df['preds_t1_mean_0'].mul(df[f'preds_t2_mean_{i}'])
        df[f'preds_t1_std_mul_t2_std_{i}'] = df['preds_t1_std_0'].mul(df[f'preds_t2_std_{i}'])
        df[f'preds_t1_std_mul_t2_mean_{i}'] = df['preds_t1_std_0'].mul(df[f'preds_t2_mean_{i}'])
        df_test[f'preds_t1_mean_mul_t2_mean_{i}'] = df_test['preds_t1_mean_0'].mul(df_test[f'preds_t2_mean_{i}'])
        df_test[f'preds_t1_std_mul_t2_std_{i}'] = df_test['preds_t1_std_0'].mul(df_test[f'preds_t2_std_{i}'])
    df_test[f'preds_t1_std_mul_t2_mean_{i}'] = df_test['preds_t1_std_0'].mul(df_test[f'preds_t2_mean_{i}'])

    ridge_alpha = 1.0
    smodel = Ridge(alpha=ridge_alpha)
    pred_oof, preds_test = predict_cv_t1(
        smodel,
        df[dense_features],
        y_t1,
        df_test[dense_features],
        folds,
    )
    auc = metrics.roc_auc_score(y_t1, pred_oof)

    rows = []
    for th in tqdm(range(490, 510, 1)):
        th = th * 0.001
        preds_th = []
        for i in pred_oof:
            if i > th:
                preds_th.append(1)
            else:
                preds_th.append(0)
        acc = metrics.accuracy_score(y_t1, preds_th)
        rows.append([th, acc])
    acc_th = pd.DataFrame(rows, columns=['th', 'acc'])
    scores = acc_th.sort_values('acc', ascending=False).head(1).values[0]
    best_th, best_acc = scores[0], scores[1]

    exp_name = '__'.join([ename for ename, _ in exp_list_t1])
    exp_name = f'stacking_{ename}'
    make_submission_t1(
        preds_test, auc=auc, best_acc=best_acc, best_th=best_th,
        EXP_NAME=exp_name)

    ridge_alpha = 1.0
    smodel = RidgeClassifier(alpha=ridge_alpha)
    pred_oof, preds_test = predict_cv_t2(
        smodel,
        df[dense_features],
        y_t2,
        df_test[dense_features],
        folds,
    )

    pred_oof_label = np.argmax(pred_oof, axis=1) + 1
    multi_acc = metrics.accuracy_score(y_t2, pred_oof_label)

    exp_name = '__'.join([ename for ename, _ in exp_list_t2])
    exp_name = f'stacking_{ename}'
    make_submission_t2(
        preds_test, multi_acc=multi_acc,
        EXP_NAME=exp_name)
