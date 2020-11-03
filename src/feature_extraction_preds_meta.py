import os
import pandas as pd
import numpy as np
import ast
import logging
from tqdm import tqdm

from sklearn import metrics

from utils import (
    seed_everything,
    Timer
)

from utils_feature import (
    label_encording,
    subject2level,
    factorize_dataAnswered,
    preprocess_DateOfBirth,
    preprocess_subject,
    extract_age,
    varlen_label_encording,
    question_subject_svd,
    user_subject_tfidfsvd
)

logging.basicConfig(level=logging.INFO)

INPUT_DIR = os.environ.get('INPUT_DIR')
FEATURE_DIR = os.environ.get('FEATURE_DIR')

VARLEN_LOWFREQ_TH = 0
RANDOM_STATE = os.environ.get('RANDOM_STATE')

exp_list_t1 = [
    ('task1_lgbm', 'bbb432be523149369a5f95cdfd63578c'),  # lgb, all feature, te_smooth5
    ('task1_xgb_2', '69f2b85366b34198a1ea2dc512e05f3a'),  # xgb, all feature, te_smooth5
    ('task1_cat', 'b286e8fb8f824027869bf2ec6247b8ec'),  # cat, all feature, te_smooth5
    ('task1_lgbm_fs100', 'bf0bfd4d47664b349c02dd92eaa50fc4'),  # lgb, fs100, te_smooth5
    ('task1_xgb_fs100', '2cf16a7701da409daf40cbfe78b3b7bf'),  # xgb, fs100, te_smooth5 
    ('task1_mlp_fs100', '92598df31e9f4311adccbc8e267e7c01'),  # mlp, fs100, te_smooth5
    ('task12_multitask_mlp_fs100', '38a138a4ad5e49b999ec0eca380be5dc'),  # mlp multi, fs100, te_smooth5
    ('task1_cat_fs100', '468d28a067bc479caa65e12153965df4'),  # cat depth 8, fs100, te_smooth5
    ('task1_cat_fs100', 'e4950f4d436a4d58a89c5bd90e254428'),  # cat depth 10, fs100, te_smooth5
    ('task1_lgbm', 'c4def7cc4f3f4f44971ac474d2993e92'),  # lgb, all feature, te_smooth2
]

exp_list_t2 = [
    ('task2_lgbm', '90443d723c874103a4b1c68f2830446e'),  # lgb, all feature, te_smooth5
    ('task2_xgb_2', '19a82f5124d349ee9550b871a88025d2'),  # xgb, all feature, te_smooth5
    ('task2_lgbm_fs100', '751bb6f02e5746798740f2b90f183a92'),  # lgb, fs100, te_smooth5
    ('task2_xgb_fs100', '89909e8bf54443328830085dca5f26cc'),  # xgb, fs100, te_smooth5 
    ('task2_mlp_fs100', '2f1d7ef010874723a8dd70aa49891f5d'),  # mlp, fs100, te_smooth5
    ('task12_multitask_mlp_fs100', '38a138a4ad5e49b999ec0eca380be5dc'),  # mlp multi, fs100, te_smooth5
    ('task2_cat_fs100', '5f4c75b620ef4eccbb14581589340db6'),  # cat depth 8, fs100, te_smooth5
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

prediction_lag_target = [
    'preds_t1_mean_0',
    'preds_t1_std_0',
    'preds_t2_mean_max_0',
    'preds_t2_mean_min_0',
    'preds_t2_std_sum_0',
]

meta_mul = []
for i in range(4):
    meta_mul.append(f'preds_t1_mean_mul_t2_mean_{i}')
    meta_mul.append(f'preds_t1_std_mul_t2_std_{i}')
    meta_mul.append(f'preds_t1_std_mul_t2_mean_{i}')

pred_lag_features = []
for i in prediction_lag_target:
    for j in ['prev', 'diff', 'cumsum']:
        pred_lag_features.append(f'{i}_{j}')

pred_eval_metric_features = []
for cat in ['UserId', 'QuestionId']:
    pred_eval_metric_features.append(f'pred_{cat}_t1_auc')
    pred_eval_metric_features.append(f'pred_{cat}_t2_acc')

dense_features = []
dense_features += meta_pred_features
dense_features += meta_agg_t1
dense_features += meta_agg_t2
dense_features += meta_mul


def add_preds(results, TARGET_TASK, EXP_NAME, run_id):
    oof = pd.read_csv(f'../save/{EXP_NAME}/preds_val_task{TARGET_TASK}_{run_id}.csv')
    test_preds = pd.read_csv(f'../save/{EXP_NAME}/preds_test_task{TARGET_TASK}_{run_id}.csv')
    results[f't{TARGET_TASK}_{EXP_NAME}_{run_id}'] = {}
    results[f't{TARGET_TASK}_{EXP_NAME}_{run_id}']['oof'] = oof
    results[f't{TARGET_TASK}_{EXP_NAME}_{run_id}']['test_preds'] = test_preds
    return results


def save_as_feather(feat, X_train, X_test):
    X_train[[feat]].reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_train.feather')
    X_test[[feat]].reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_test.feather')
    return


if __name__ == "__main__":

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer(f'read data'):
        data_path = f'{INPUT_DIR}/train_data/train_task_1_2.csv'
        answer_path = f'{INPUT_DIR}/metadata/answer_metadata_task_1_2.csv'
        test_data_path = f'../submission_templates/submission_task_1_2.csv'

        train = pd.read_csv(data_path)
        test = pd.read_csv(test_data_path)
        answer = pd.read_csv(answer_path)

    with t.timer(f'preprocess'):

        answer['AnswerId'] = answer['AnswerId'].fillna(-1).astype('int32')
        answer['Confidence'] = answer['Confidence'].fillna(-1).astype('int8')
        answer['SchemeOfWorkId'] = answer['SchemeOfWorkId'].fillna(-1).astype('int16')
        answer = factorize_dataAnswered(answer)

    with t.timer(f'marge dataframe'):

        df = pd.merge(train, answer, on='AnswerId', how='left')
        df_test = pd.merge(test, answer, on='AnswerId', how='left')

    with t.timer(f'read prediction'):

        results_t1 = {}
        for ename, run_id in exp_list_t1:
            results_t1 = add_preds(results_t1, TARGET_TASK='1', EXP_NAME=ename, run_id=run_id)

        results_t2 = {}
        for ename, run_id in exp_list_t2:
            results_t2 = add_preds(results_t2, TARGET_TASK='2', EXP_NAME=ename, run_id=run_id)

    with t.timer(f'prediction agg'):

        for ename, run_id in exp_list_t1:
            df[f't1_{ename}_{run_id}'] = results_t1[f't1_{ename}_{run_id}']['oof']
            df_test[f't1_{ename}_{run_id}'] = results_t1[f't1_{ename}_{run_id}']['test_preds']

        rdf_oof = pd.DataFrame()
        rdf_test = pd.DataFrame()
        for ename, run_id in exp_list_t1:
            rdf_oof = pd.concat([rdf_oof, results_t1[f't1_{ename}_{run_id}']['oof']], axis=1)
            rdf_test = pd.concat([rdf_test, results_t1[f't1_{ename}_{run_id}']['test_preds']], axis=1)

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
            df_test = pd.concat([df_test, pd.DataFrame(tests[:, :, eidx]).add_prefix(f't2_{ename}_{run_id}_')], axis=1)

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

    with t.timer(f'prediction mul'):
        for i in range(4):
            df[f'preds_t1_mean_mul_t2_mean_{i}'] = df['preds_t1_mean_0'].mul(df[f'preds_t2_mean_{i}'])
            df[f'preds_t1_std_mul_t2_std_{i}'] = df['preds_t1_std_0'].mul(df[f'preds_t2_std_{i}'])
            df[f'preds_t1_std_mul_t2_mean_{i}'] = df['preds_t1_std_0'].mul(df[f'preds_t2_mean_{i}'])
            df_test[f'preds_t1_mean_mul_t2_mean_{i}'] = df_test['preds_t1_mean_0'].mul(df_test[f'preds_t2_mean_{i}'])
            df_test[f'preds_t1_std_mul_t2_std_{i}'] = df_test['preds_t1_std_0'].mul(df_test[f'preds_t2_std_{i}'])
            df_test[f'preds_t1_std_mul_t2_mean_{i}'] = df_test['preds_t1_std_0'].mul(df_test[f'preds_t2_mean_{i}'])       


    with t.timer('save dense features'):
        for feat in dense_features:
            logging.info(f'save {feat}')
            save_as_feather(feat, df, df_test)