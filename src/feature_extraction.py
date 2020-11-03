import os
import pandas as pd
import numpy as np
import ast
import logging
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xfeat import CountEncoder

from utils import (
    seed_everything,
    Timer,
    reduce_mem_usage
)

from utils_feature import (
    target_encode_cudf_v3,
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

subject_id_list = [
    3,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    126,
    128,
    129,
    130,
    131,
    137,
    139,
    140,
    141,
    142,
    144,
    146,
    149,
    151,
    152,
    153,
    154,
    156,
    157,
    158,
    159,
    160,
    163,
    164,
    165,
    166,
    167,
    168,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    195,
    196,
    197,
    198,
    199,
    200,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    298,
    313,
    315,
    317,
    331,
    332,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    341,
    342,
    343,
    344,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    361,
    365,
    366,
    367,
    369,
    370,
    371,
    372,
    374,
    375,
    376,
    377,
    388,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    416,
    417,
    418,
    430,
    431,
    432,
    434,
    435,
    436,
    437,
    439,
    441,
    442,
    446,
    447,
    448,
    451,
    453,
    462,
    474,
    480,
    487,
    539,
    540,
    649,
    654,
    655,
    656,
    657,
    692,
    698,
    700,
    1059,
    1076,
    1077,
    1078,
    1079,
    1080,
    1081,
    1082,
    1156,
    1157,
    1158,
    1159,
    1160,
    1161,
    1162,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1171,
    1172,
    1173,
    1174,
    1175,
    1176,
    1177,
    1178,
    1179,
    1180,
    1181,
    1182,
    1183,
    1184,
    1185,
    1186,
    1187,
    1188,
    1189,
    1200,
    1201,
    1202,
    1203,
    1204,
    1207,
    1208,
    1209,
    1210,
    1211,
    1212,
    1213,
    1214,
    1215,
    1216,
    1217,
    1218,
    1219,
    1263,
    1264,
    1265,
    1266,
    1636,
    1642,
    1647,
    1648,
    1649,
    1650,
    1651,
    1675,
    1676,
    1750,
    1975,
    1976,
    1977,
    1980,
    1982,
    1983,
    1985,
    1987,
    1988
]

level_cnum_list = [
    '0_8',
    '1_16',
    '3_0',
    '2_3',
    '2_4',
    '2_5',
    '2_10',
    '2_7',
    '1_14',
    '2_11',
    '2_6',
    '2_9',
    '2_8',
    '1_5',
    '1_12',
    '2_1',
    '2_13',
    '1_1',
    '1_4',
    '2_2',
    '2_14',
    '1_0',
    '2_0',
    '0_1'
]

subject_meta_cols = [
    'num',
    'max_level',
    'sum_level',
    'max_cnum',
    'sum_cnum',
]

subject_features = [f'subj_{f}' for f in subject_id_list]
level_cnum_features = [f'subj_{f}' for f in level_cnum_list]
subject_meta_features = [f'subj_{f}' for f in subject_meta_cols]

user_lag_num_features = [
    'DateAnswered_dt_diff',
    'DateAnswered_dt_diff_cumsum',
    'DateAnswered_dt_diff_shift',
    'DateAnswered_dt_diff_cumsum_shift',
    'answer_num',
    'answer_num_norm',
    'quiz_answer_num',
    'quiz_answer_num_norm',
    'quiz_unique_num',
    'subj_unique_num',
    'group_unique_num',
    'subjcat_unique_num',
]
user_lag_cat_features = [
    'answer_num_div5',
    'quiz_answer_num_div5',
    'change_subjcat',
    'answered_subjcat',
    'prev_question',
    'prev_subjcat',
]
user_lag_multicat_features = [
    'prev10_question',
    'prev10_subjcat',
]
user_lag_features = user_lag_num_features + user_lag_cat_features + user_lag_multicat_features

answer_date_features = [
    'DateAnswered_weekday',
    'DateAnswered_hour',
    'DateAnswered_day',
    'DateAnswered_wom'
]
count_encording_cat = [
    'QuestionId',
    'UserId',
    'Gender',
    'PremiumPupil',
    'Confidence',
    'GroupId',
    'QuizId',
    'SchemeOfWorkId',
    'age_years',
    'DateAnswered_weekday',
    'DateAnswered_hour',
    'DateAnswered_day',
    'DateAnswered_wom',
    'answer_num_div5',
    'quiz_answer_num_div5',
    'change_subjcat',
    'answered_subjcat',
    'prev_question',
    'prev_subjcat',
    'SubjectId_cat',
    'pri_to_high_stu',
    ['UserId', 'DateAnswered_weekday'],
    ['UserId', 'DateAnswered_hour'],
    ['UserId', 'DateAnswered_day'],
    ['UserId', 'DateAnswered_wom'],
    ['UserId', 'DateAnswered_weekday', 'DateAnswered_hour'],
    ['UserId', 'DateAnswered_weekday', 'DateAnswered_wom'],
    ['UserId', 'Confidence'],
    ['UserId', 'SchemeOfWorkId'],
    ['UserId', 'GroupId'],
    ['UserId', 'QuizId'],
    ['UserId', 'SubjectId_cat'],
    ['UserId', 'answer_num_div5'],
    ['UserId', 'quiz_answer_num_div5'],
    ['UserId', 'change_subjcat'],
    ['UserId', 'answered_subjcat'],
    ['UserId', 'age_years'],
    ['UserId', 'age_years', 'Confidence'],
    ['QuestionId', 'Confidence'],
    ['QuestionId', 'SchemeOfWorkId'],
    ['QuestionId', 'age_years'],
    ['QuestionId', 'Gender'],
    ['QuestionId', 'answer_num_div5'],
    ['QuestionId', 'quiz_answer_num_div5'],
    ['QuestionId', 'change_subjcat'],
    ['QuestionId', 'answered_subjcat'],
    ['QuestionId', 'PremiumPupil'],
    ['QuestionId', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'age_years', 'Gender'],
    ['QuestionId', 'age_years', 'PremiumPupil'],
    ['QuestionId', 'age_years', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'age_years', 'Gender'],
    ['QuestionId', 'Confidence', 'age_years', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'age_years', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'prev_question'],
    ['QuestionId', 'DateOfBirth_NaN'],
    ['QuestionId', 'pri_to_high_stu'],
    ['SubjectId_cat', 'Confidence'],
    ['SubjectId_cat', 'SchemeOfWorkId'],
    ['SubjectId_cat', 'age_years'],
    ['SubjectId_cat', 'Gender'],
    ['SubjectId_cat', 'answer_num_div5'],
    ['SubjectId_cat', 'quiz_answer_num_div5'],
    ['SubjectId_cat', 'change_subjcat'],
    ['SubjectId_cat', 'answered_subjcat'],
    ['QuestionId', 'GroupId'],
    ['QuestionId', 'QuizId'],
    ['SchemeOfWorkId', 'Confidence'],
    ['SchemeOfWorkId', 'GroupId'],
    ['SchemeOfWorkId', 'QuizId'],
    ['SchemeOfWorkId', 'age_years'],
    ['SchemeOfWorkId', 'Gender'],
    ['SchemeOfWorkId', 'answer_num_div5'],
    ['SchemeOfWorkId', 'quiz_answer_num_div5'],
    ['SchemeOfWorkId', 'change_subjcat'],
    ['SchemeOfWorkId', 'answered_subjcat'],
    ['SchemeOfWorkId', 'PremiumPupil'],
    ['SchemeOfWorkId', 'Gender', 'PremiumPupil'],
    ['SchemeOfWorkId', 'age_years', 'Gender'],
    ['SchemeOfWorkId', 'age_years', 'PremiumPupil'],
    ['SchemeOfWorkId', 'age_years', 'Gender', 'PremiumPupil'],
]
count_encording_features = []
for col in count_encording_cat:
    if not isinstance(col, list):
        col = [col]
    name = "_".join(col)
    count_encording_features.append(f'{name}_ce')

te_smooth_factor = 5
# te_smooth_factor = 2
target_encording_cat = [
    'QuestionId',
    'UserId',
    'Gender',
    'PremiumPupil',
    'Confidence',
    'GroupId',
    'QuizId',
    'SchemeOfWorkId',
    'age_years',
    'DateAnswered_weekday',
    'DateAnswered_hour',
    'DateAnswered_day',
    'DateAnswered_wom',
    'answer_num_div5',
    'quiz_answer_num_div5',
    'change_subjcat',
    'answered_subjcat',
    'prev_question',
    'prev_subjcat',
    'SubjectId_cat',
    'DateOfBirth_NaN',
    'pri_to_high_stu',
    ['DateAnswered_day', 'DateAnswered_hour'],
    ['DateAnswered_weekday', 'DateAnswered_hour'],
    ['DateAnswered_weekday', 'DateAnswered_wom'],
    ['UserId', 'DateAnswered_weekday'],
    ['UserId', 'DateAnswered_hour'],
    ['UserId', 'DateAnswered_day'],
    ['UserId', 'DateAnswered_wom'],
    ['UserId', 'DateAnswered_weekday', 'DateAnswered_hour'],
    ['UserId', 'DateAnswered_weekday', 'DateAnswered_wom'],
    ['UserId', 'Confidence'],
    ['UserId', 'SchemeOfWorkId'],
    ['UserId', 'GroupId'],
    ['UserId', 'QuizId'],
    ['UserId', 'SubjectId_cat'],
    ['UserId', 'answer_num_div5'],
    ['UserId', 'quiz_answer_num_div5'],
    ['UserId', 'change_subjcat'],
    ['UserId', 'answered_subjcat'],
    ['UserId', 'age_years'],
    ['UserId', 'age_years', 'Confidence'],
    ['QuestionId', 'Confidence'],
    ['QuestionId', 'SchemeOfWorkId'],
    ['QuestionId', 'age_years'],
    ['QuestionId', 'Gender'],
    ['QuestionId', 'PremiumPupil'],
    ['QuestionId', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'age_years', 'Gender'],
    ['QuestionId', 'age_years', 'PremiumPupil'],
    ['QuestionId', 'age_years', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'age_years', 'Gender'],
    ['QuestionId', 'Confidence', 'age_years', 'PremiumPupil'],
    ['QuestionId', 'Confidence', 'age_years', 'Gender', 'PremiumPupil'],
    ['QuestionId', 'answer_num_div5'],
    ['QuestionId', 'quiz_answer_num_div5'],
    ['QuestionId', 'GroupId'],
    ['QuestionId', 'QuizId'],
    ['QuestionId', 'change_subjcat'],
    ['QuestionId', 'answered_subjcat'],
    ['QuestionId', 'prev_question'],
    ['QuestionId', 'DateOfBirth_NaN'],
    ['QuestionId', 'pri_to_high_stu'],
    ['SubjectId_cat', 'Confidence'],
    ['SubjectId_cat', 'SchemeOfWorkId'],
    ['SubjectId_cat', 'age_years'],
    ['SubjectId_cat', 'Gender'],
    ['SubjectId_cat', 'answer_num_div5'],
    ['SubjectId_cat', 'quiz_answer_num_div5'],
    ['SubjectId_cat', 'change_subjcat'],
    ['SubjectId_cat', 'answered_subjcat'],
    ['SchemeOfWorkId', 'Confidence'],
    ['SchemeOfWorkId', 'GroupId'],
    ['SchemeOfWorkId', 'QuizId'],
    ['SchemeOfWorkId', 'age_years'],
    ['SchemeOfWorkId', 'Gender'],
    ['SchemeOfWorkId', 'answer_num_div5'],
    ['SchemeOfWorkId', 'quiz_answer_num_div5'],
    ['SchemeOfWorkId', 'change_subjcat'],
    ['SchemeOfWorkId', 'answered_subjcat'],
    ['SchemeOfWorkId', 'PremiumPupil'],
    ['SchemeOfWorkId', 'Gender', 'PremiumPupil'],
    ['SchemeOfWorkId', 'age_years', 'Gender'],
    ['SchemeOfWorkId', 'age_years', 'PremiumPupil'],
    ['SchemeOfWorkId', 'age_years', 'Gender', 'PremiumPupil'],
]
target_encording_features = []
for tar in ['IsCorrect']:
    for col in target_encording_cat:
        if not isinstance(col, list):
            col = [col]
        name = "_".join(col)
        target_encording_features.append(f'TE_s{te_smooth_factor}_{name}_{tar}')

target_encording_ansval_features = []
for tar in ['AnswerValue_1', 'AnswerValue_2', 'AnswerValue_3', 'AnswerValue_4']:
    for col in target_encording_cat:
        if not isinstance(col, list):
            col = [col]
        name = "_".join(col)
        target_encording_ansval_features.append(f'TE_s{te_smooth_factor}_{name}_{tar}')

target_encording_subj_features = []
for tar in ['IsCorrect']:
    for col in subject_features:
        if not isinstance(col, list):
            col = [col]
        name = "_".join(col)
        target_encording_subj_features.append(f'TE_s{te_smooth_factor}_{name}_{tar}')

subj_conbi_cols = [
    'UserId',
    'age_years',
    'answered_subjcat',
    'SchemeOfWorkId',
    'Confidence',
]
target_encording_subj_conbi_cat = []
for col in subject_features:
    for col2 in subj_conbi_cols:
        target_encording_subj_conbi_cat.append([col, col2])

target_encording_subj_conbi_features = []
for tar in ['IsCorrect']:
    for col in target_encording_subj_conbi_cat:
        if not isinstance(col, list):
            col = [col]
        name = "_".join(col)
        target_encording_subj_conbi_features.append(f'TE_s{te_smooth_factor}_{name}_{tar}')

target_encording_subj_agg_features = []
for agg_func in ['sum', 'mean', 'std', 'max', 'min']:
    target_encording_subj_agg_features.append(f'TE_s{te_smooth_factor}_subj_agg_{agg_func}_IsCorrect')
for agg_func in ['sum', 'mean', 'std', 'max', 'min']:
    for conbi_col in subj_conbi_cols:
        target_encording_subj_agg_features.append(f'TE_s{te_smooth_factor}_subj_{conbi_col}_agg_{agg_func}_IsCorrect')

svd_n_components = 5
svd_features = []
svd_features += [f'ques_subj_svd_{i}' for i in range(svd_n_components)]
svd_features += [f'user_subj_svd_{i}' for i in range(svd_n_components)]


dense_features = [
    'age_days'
]
dense_features += count_encording_features
dense_features += target_encording_features
dense_features += target_encording_ansval_features
dense_features += subject_meta_features
dense_features += target_encording_subj_features
dense_features += user_lag_num_features
dense_features += target_encording_subj_conbi_features
dense_features += target_encording_subj_agg_features
dense_features += svd_features

sparse_features = [
    'QuestionId',
    'UserId',
    'Gender',
    'PremiumPupil',
    'Confidence',
    'GroupId',
    'QuizId',
    'SchemeOfWorkId',
    'age_years',
    'SubjectId_cat',
    'DateOfBirth_NaN',
    'pri_to_high_stu',
]
sparse_features += answer_date_features
sparse_features += subject_features
sparse_features += level_cnum_features
sparse_features += user_lag_cat_features

varlen_sparse_features = [
    'SubjectId',
    'SubjectId_level'
]
varlen_sparse_features += user_lag_multicat_features


def save_as_feather(feat, X_train, X_test):
    X_train[[feat]].reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_train.feather')
    X_test[[feat]].reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_test.feather')
    return


def save_as_parquet(feat, X_train, X_test):
    X_train[[feat]].reset_index(drop=True).to_parquet(f'{FEATURE_DIR}/{feat}_train.parquet')
    X_test[[feat]].reset_index(drop=True).to_parquet(f'{FEATURE_DIR}/{feat}_test.parquet')
    return


def save_subject_id_feature(df, subject, train):

    subject_oht = np.zeros((len(df), len(subject_id_list)), dtype='int8')
    for i, slist in tqdm(enumerate(df['SubjectId'].values), total=len(df)):
        for sid in slist:
            j = subject_id_list.index(sid)
            subject_oht[i, j] += 1

    for i, feat in enumerate(subject_features):
        logging.info(f'save {feat}')
        pd.DataFrame(subject_oht[:, i], columns=[feat]).reset_index(drop=True).to_feather(f'{FEATURE_DIR}/{feat}_{train}.feather')
    return


def add_subject_feature(df, subject):

    sid2lcnum = {i: j for i, j in subject[['SubjectId', 'Level__SubjectId_cnum']].values}
    sid2lev = {i: j for i, j in subject[['SubjectId', 'Level']].values}
    sid2cnum = {i: j for i, j in subject[['SubjectId', 'SubjectId_cnum']].values}

    level_cnum_oht = []
    subject_meta = []
    for slist in tqdm(df['SubjectId'].values, total=len(df)):
        lsoht = np.zeros(len(level_cnum_list), dtype=int)
        for sid in slist:
            i = level_cnum_list.index(sid2lcnum[sid])
            lsoht[i] += 1
        snum = len(slist)
        levlist = [sid2lev[sid] for sid in slist]
        cnumlist = [sid2cnum[sid] for sid in slist]
        level_cnum_oht.append(lsoht)
        subject_meta.append([
            snum, max(levlist), sum(levlist), max(cnumlist), sum(cnumlist)
        ])

    level_subject_oht = pd.DataFrame(level_cnum_oht, columns=level_cnum_list).add_prefix('subj_')
    level_subject_oht = reduce_mem_usage(level_subject_oht)
    subject_meta = pd.DataFrame(subject_meta, columns=subject_meta_cols).add_prefix('subj_')
    subject_meta = reduce_mem_usage(subject_meta)

    df = pd.concat([df, level_subject_oht, subject_meta], axis=1)
    return df


if __name__ == "__main__":

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer(f'read data'):
        data_path = f'{INPUT_DIR}/train_data/train_task_1_2.csv'
        answer_path = f'{INPUT_DIR}/metadata/answer_metadata_task_1_2.csv'
        question_path = f'{INPUT_DIR}/metadata/question_metadata_task_1_2.csv'
        student_path = f'{INPUT_DIR}/metadata/student_metadata_task_1_2.csv'
        subject_path = f'{INPUT_DIR}/metadata/subject_metadata.csv'
        test_data_path = f'../submission_templates/submission_task_1_2.csv'

        train = pd.read_csv(data_path)
        test = pd.read_csv(test_data_path)
        answer = pd.read_csv(answer_path)
        student = pd.read_csv(student_path)
        question = pd.read_csv(question_path)
        question['SubjectId'] = question['SubjectId'].apply(ast.literal_eval)
        subject = pd.read_csv(subject_path)

    with t.timer(f'preprocess'):

        answer['AnswerId'] = answer['AnswerId'].fillna(-1).astype('int32')
        answer['Confidence'] = answer['Confidence'].fillna(-1).astype('int8')
        answer['SchemeOfWorkId'] = answer['SchemeOfWorkId'].fillna(-1).astype('int16')
        answer = factorize_dataAnswered(answer)

        question['SubjectId_num'] = question['SubjectId'].apply(len)
        lbe = LabelEncoder()
        lbe.fit(question['SubjectId'].astype(str).values)
        question['SubjectId_cat'] = lbe.transform(question['SubjectId'].astype(str).values)

        student['PremiumPupil'] = student['PremiumPupil'].fillna(-1).astype('int8')
        student = preprocess_DateOfBirth(student)
        student['DateOfBirth_NaN'] = student['DateOfBirth'].isnull()

        subject = preprocess_subject(subject)

    with t.timer(f'marge dataframe'):

        df = pd.merge(train, student, on='UserId', how='left')
        df = pd.merge(df, question, on='QuestionId', how='left')
        df = pd.merge(df, answer, on='AnswerId', how='left')

        df_test = pd.merge(test, student, on='UserId', how='left')
        df_test = pd.merge(df_test, question, on='QuestionId', how='left')
        df_test = pd.merge(df_test, answer, on='AnswerId', how='left')

    with t.timer(f'add svd features'):

        ques_subj_svd = question_subject_svd(question, subject_id_list, 5)
        user_subj_svd = user_subject_tfidfsvd(
            pd.concat([df[['UserId', 'SubjectId']], df_test[['UserId', 'SubjectId']]]), 5
        )

        df = pd.merge(df, ques_subj_svd, on='QuestionId', how='left')
        df_test = pd.merge(df_test, ques_subj_svd, on='QuestionId', how='left')

        df = pd.merge(df, user_subj_svd, on='UserId', how='left')
        df_test = pd.merge(df_test, user_subj_svd, on='UserId', how='left')

    with t.timer(f'user age features'):
        df = extract_age(df)
        df_test = extract_age(df_test)

    with t.timer(f'subject preprocess'):
        df = df.drop(columns=['DateOfBirth', 'DateAnswered', 'DateOfBirth_dt', 'DateAnswered_dt'])
        df_test = df_test.drop(columns=['DateOfBirth', 'DateAnswered', 'DateOfBirth_dt', 'DateAnswered_dt'])
        df = reduce_mem_usage(df)
        df_test = reduce_mem_usage(df_test)

        subject = preprocess_subject(subject)

        df = add_subject_feature(df, subject)
        df_test = add_subject_feature(df_test, subject)

        save_subject_id_feature(df, subject, 'train')
        save_subject_id_feature(df_test, subject, 'test')

    with t.timer(f'user lag features'):
        X_tmp = pd.concat([
            df[['AnswerId', 'QuestionId', 'SubjectId_cat', 'SubjectId', 'UserId', 'Confidence', 'GroupId', 'QuizId', 'SchemeOfWorkId', 'DateAnswered_dt']],
            df_test[['AnswerId', 'QuestionId', 'SubjectId_cat', 'SubjectId', 'UserId', 'Confidence', 'GroupId', 'QuizId', 'SchemeOfWorkId', 'DateAnswered_dt']]
        ]).sort_values('DateAnswered_dt')

        answer_id_list = []
        dt_diff  = []
        dt_diff_cumsum = []
        dt_diff_shift = []
        dt_diff_cumsum_shift = []
        answer_num = []
        answer_num_norm = []
        answer_num_div5 = []
        quiz_answer_num = []
        quiz_answer_num_norm = []
        quiz_answer_num_div5 = []
        quiz_unique_num = []

        subj_unique_num = []
        group_unique_num = []
        subjcat_unique_num = []
        prev_question = []
        prev_subjcat = []
        prev10_question = []
        prev10_subjcat = []
        change_subjcat = []
        answered_subjcat = []
        for user_id, user_df in tqdm(X_tmp.groupby('UserId'), total=X_tmp['UserId'].nunique()):
            user_df = user_df.reset_index(drop=True)
            user_df['answer_num'] = user_df.index
            user_df['answer_num_norm'] = user_df['answer_num'] / (user_df['answer_num'].max() + 1)
            user_df['answer_num_div5'] = user_df['answer_num'] // 5

            qidx = 1
            for quiz_id, j in user_df.groupby('QuizId'):
                diff = [0.0] + [j.item() for i, j in enumerate(j['DateAnswered_dt'].diff().astype('timedelta64[m]').values) if i != 0]
                j['DateAnswered_dt_diff'] = diff
                j['DateAnswered_dt_diff_cumsum'] = j['DateAnswered_dt_diff'].cumsum()
                j['DateAnswered_dt_diff_shift'] = j['DateAnswered_dt_diff'].shift(-1).fillna(0.0)
                j['DateAnswered_dt_diff_cumsum_shift'] = j['DateAnswered_dt_diff_cumsum'].shift(-1).fillna(0.0)

                j = j.reset_index(drop=True)
                j['quiz_answer_num'] = j.index
                j['quiz_answer_num_norm'] = j['quiz_answer_num'] / (j['quiz_answer_num'].max() + 1)
                j['quiz_answer_num_div5'] = j['quiz_answer_num'] // 5

                answer_id_list.extend(j['AnswerId'].values.tolist())
                dt_diff.extend(diff)
                dt_diff_cumsum.extend(j['DateAnswered_dt_diff_cumsum'].values.tolist())
                dt_diff_shift.extend(j['DateAnswered_dt_diff_shift'].values.tolist())
                dt_diff_cumsum_shift.extend(j['DateAnswered_dt_diff_cumsum_shift'].values.tolist())

                quiz_answer_num.extend(j['quiz_answer_num'].values.tolist())
                quiz_answer_num_norm.extend(j['quiz_answer_num_norm'].values.tolist())
                quiz_answer_num_div5.extend(j['quiz_answer_num_div5'].values.tolist())

                quiz_unique_num.extend([qidx] * len(j))

            u_subj_unique_num = []
            u_group_unique_num = []
            u_subjcat_unique_num = []
            u_prev_question = []
            u_prev_subjcat = []
            u_prev10_question = []
            u_prev10_subjcat = []
            u_change_subjcat = []
            u_answered_subjcat = []

            hist_question = [-1]
            hist_subj_cat = [-1]
            hist_subj = set()
            hist_group = set()
            for x in user_df[['SubjectId', 'SubjectId_cat', 'GroupId', 'QuestionId']].values:
                hist_subj = hist_subj | set(x[0])
                hist_group.add(x[2])
                hist_subj_cat.append(x[1])
                hist_question.append(x[3])

                u_subj_unique_num.append(len(hist_subj))
                u_group_unique_num.append(len(hist_group))
                u_subjcat_unique_num.append(len(set(hist_subj_cat)))
                u_prev_question.append(hist_question[-2])
                u_prev_subjcat.append(hist_subj_cat[-2])
                u_prev10_question.append(hist_question[-12:-2])
                u_prev10_subjcat.append(hist_subj_cat[-12:-2])
                u_change_subjcat.append(hist_subj_cat[-1] == hist_subj_cat[-2])
                u_answered_subjcat.append(hist_subj_cat[-1] in hist_subj_cat[:-2])

            answer_num.extend(user_df['answer_num'].values.tolist())
            answer_num_norm.extend(user_df['answer_num_norm'].values.tolist())
            answer_num_div5.extend(user_df['answer_num_div5'].values.tolist())

            subj_unique_num.extend(u_subj_unique_num)
            group_unique_num.extend(u_group_unique_num)
            subjcat_unique_num.extend(u_subjcat_unique_num)
            prev_question.extend(u_prev_question)
            prev_subjcat.extend(u_prev_subjcat)
            prev10_question.extend(u_prev10_question)
            prev10_subjcat.extend(u_prev10_subjcat)
            change_subjcat.extend(u_change_subjcat)
            answered_subjcat.extend(u_answered_subjcat)

        X = np.array([
            answer_id_list,
            dt_diff,
            dt_diff_cumsum,
            dt_diff_shift,
            dt_diff_cumsum_shift,
            answer_num,
            answer_num_norm,
            quiz_answer_num,
            quiz_answer_num_norm,
            quiz_unique_num,
            subj_unique_num,
            group_unique_num,
            subjcat_unique_num,
            answer_num_div5,
            quiz_answer_num_div5,
            change_subjcat,
            answered_subjcat,
            prev_question,
            prev_subjcat,
            prev10_question,
            prev10_subjcat,
        ]).T

        X = pd.DataFrame(X, columns=['AnswerId'] + user_lag_features)
        df = pd.merge(df, X[['AnswerId'] + user_lag_features], on='AnswerId', how='left')
        df_test = pd.merge(df_test, X[['AnswerId'] + user_lag_features], on='AnswerId', how='left')

    with t.timer(f'varlen sparse feature preprocess'):
        df, df_test = subject2level(subject, df, df_test)

    with t.timer(f'count encording'):
        tmp = []
        for c in count_encording_cat:
            if not isinstance(c, list):
                c = [c]
            tmp.extend(c)
        tmp = set(tmp)

        for c in tmp:
            if c not in df.columns:
                df = pd.concat([
                    df, pd.read_feather(f'{FEATURE_DIR}/{c}_train.feather')
                ], axis=1)
                df_test = pd.concat([
                    df_test, pd.read_feather(f'{FEATURE_DIR}/{c}_test.feather')
                ], axis=1)

        for c in count_encording_cat:
            if not isinstance(c, list):
                c = [c]
            print(f'CE', c)
            if len(c) == 1:
                encoder = CountEncoder(
                    input_cols=c
                )
                df = encoder.fit_transform(df)
                df_test = encoder.transform(df_test)
            else:
                fname = '_'.join(c)
                df[fname] = df[c[0]]
                df_test[fname] = df_test[c[0]]
                for x in c[1:]:
                    df[fname] = df[fname].astype(str).str.cat(df[x].astype(str))
                    df_test[fname] = df_test[fname].astype(str).str.cat(df_test[x].astype(str))
                encoder = CountEncoder(
                    input_cols=[fname]
                )
                df = encoder.fit_transform(df)
                df_test = encoder.transform(df_test)

    with t.timer(f'target encording'):
        tmp = []
        for c in target_encording_cat + subj_conbi_cols:
            if not isinstance(c, list):
                c = [c]
            tmp.extend(c)
        tmp = set(tmp)

        for c in tmp:
            if c not in df.columns:
                df = pd.concat([
                    df, pd.read_feather(f'{FEATURE_DIR}/{c}_train.feather')
                ], axis=1)
                df_test = pd.concat([
                    df_test, pd.read_feather(f'{FEATURE_DIR}/{c}_test.feather')
                ], axis=1)

        # 'IsCorrect'
        for tar in ['IsCorrect']:
            for c in target_encording_cat:
                print('TE', tar, c)
                df, df_test = target_encode_cudf_v3(
                    df, df_test, col=c, tar=tar, smooth=te_smooth_factor, min_ct=0, x=-1, shuffle=True)

        for tar in ['IsCorrect']:
            for c in tqdm(target_encording_subj_conbi_cat, total=len(target_encording_subj_conbi_cat)):
                # print('TE', tar, c)
                df_tmp = df[[c[1], tar]]
                df_test_tmp = df_test[[c[1]]]
                df_tmp, df_test_tmp = target_encode_cudf_v3(
                    pd.concat([df_tmp, pd.read_feather(f'{FEATURE_DIR}/{c[0]}_train.feather')], axis=1),
                    pd.concat([df_test_tmp, pd.read_feather(f'{FEATURE_DIR}/{c[0]}_test.feather')], axis=1),
                    col=c, tar=tar, smooth=te_smooth_factor, min_ct=0, x=-1, shuffle=True)
                name = "_".join(c)
                feat = f'TE_s{te_smooth_factor}_{name}_{tar}'
                save_as_feather(feat, df_tmp, df_test_tmp)

        # subject features
        for c in tqdm(subject_features, total=len(subject_features)):
            print(tar, c)
            df, df_test = target_encode_cudf_v3(
                pd.concat([df, pd.read_feather(f'{FEATURE_DIR}/{c}_train.feather')], axis=1),
                pd.concat([df_test, pd.read_feather(f'{FEATURE_DIR}/{c}_test.feather')], axis=1),
                col=c, tar=tar, smooth=te_smooth_factor, min_ct=0, x=-1, shuffle=True)

        # AnswerValue
        target2 = 'AnswerValue'
        df = pd.concat([
            df, pd.get_dummies(df[target2]).add_prefix(f'{target2}_')], axis=1)
        for tar in ['AnswerValue_1', 'AnswerValue_2', 'AnswerValue_3', 'AnswerValue_4']:
            for c in target_encording_cat:
                print('TE', tar, c)
                df, df_test = target_encode_cudf_v3(
                    df, df_test, col=c, tar=tar, smooth=te_smooth_factor, min_ct=0, x=-1, shuffle=True)

    with t.timer(f'target encording - subj agg'):
        subj_te_train = np.zeros((len(df), 388))
        subj_te_test = np.zeros((len(df_test), 388))
        i = 0
        for feat in tqdm(target_encording_subj_features):
            feat_train = pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')[feat].values
            feat_test = pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')[feat].values
            subj_te_train[:, i] = feat_train
            subj_te_test[:, i] = feat_test
            i += 1
        subj_te_train = pd.DataFrame(subj_te_train)
        subj_te_test = pd.DataFrame(subj_te_test)

        feat = 'TE_s5_subj_agg_sum_IsCorrect'
        df[feat] = subj_te_train.sum(axis=1)
        df_test[feat] = subj_te_test.sum(axis=1)

        feat = 'TE_s5_subj_agg_mean_IsCorrect'
        df[feat] = subj_te_train.mean(axis=1)
        df_test[feat] = subj_te_test.mean(axis=1)

        feat = 'TE_s5_subj_agg_std_IsCorrect'
        df[feat] = subj_te_train.std(axis=1)
        df_test[feat] = subj_te_test.std(axis=1)

        feat = 'TE_s5_subj_agg_max_IsCorrect'
        df[feat] = subj_te_train.max(axis=1)
        df_test[feat] = subj_te_test.max(axis=1)

        feat = 'TE_s5_subj_agg_min_IsCorrect'
        df[feat] = subj_te_train.min(axis=1)
        df_test[feat] = subj_te_test.min(axis=1)

        for conbi_col in subj_conbi_cols:
            subj_te_train = np.zeros((len(df), 388))
            subj_te_test = np.zeros((len(df_test), 388))
            i = 0
            for feat in tqdm(target_encording_subj_conbi_features):
                if conbi_col in feat:
                    feat_train = pd.read_feather(f'{FEATURE_DIR}/{feat}_train.feather')[feat].values
                    feat_test = pd.read_feather(f'{FEATURE_DIR}/{feat}_test.feather')[feat].values
                    subj_te_train[:, i] = feat_train
                    subj_te_test[:, i] = feat_test
                    i += 1
            subj_te_train = pd.DataFrame(subj_te_train)
            subj_te_test = pd.DataFrame(subj_te_test)

            feat = f'TE_s5_subj_{conbi_col}_agg_sum_IsCorrect'
            df[feat] = subj_te_train.sum(axis=1)
            df_test[feat] = subj_te_test.sum(axis=1)

            feat = f'TE_s5_subj_{conbi_col}_agg_mean_IsCorrect'
            df[feat] = subj_te_train.mean(axis=1)
            df_test[feat] = subj_te_test.mean(axis=1)

            feat = f'TE_s5_subj_{conbi_col}_agg_std_IsCorrect'
            df[feat] = subj_te_train.std(axis=1)
            df_test[feat] = subj_te_test.std(axis=1)

            feat = f'TE_s5_subj_{conbi_col}_agg_max_IsCorrect'
            df[feat] = subj_te_train.max(axis=1)
            df_test[feat] = subj_te_test.max(axis=1)

            feat = f'TE_s5_subj_{conbi_col}_agg_min_IsCorrect'
            df[feat] = subj_te_train.min(axis=1)
            df_test[feat] = subj_te_test.min(axis=1)

    with t.timer('save dense features'):
        for feat in dense_features:
            logging.info(f'save {feat}')
            save_as_feather(feat, df, df_test)

    with t.timer('save sparse features'):
        for feat in sparse_features:
            logging.info(f'save {feat}')
            df, df_test, unique_num = label_encording(
                feat, df, df_test
            )
            save_as_feather(feat, df, df_test)

    with t.timer('save varlen sparse features'):
        for feat in varlen_sparse_features:
            logging.info(f'save {feat}')
            df, df_test, key2index, unique_num = varlen_label_encording(
                feat, df, df_test
            )
            save_as_parquet(feat, df, df_test)