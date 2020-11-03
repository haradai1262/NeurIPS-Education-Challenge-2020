import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from math import ceil
from tqdm import tqdm
import datetime
import cupy
import cudf
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def reduce_svd(features, n_components):
    dr_model = TruncatedSVD(n_components=n_components, random_state=46)
    features_dr = dr_model.fit_transform(features)
    return features_dr


def tfidf_reduce_svd(sentences, n_gram, n_components):
    tfv = TfidfVectorizer(min_df=1, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, n_gram), use_idf=1, smooth_idf=1, sublinear_tf=1)
    dr_model = TruncatedSVD(n_components=n_components, random_state=46)

    tfidf_col = tfv.fit_transform(sentences)
    dr_col = dr_model.fit_transform(tfidf_col)
    return dr_col


def question_subject_svd(question, subject_id_list, n_conponents):
    subject_oht = np.zeros((len(question), len(subject_id_list)), dtype='int8')
    for i, slist in tqdm(enumerate(question['SubjectId'].values), total=len(question)):
        for sid in slist:
            j = subject_id_list.index(sid)
            subject_oht[i, j] += 1

    ques_subj_svd = reduce_svd(subject_oht, n_conponents)
    ques_subj_svd = pd.DataFrame(ques_subj_svd).add_prefix('ques_subj_svd_')
    ques_subj_svd['QuestionId'] = question['QuestionId'].values
    return ques_subj_svd


def user_subject_tfidfsvd(df, n_conponents):
    uids = []
    u_subjects = []
    for uid, udf in tqdm(df[['UserId', 'SubjectId']].groupby('UserId'), total=df['UserId'].nunique()):
        u_subject = []
        for x in udf['SubjectId'].values:
            u_subject.extend(x)
        uids.append(uid)
        u_subjects.append(u_subject)
    u_subjects = [' '.join(map(str,x)) for x in u_subjects]
    user_subj_svd = tfidf_reduce_svd(u_subjects, n_gram=1, n_components=n_conponents)
    user_subj_svd = pd.DataFrame(user_subj_svd).add_prefix('user_subj_svd_')
    user_subj_svd['UserId'] = uids
    return user_subj_svd



def target_encode_cudf_v3(train, valid, col, tar, n_folds=5, min_ct=0, smooth=20, 
                          seed=42, shuffle=False, t2=None, v2=None, x=-1):
    '''
    ref: https://github.com/rapidsai/deeplearning/blob/main/RecSys2020/02_ModelsCompetition/XGBoost3/XGBoost3.ipynb
    '''
    #
    # col = column to target encode (or if list of columns then multiple groupby)
    # tar = tar column encode against
    # if min_ct>0 then all classes with <= min_ct are consider in new class "other"
    # smooth = Bayesian smooth parameter
    # seed = for 5 Fold if shuffle==True
    # if x==-1 result appended to train and valid
    # if x>=0 then result returned in column x of t2 and v2
    #
    # SINGLE OR MULTIPLE COLUMN
    if not isinstance(col, list):
        col = [col]
    if (min_ct > 0) & (len(col) > 1):
        print('WARNING: Setting min_ct=0 with multiple columns. Not implemented')
        min_ct = 0
    name = "_".join(col)

    # FIT ALL TRAIN
    gf = cudf.from_pandas(train[col + [tar]]).reset_index(drop=True)
    gf['idx'] = gf.index  # needed because cuDF merge returns out of order
    if min_ct > 0:  # USE MIN_CT?
        other = gf.groupby(col[0]).size()
        other = other[other <= min_ct].index
        save = gf[col[0]].values.copy()
        gf.loc[gf[col[0]].isin(other), col[0]] = -1
    te = gf.groupby(col)[[tar]].agg(['mean', 'count']).reset_index()
    te.columns = col + ['m', 'c']
    mn = gf[tar].mean().astype('float32')
    te['smooth'] = ((te['m'] * te['c']) + (mn * smooth)) / (te['c'] + smooth)
    if min_ct > 0:
        gf[col[0]] = save.copy()

    # PREDICT VALID
    gf2 = cudf.from_pandas(valid[col]).reset_index(drop=True)
    gf2['idx'] = gf2.index
    if min_ct > 0:
        gf2.loc[gf2[col[0]].isin(other), col[0]] = -1
    gf2 = gf2.merge(te[col + ['smooth']], on=col, how='left', sort=False).sort_values('idx')
    if x == -1:
        valid[f'TE_s{smooth}_{name}_{tar}'] = gf2['smooth'].fillna(mn).astype('float32').to_array()
    elif x >= 0:
        v2[:, x] = gf2['smooth'].fillna(mn).astype('float32').to_array()

    # KFOLD ON TRAIN
    tmp = cupy.zeros((train.shape[0]), dtype='float32')
    gf['fold'] = 0
    if shuffle:  # shuffling is 2x slower
        kf = KFold(n_folds, random_state=seed, shuffle=shuffle)
        for k, (idxT, idxV) in enumerate(kf.split(train)):
            gf.loc[idxV, 'fold'] = k
    else:
        fsize = train.shape[0] // n_folds
        gf['fold'] = cupy.clip(gf.idx.values // fsize, 0, n_folds - 1)
    for k in range(n_folds):
        if min_ct > 0:  # USE MIN CT?
            if k < n_folds - 1:
                save = gf[col[0]].values.copy()
            other = gf.loc[gf.fold != k].groupby(col[0]).size()
            other = other[other <= min_ct].index
            gf.loc[gf[col[0]].isin(other), col[0]] = -1
        te = gf.loc[gf.fold != k].groupby(col)[[tar]].agg(['mean', 'count']).reset_index()
        te.columns = col + ['m', 'c']
        mn = gf.loc[gf.fold != k, tar].mean().astype('float32')
        te['smooth'] = ((te['m'] * te['c']) + (mn * smooth)) / (te['c'] + smooth)
        gf = gf.merge(te[col + ['smooth']], on=col, how='left', sort=False).sort_values('idx')
        tmp[(gf.fold.values == k)] = gf.loc[gf.fold == k, 'smooth'].fillna(mn).astype('float32').values
        gf.drop_column('smooth')
        if (min_ct > 0) & (k < n_folds - 1):
            gf[col[0]] = save.copy()
    if x == -1:
        train[f'TE_s{smooth}_{name}_{tar}'] = cupy.asnumpy(tmp.astype('float32'))
    elif x >= 0:
        t2[:, x] = cupy.asnumpy(tmp.astype('float32'))
    return train, valid


def extract_age(df):
    # Diagnostic Questions: The NeurIPS 2020 Education Challenge
    # roughly between 7 and 18 years old [3]
    age_days = []
    age_years = []
    pri_to_high_stu = []
    for i, j, k in tqdm(df[['DateAnswered_dt', 'DateOfBirth_dt', 'DateOfBirth_NaN']].values, total=len(df)):
        if j.year <= 2011 and j.year >= 2000:
            age = i - j
            days = age.days
            years = int(days / 365)           
            if years >= 7 and years <= 18:
                age_years.append(years)
                age_days.append(days)
                pri_to_high_stu.append(True)
            elif years < 7:
                age_years.append(7)
                age_days.append(7 * 365)
                pri_to_high_stu.append(False)
            elif years > 18:
                age_years.append(18)
                age_days.append(18 * 365)
                pri_to_high_stu.append(False)
        elif j.year > 2011:
            age = i - datetime.datetime(2011, 1, 1, 0, 0)
            days = age.days
            years = int(days / 365)
            age_days.append(days)
            age_years.append(years)
            pri_to_high_stu.append(False)
        elif j.year < 2000:
            age = i - datetime.datetime(1999, 1, 1, 0, 0)
            days = age.days
            years = int(days / 365)
            age_days.append(days)
            age_years.append(years)
            pri_to_high_stu.append(False)
    df['age_days'] = age_days
    df['age_years'] = age_years
    df['pri_to_high_stu'] = pri_to_high_stu
    return df


def preprocess_subject(subject):
    subject['ParentId'] = subject['ParentId'].fillna(0.0).astype(int)

    def chilidren_num(x):
        return len(childrens[x])

    childrens = {}
    for i, j in subject[['SubjectId', 'ParentId']].values:
        if j not in childrens:
            childrens[j] = []
        if i not in childrens:
            childrens[i] = []
        childrens[j].append(i)

    parent_dict = {}
    for i, j in subject[['SubjectId', 'ParentId']].values:
        parent_dict[i]= j

    subject['SubjectId_cnum'] = subject['SubjectId'].apply(chilidren_num)
    subject['Level__SubjectId_cnum'] = subject['Level'].astype(str) + '_' + subject['SubjectId_cnum'].astype(str)

    sid_w_parents = []
    for i in subject['SubjectId'].values:
        x = i
        parenets = [x]
        while parent_dict[x] != 0:
            parenets.append(parent_dict[x])
            x = parent_dict[x]
        sid_w_parents.append(parenets)
    subject['SubjectId_with_parents'] = sid_w_parents

    return subject


def preprocess_DateOfBirth(student):

    # cal median
    timestamp = []
    for x in tqdm(student['DateOfBirth'].values, total=len(student)):
        if type(x) != str:
            continue
        else:
            timestamp.append(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.000').timestamp())
    medain_dt = pd.Series(timestamp).median()
    medain_dt = datetime.datetime.fromtimestamp(medain_dt)

    # fill median
    timestamp = []
    for x in tqdm(student['DateOfBirth'].values, total=len(student)):
        if type(x) != str:
            timestamp.append(medain_dt)
        else:
            timestamp.append(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.000'))
    
    student['DateOfBirth_dt'] = timestamp
    return student


def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))


def factorize_dataAnswered(answer):
    timestamp = []
    for x in tqdm(answer['DateAnswered'].values, total=len(answer)):
        timestamp.append(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.000'))

    weekday = [ts.weekday() for ts in timestamp]
    hour = [ts.hour for ts in timestamp]
    day = [ts.day for ts in timestamp]
    wom = [week_of_month(ts) for ts in timestamp]

    answer['DateAnswered_dt'] = timestamp
    answer['DateAnswered_weekday'] = weekday
    answer['DateAnswered_hour'] = hour
    answer['DateAnswered_day'] = day
    answer['DateAnswered_wom'] = wom
    return answer


def label_encording(colname, X_train, X_test):
    lbe = LabelEncoder()
    tmp = pd.concat([
        X_train[colname], X_test[colname]])
    lbe.fit(tmp)
    X_train[colname] = lbe.transform(X_train[colname])
    X_test[colname] = lbe.transform(X_test[colname])
    return X_train, X_test, len(lbe.classes_)


def subject2level(subject, df, df_test):

    subject2level_dic = {i: j for i, j in subject[['SubjectId', 'Level']].values}
    subject_levels = []
    for i in df['SubjectId'].values:
        row = []
        for s in i:
            row.append(subject2level_dic[s])
        subject_levels.append(row)
    df['SubjectId_level'] = subject_levels

    subject_levels = []
    for i in df_test['SubjectId'].values:
        row = []
        for s in i:
            row.append(subject2level_dic[s])
        subject_levels.append(row)
    df_test['SubjectId_level'] = subject_levels

    return df, df_test


def map_varlen(x, dict):
    return [dict[i] for i in x]


def varlen_label_encording(colname, X_train, X_test):

    entities = []
    for x in X_train[colname].values:
        entities.extend(x)
    for x in X_test[colname].values:
        entities.extend(x)
    entities = pd.Series(entities)

    count = entities.value_counts()

    key2index = {}  # null: 0, ...
    idx = 1
    for i in count.index:
        key2index[i] = idx
        idx += 1
    unique_num = idx

    X_train[colname] = X_train[colname].apply(map_varlen, dict=key2index)
    X_test[colname] = X_test[colname].apply(map_varlen, dict=key2index)

    print(f'{colname} - Num. of unique category: {unique_num}')

    return X_train, X_test, key2index, unique_num


def varlen_label_encording_threshold(colname, X_train, X_test, low_freq_th):

    entities = []
    for x in X_train[colname].values:
        entities.extend(x)
    for x in X_test[colname].values:
        entities.extend(x)
    entities = pd.Series(entities)

    count = entities.value_counts()
    low_freq_list = count[count <= low_freq_th].index.tolist()
    count = count[count > low_freq_th]

    key2index = {i: 1 for i in low_freq_list}  # null: 0, low_freq: 1, ...
    idx = 2
    for i in count.index:
        key2index[i] = idx
        idx += 1
    unique_num = idx + 1

    X_train[colname] = X_train[colname].apply(map_varlen, dict=key2index)
    X_test[colname] = X_test[colname].apply(map_varlen, dict=key2index)

    print(f'{colname} - Num. of use as low frequency category: {len(low_freq_list)}')
    print(f'{colname} - Num. of unique category: {unique_num}')

    return X_train, X_test, key2index, unique_num
