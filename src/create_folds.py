import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

INPUT_DIR = os.environ.get('INPUT_DIR')
FOLD_DIR = os.environ.get('FOLD_DIR')

RANDOM_STATE = os.environ.get('RANDOM_STATE')

FOLD_NAME = os.environ.get('FOLD_NAME')
FOLD_NUM = os.environ.get('FOLD_NUM')

if __name__ == "__main__":

    data_path = f'{INPUT_DIR}/train_data/train_task_1_2.csv'
    X = pd.read_csv(data_path)
    y = X['IsCorrect']

    X.loc[:, 'kfold'] = 0
    if FOLD_NAME == 'mskf_user':
        mskf = MultilabelStratifiedKFold(n_splits=FOLD_NUM, random_state=RANDOM_STATE, shuffle=True)
        labels = X[['IsCorrect', 'UserId']].values
        splits = mskf.split(X.values, labels)

    for fold, (trn, val) in enumerate(splits):
        X.loc[val, 'kfold'] = fold

    print(X.kfold.value_counts())
    X[['kfold']].to_csv(f'{FOLD_DIR}/train_folds_{FOLD_NAME}{FOLD_NUM}_RS{RANDOM_STATE}.csv', index=False)
