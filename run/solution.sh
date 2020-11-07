
export RANDOM_STATE='46'
export INPUT_DIR='../data'
export FEATURE_DIR='../features'
export FOLD_DIR='../folds'
export SUB_DIR='../submission'
export EXP_DIR='../exp'
# export DEVICE='cuda:1'

'''
- create folds
  - `create_folds.py`
'''
export FOLD_NAME='mskf_user'
export FOLD_NUM='5'
python ../src/create_folds.py

'''
- Feature extraction
  - `feature_extraction.py`
  - extract features from original data for training the model
'''
python ../src/feature_extraction.py

'''
- Training the models using all features
  - EXP_NAME: `task{1 or 2}_{model}`
  - script: `train_{model}.py`
  - hyperparameters and evaluation scores of each model are saved in mlflow
  - refer to `config.py` to set the features and hyperparameters to be used to train the model
'''

export EXP_NAME='task1_lgbm' # lgb, all feature, te_smooth5
python ../src/train_lgbm.py

export EXP_NAME='task1_lgbm_2' # lgb, all feature, te_smooth2
python ../src/train_lgbm.py

export EXP_NAME='task1_xgb_2' # xgb, all feature, te_smooth5
python ../src/train_xgb.py

export EXP_NAME='task1_cat' # cat, all feature, te_smooth5
python ../src/train_cat.py

export EXP_NAME='task2_lgbm' # lgb, all feature, te_smooth5
python ../src/train_lgbm.py

export EXP_NAME='task2_xgb_2' # xgb, all feature, te_smooth5
python ../src/train_xgb.py


'''
- Training the models using selected features
  - EXP_NAME: `task{1 or 2 or 12}_{model}_{feature_select}`
  - script: `train_{model}.py`
'''

export EXP_NAME='task1_lgbm_fs100' # lgb, fs100, te_smooth5
python ../src/train_lgbm.py

export EXP_NAME='task1_xgb_fs100' # xgb, fs100, te_smooth5
python ../src/train_xgb.py

export EXP_NAME='task1_cat_fs100' # cat depth 8, fs100, te_smooth5
python ../src/train_cat.py

export EXP_NAME='task1_cat_2_fs100' # cat depth 10, fs100, te_smooth5
python ../src/train_cat.py

export EXP_NAME='task1_mlp_fs100' # mlp, fs100, te_smooth5
python ../src/train_mlp.py

export EXP_NAME='task2_lgbm_fs100' # lgb, fs100, te_smooth5
python ../src/train_lgbm.py

export EXP_NAME='task2_xgb_fs100' # xgb, fs100, te_smooth5
python ../src/train_xgb.py

export EXP_NAME='task2_cat_fs100' # cat depth 8, fs100, te_smooth5
python ../src/train_cat.py

export EXP_NAME='task2_mlp_fs100' # mlp, fs100, te_smooth5
python ../src/train_mlp.py

export EXP_NAME='task12_multitask_mlp_fs100' # mlp multi, fs100, te_smooth5
python ../src/train_mlp_multitask.py

# ...

'''
- Feature extraction from prediction results
'''
python ../src/feature_extraction_preds_meta.py

'''
- Training the models using prediction resuls features, 2nd-stage
'''

export EXP_NAME='task1_xgb_fs100_meta' 
python ../src/train_xgb.py

export EXP_NAME='task1_xgb_fs50_meta'
python ../src/train_xgb.py

export EXP_NAME='task1_cat_fs50_meta'
python ../src/train_cat.py

export EXP_NAME='task2_xgb_fs100_meta' 
python ../src/train_xgb.py

export EXP_NAME='task2_xgb_fs50_meta'
python ../src/train_xgb.py

export EXP_NAME='task12_multitask_mlp_fs50_meta'
python ../src/train_mlp_multitask.py

'''
- Ridge stacking
'''
python ../stacking.py