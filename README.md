# Solution of NeurIPS-Education-Challenge-2020

## overview

...

## usage

### download data

- ...

### run

```
$ cd ./run
$ sh solution.sh
```

## results

- ...

## folder structure

```
Diagnostic-Questions_NeurIPS-Education-Challenge-2020

├── exp # deploy the experiment's config
     └── (experiment name)
             └── config.py

├── features # save extracted feature files
     └── (feature_name).feather

├── folds # save cross validation data table files
     └── (folds_name).csv

├── input
     └── public_dat
         └── feature.name
         └── train.data
         └── validation.data
         └── test.data

├── run
     └── solution.sh
     └── ensemble.py

├── save
     └── model/
     └── model_predict/
     └── predict/

├── src
     └── create_folds.py
     └── dataset.py
     └── feature_extraction.py
     └── model.py
     └── train_5fold.py
     └── utils.py
     └── utils_feature.py
     └── utils_model.py

├── submission # save submission files
     └── (submission_name).zip
```