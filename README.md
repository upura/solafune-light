# Solafune: 夜間光データから土地価格を予測 6th place solution

This repository contains source code of 6th place solution for a machine learning competition named "[Solafune: 夜間光データから土地価格を予測](https://solafune.com/#/competitions/f03f39cc-597b-4819-b1a5-41479d4b73d6)".

## Overview

The best score is given by weighted averaging of two models:

1. LightGBM with 377 features and pseudo labels.
2. CatBoost with 377 features and pseudo labels.

### Preprocessing

I used the following features:

- original features
- label encoding for categorical features
- count encoding for categorical features
- aggregation features
- [mst8823's baseline feature](https://zenn.dev/mst8823/articles/cd40cb971f702e)

### Training

I first trained model by LightGBM, then added test data with pseudo labels for re-training.
The same precess was conducted for CatBoost.

### Prediction

The final prediction was gained by weighted averaging of the two predictions.
The weight was determined to minimize the validation score.

- LightGBM (weight: 0.70615234): 0.5019390454450023
- CatBoost (weight: 1 - 0.70615234): 0.5133175632686356
- optimized: 0.49952261316789714

## How to Reproduce

This implementation uses a supporting tool for machine learning competitions named [Ayniy](https://github.com/upura/ayniy), and feature engineering tool named [kaggle_utils](https://github.com/Ynakatsuka/kaggle_utils).

### Environment

```bash
docker-compose -d --build
docker exec -it solafune-light bash
```

### Run

```bash
cd experiments
sh 1_preprocess.sh
sh 2_train.sh
sh 3_predict.sh
```
