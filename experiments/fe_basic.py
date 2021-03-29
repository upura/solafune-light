import numpy as np
import pandas as pd

from kaggle_utils.features import count_encoding
from kaggle_utils.features.category_encoding import CategoricalEncoder
from kaggle_utils.features.groupby import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer


if __name__ == '__main__':
    train = pd.read_csv('../input/solafune-light/TrainDataSet.csv')
    test = pd.read_csv('../input/solafune-light/EvaluationData.csv')
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    print(train.shape, test.shape)
    # (8359, 16) (8360, 11)
    categorical_cols = [
        'PlaceID', 'Year'
    ]
    numerical_cols = [
        'MeanLight', 'SumLight'
    ]
    target_col = 'AverageLandPrice'

    # target transformation
    train_test[target_col] = np.log1p(train_test[target_col])

    # label encoding
    ce = CategoricalEncoder(categorical_cols)
    train_test = ce.transform(train_test)

    # base
    train_test[['Year'] + numerical_cols + [target_col]].to_feather('../input/feather/train_test.ftr')

    # count encoding
    count_encoding(train_test, categorical_cols).to_feather('../input/feather/count_encoding.ftr')

    # aggregation
    groupby_dict = [{
        'key': [
            'PlaceID'
        ],
        'var': [
            'MeanLight',
            'ce_Year',
            'SumLight'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }]

    original_cols = train_test.columns
    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train_test = groupby.transform(train_test)
    diff = DiffGroupbyTransformer(param_dict=groupby_dict)
    train_test = diff.transform(train_test)
    ratio = RatioGroupbyTransformer(param_dict=groupby_dict)
    train_test = ratio.transform(train_test)
    train_test[list(set(train_test.columns) - set(original_cols))].to_feather('../input/feather/aggregation.ftr')
