import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


# aggregationのagg_methodsで使用
def max_min(x):
    return x.max()-x.min()

def q75_q25(x):
    return x.quantile(0.75) - x.quantile(0.25)

# xfeat.aggregationとほぼ同じ
def aggregation(input_df, group_key, group_values, agg_methods):
    new_df = []
    for agg_method in agg_methods:
        for col in group_values:
            if callable(agg_method):
                agg_method_name = agg_method.__name__
            else:
                agg_method_name = agg_method
            new_col = f"agg_{agg_method_name}_{col}_grpby_{group_key}"
            df_agg = (input_df[[col] + [group_key]].groupby(group_key)[[col]].agg(agg_method))
            df_agg.columns = [new_col]
            new_df.append(df_agg)
            
    _df = pd.concat(new_df, axis=1).reset_index()
    output_df = pd.merge(input_df[[group_key]], _df, on=group_key, how="left")
    return output_df.drop(group_key, axis=1)

# group 内で diffをとる関数
def diff_aggregation(input_df, group_key, group_values, num_diffs):
    dfs = []
    for nd in num_diffs:
        _df = input_df.groupby(group_key)[group_values].diff(nd)
        _df.columns = [f'diff={nd}_{col}_grpby_{group_key}' for col in group_values]
        dfs.append(_df)
    output_df = pd.concat(dfs, axis=1)
    return output_df

# group 内で shiftをとる関数
def shift_aggregation(input_df, group_key, group_values, num_shifts):
    dfs = []
    for ns in num_shifts:
        _df = input_df.groupby(group_key)[group_values].shift(ns)
        _df.columns = [f'shift={ns}_{col}_grpby_{group_key}' for col in group_values]
        dfs.append(_df)
    output_df = pd.concat(dfs, axis=1)
    return output_df


# 面積(?)
def get_area_feature(input_df):
    output_df = pd.DataFrame()
    output_df["Area"] = input_df["SumLight"] / (input_df["MeanLight"]+1e-3)
    return output_df

# PlaceIDをキーにした集約特徴量
def get_agg_place_id_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)
    group_key = "PlaceID"
    group_values = ["MeanLight", "SumLight", "Area"]
    agg_methods = ["min", "max", "median", "mean", "std", max_min, q75_q25]
    output_df = aggregation(_input_df, 
                            group_key=group_key, 
                            group_values=group_values,
                            agg_methods=agg_methods)
    return output_df

# Year をキーにした集約特徴量
def get_agg_year_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)   
    group_key = "Year"
    group_values = ["MeanLight", "SumLight", "Area"]
    agg_methods = ["min", "max", "median", "mean", "std", max_min, q75_q25]
    output_df = aggregation(_input_df, 
                            group_key=group_key, 
                            group_values=group_values,
                            agg_methods=agg_methods)
    return output_df

# PlaceID をキーにしたグループ内差分
def get_diff_agg_place_id_features(input_df):
    group_key = "PlaceID"
    group_values = ["MeanLight", "SumLight"]
    num_diffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    output_df = diff_aggregation(input_df, 
                                 group_key=group_key, 
                                 group_values=group_values, 
                                 num_diffs=num_diffs)
    return output_df

# PlaceID をキーにしたグループ内シフト
def get_shift_agg_place_id_features(input_df):
    group_key = "PlaceID"
    group_values = ["MeanLight", "SumLight"]
    num_shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    output_df = shift_aggregation(input_df, 
                                  group_key=group_key, 
                                  group_values=group_values, 
                                  num_shifts=num_shifts)
    return output_df

# pivot tabel を用いた特徴量
def get_place_id_vecs_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)
    # pivot table
    area_df = pd.pivot_table(_input_df, index="PlaceID", columns="Year", values="Area").add_prefix("Area=")
    mean_light_df = pd.pivot_table(_input_df, index="PlaceID", columns="Year", values="MeanLight").add_prefix("MeanLight=")
    sum_light_df = pd.pivot_table(_input_df, index="PlaceID", columns="Year", values="SumLight").add_prefix("SumLight=")
    all_df = pd.concat([area_df, mean_light_df, sum_light_df], axis=1)
    
    # PCA all 
    sc_all_df = StandardScaler().fit_transform(all_df.fillna(0))
    pca = PCA(n_components=64, random_state=2021)
    pca_all_df = pd.DataFrame(pca.fit_transform(sc_all_df), index=all_df.index).rename(columns=lambda x: f"PlaceID_all_PCA={x:03}")
    # PCA Area
    sc_area_df = StandardScaler().fit_transform(area_df.fillna(0))
    pca = PCA(n_components=16, random_state=2021)
    pca_area_df = pd.DataFrame(pca.fit_transform(sc_area_df), index=all_df.index).rename(columns=lambda x: f"PlaceID_Area_PCA={x:03}")
    # PCA MeanLight
    sc_mean_light_df = StandardScaler().fit_transform(mean_light_df.fillna(0))
    pca = PCA(n_components=16, random_state=2021)
    pca_mean_light_df = pd.DataFrame(pca.fit_transform(sc_mean_light_df), index=all_df.index).rename(columns=lambda x: f"PlaceID_MeanLight_PCA={x:03}")
    # PCA SumLight
    sc_sum_light_df = StandardScaler().fit_transform(sum_light_df.fillna(0))
    pca = PCA(n_components=16, random_state=2021)
    pca_sum_light_df = pd.DataFrame(pca.fit_transform(sc_sum_light_df), index=all_df.index).rename(columns=lambda x: f"PlaceID_SumLight_PCA={x:03}")
    
    df = pd.concat([all_df, pca_all_df, pca_area_df, pca_mean_light_df, pca_sum_light_df], axis=1)
    output_df = pd.merge(_input_df[["PlaceID"]], df, left_on="PlaceID", right_index=True, how="left")
    return output_df.drop("PlaceID", axis=1)

# PlaceIDをキーにしたグループ内相関係数
def get_corr_features(input_df):
    _input_df = pd.concat([input_df, get_area_feature(input_df)], axis=1)
    group_key = "PlaceID"
    group_vlaues = [
        ["Year", "MeanLight"],
        ["Year", "SumLight"],
        ["Year", "Area"],
    ]
    dfs = []
    for gv in group_vlaues:
        _df = _input_df.groupby(group_key)[gv].corr().unstack().iloc[:, 1].rename(f"Corr={gv[0]}-{gv[1]}")
        dfs.append(pd.DataFrame(_df))
    dfs = pd.concat(dfs, axis=1)
    output_df = pd.merge(_input_df[[group_key]], dfs, left_on=group_key, right_index=True, how="left").drop(group_key, axis=1)
    return output_df
    
# count 63
def get_count63_feature(input_df):
    # 各地域でMeanLightが63をとった回数を特徴量にする
    _mapping = input_df[input_df['MeanLight']==63].groupby('PlaceID').size()
    
    output_df = pd.DataFrame()
    output_df['count63'] = input_df['PlaceID'].map(_mapping).fillna(0)
    return output_df


# 特徴量作成用の関数を実行する関数
def preprocess(train, test):
    input_df = pd.concat([train, test]).reset_index(drop=True)  # use concat data

    # load process functions
    process_blocks = [
        get_area_feature,
        get_agg_place_id_features,
        get_agg_year_features,
        get_diff_agg_place_id_features,
        get_shift_agg_place_id_features,
        get_place_id_vecs_features,
        get_corr_features,
        get_count63_feature
    ]

    # preprocess
    output_df = []
    for func in tqdm(process_blocks):
        _df = func(input_df)
        output_df.append(_df)
    output_df = pd.concat(output_df, axis=1)
    
    return output_df


if __name__ == '__main__':
    train = pd.read_csv('../input/solafune-light/TrainDataSet.csv')
    test = pd.read_csv('../input/solafune-light/EvaluationData.csv')
    output_df = preprocess(train, test)
    output_df.to_feather('../input/feather/mst.ftr')
