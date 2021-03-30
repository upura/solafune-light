import pandas as pd


if __name__ == '__main__':
    sub = pd.read_csv('../output/submissions/submission_weight001.csv')
    sub['LandPrice'] = sub['LandPrice'] * 0.9
    # sub['LandPrice'] = sub.groupby('PlaceID')['LandPrice'].transform('mean').values
    sub.to_csv('../output/submissions/submission_weight001_09.csv', index=False)
