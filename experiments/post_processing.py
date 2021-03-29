import pandas as pd


if __name__ == '__main__':
    sub = pd.read_csv('../output/submissions/submission_weight000.csv')
    sub['LandPrice'] = sub.groupby('PlaceID')['LandPrice'].transform('mean').values
    sub.to_csv('../output/submissions/submission_weight000_pp.csv', index=False)
