import argparse

import yaml
from ayniy.model.runner import Runner
from sklearn.model_selection import GroupKFold

if __name__ == "__main__":
    """
    cd experiments
    python runner.py --run configs/run000.yml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    args = parser.parse_args()

    cv = GroupKFold(n_splits=10)

    if args.run:
        f = open(args.run, "r+")
        run_configs = yaml.load(f, Loader=yaml.SafeLoader)

        runner = Runner(run_configs, cv)
        runner.run_train_cv()
        runner.run_predict_cv()
        runner.submission()
