import argparse
from pathlib import Path
import pandas as pd
from code.util import Submission
from code.runner import Runner
from code.model_lgb import ModelLGB
from code.model_xgb import ModelXGB
from code.model_cb import ModelCB
from code.model_nn import ModelNN

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_data(target: str):
    train = pd.read_csv(ROOT_DIR / 'input/train.csv')
    test = pd.read_csv(ROOT_DIR / 'input/test.csv')
    y = train[target]
    train = train.drop(columns=[target])
    full = pd.concat([train, test], axis=0)
    full = pd.get_dummies(full)
    x_train = full.iloc[:len(train)].reset_index(drop=True)
    x_test = full.iloc[len(train):].reset_index(drop=True)
    return x_train, y, x_test


def get_model_cls(name: str):
    if name == 'lgb':
        return ModelLGB
    if name == 'xgb':
        return ModelXGB
    if name == 'cb':
        return ModelCB
    if name == 'nn':
        return ModelNN
    raise ValueError(f'unknown model: {name}')


def main():
    parser = argparse.ArgumentParser(description='Simple training pipeline')
    parser.add_argument('--model', default='lgb', choices=['lgb', 'xgb', 'cb', 'nn'])
    parser.add_argument('--run-name', default='exp')
    parser.add_argument('--target', default='Survived')
    args = parser.parse_args()

    train_x, train_y, test_x = load_data(args.target)
    model_cls = get_model_cls(args.model)
    params = {}

    runner = Runner(run_name=args.run_name,
                    model_cls=model_cls,
                    params=params,
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission(f'{args.run_name}-cv')


if __name__ == '__main__':
    main()
