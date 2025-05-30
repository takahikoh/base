import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Callable, List, Dict, Any
from runner import Runner
from model import Model
from util import logger, Util

class OptunaRunner(Runner):
    def __init__(self,
                 run_name: str,
                 model_cls: Callable[[str, dict], Model], 
                 params: Dict[str, Any], 
                 param_ranges: Dict[str, Dict[str, Any]],
                 train_x: pd.DataFrame, 
                 train_y: pd.Series, 
                 test_x: pd.DataFrame) -> None:
        super().__init__(run_name, model_cls, params, train_x, train_y, test_x)
        self.param_ranges = param_ranges

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Optuna の trial によりパラメータをサンプリングし、クロスバリデーションの平均スコアを返します。
        """
        for param, param_info in self.param_ranges.items():
            if param_info['type'] == 'int':
                self.params[param] = trial.suggest_int(param, param_info['low'], param_info['high'])
            elif param_info['type'] == 'float':
                self.params[param] = trial.suggest_float(param, param_info['low'], param_info['high'],
                                                          log=param_info.get('log', False))
        scores = []
        for i_fold in range(self.n_fold):
            _, _, _, score = self.train_fold(i_fold)
            scores.append(score)
        mean_score = np.mean(scores)
        return mean_score

    def run_optuna(self, n_trials: int = 100) -> None:
        """
        Optuna により、パラメータ探索を行います。
        self.params を更新し、そのままRunnnerクラスと同様に実行できます。
        """
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(self.objective, n_trials=n_trials)
        self.params.update(study.best_params)
        logger.info(f'Best parameters found: {study.best_params}')
        logger.info(f'Best score: {study.best_value}')
