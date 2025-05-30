import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from typing import Callable, List, Dict, Any
from runner import Runner
from model import Model
from util import logger, Util

class GridSearchRunner(Runner):
    def __init__(self,
                 run_name: str,
                 model_cls: Callable[[str, dict], Model], 
                 params: Dict[str, Any],
                 param_grid: Dict[str, List[Any]],
                 train_x: pd.DataFrame, 
                 train_y: pd.Series, 
                 test_x: pd.DataFrame) -> None:
        super().__init__(run_name, model_cls, params, train_x, train_y, test_x)
        self.param_grid = param_grid

    def run_grid_search(self) -> None:
        """
        グリッドサーチにより、パラメータ探索を行います。
        内部で self.params を更新し、そのままRunnnerクラスと同様に実行できます。
        """
        logger.info(f'{self.run_name} - start grid search')
        best_score = float('inf')
        best_params = {}

        for param in ParameterGrid(self.param_grid):
            logger.info(f'Starting cross-validation for params: {param}')
            self.params.update(param)
            scores = []

            for i_fold in range(self.n_fold):
                logger.info(f'{self.run_name} fold {i_fold} - start training')
                model, _, _, score = self.train_fold(i_fold)
                scores.append(score)

            mean_score: float = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = param

            logger.info(f'Params: {param}, Score: {mean_score}')

        self.params.update(best_params)
        logger.info(f'Best parameters found: {best_params}')
        logger.info(f'Best score: {best_score}')
        logger.info(f'{self.run_name} - end grid search')
