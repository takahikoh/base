import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from model import Model
from util import Util


class ModelLGB(Model):
    def __init__(self, run_fold_name: str, params: dict):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        
        # データセット
        validation = va_x is not None
        dtrain = lgb.Dataset(tr_x, label=tr_y)
        if validation:
            dvalid = lgb.Dataset(va_x, label=va_y, reference=dtrain)
        
        # ハイパーパラメータの設定
        params = dict(self.params)
        params['seed'] = 42
        num_boost_round = params.pop('num_iterations')

        # コールバック
        es  = lgb.early_stopping(stopping_rounds=50, first_metric_only=False, verbose=True, min_delta=0.0001)
        log = lgb.log_evaluation(period=50)

        # 学習
        if validation:
            self.model = lgb.train(params,
                                   dtrain,
                                   num_boost_round=num_boost_round,
                                   valid_sets=[dtrain, dvalid],
                                   valid_names=['train', 'eval'],
                                   callbacks=[es, log]) 

        else:
            self.model = lgb.train(params,
                                   dtrain,
                                   num_boost_round=num_boost_round,
                                   valid_sets=[dtrain],
                                   valid_names=['train'],
                                   callbacks=[log]) 

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)
    
    # save_model and load_model are inherited from Model
