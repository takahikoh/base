import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from model import Model
from util import Util


class ModelLGB_2(Model):
    def __init__(self, run_fold_name: str, params: dict):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        
        # データセット
        validation = va_x is not None
        train_weights = np.where(tr_y == 1, 2, 1)
        dtrain = lgb.Dataset(tr_x, label=tr_y, weight=train_weights)
        if validation:
            valid_weights = np.where(va_y == 1, 2, 1)
            dvalid = lgb.Dataset(va_x, label=va_y, weight=valid_weights, reference=dtrain)
        
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
    
    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)
        
    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)