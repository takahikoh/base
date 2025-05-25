import os
import numpy as np
import pandas as pd
import xgboost as xgb
from model import Model
from util import Util


class ModelXGB(Model):
    def __init__(self, run_fold_name: str, params: dict):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        #パラメータ
        params = dict(self.params)
        params['seed'] = 42
        num_boost_round = params.pop('num_boost_round')
        
        # コールバック
        es = xgb.callback.EarlyStopping(
          rounds=50,
          min_delta=0.0001,
          save_best=True,
          maximize=True,
          metric_name='auc'
          )
        
        # 学習
        if validation:
            self.model = xgb.train(params=params,
                                   dtrain=dtrain,
                                   num_boost_round=num_boost_round,
                                   evals=[(dtrain, 'train'), (dvalid, 'eval')],
                                   callbacks=[es],
                                   verbose_eval = 50
                                   )
            
        else:
            self.model = xgb.train(params=params,
                                   dtrain=dtrain,
                                   num_boost_round=num_boost_round,
                                   evals=[(dtrain, 'train')],
                                   verbose_eval = 50
                                   )

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest)
                      
    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)
        
    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
        