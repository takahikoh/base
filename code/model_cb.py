import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from model import Model
from util import Util

class ModelCB(Model):
    def __init__(self, run_fold_name: str, params: dict):
        super().__init__(run_fold_name, params)

    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
              va_x: pd.DataFrame = None, va_y: pd.Series = None) -> None:

        validation = va_x is not None
        
        # ハイパーパラメータの設定
        params = dict(self.params)
        params['random_seed'] = 42

        # CatBoostClassifier のインスタンスを生成
        self.model = CatBoostClassifier(**params)
        
        # 学習
        if validation:
            self.model.fit(
                tr_x,
                y=tr_y,
                eval_set=(va_x, va_y),
                early_stopping_rounds=50,
                verbose=50
            )
        else:
            self.model.fit(tr_x,
                y=tr_y,
                verbose=50)

    def predict(self, te_x: pd.DataFrame) -> np.array:
        return self.model.predict_proba(te_x)[:, 1]

    def save_model(self) -> None:
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)

  