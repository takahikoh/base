import os
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional

from util import Util
import config


class Model(metaclass=ABCMeta):

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """コンストラクタ

        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する

        :param tr_x: 学習データの特徴量
        :param tr_y: 学習データの目的変数
        :param va_x: バリデーションデータの特徴量
        :param va_y: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """学習済のモデルでの予測値を返す

        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    FILE_EXT = '.model'

    def _model_path(self) -> str:
        return os.path.join(config.MODEL_DIR, f'{self.run_fold_name}{self.FILE_EXT}')

    def save_model(self) -> None:
        """モデルの保存を行う"""
        model_path = self._model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        model_path = self._model_path()
        self.model = Util.load(model_path)
