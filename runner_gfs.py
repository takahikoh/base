import os
import numpy as np
import pandas as pd
from model import Model
from runner import Runner
from typing import Callable, List, Optional, Tuple, Union
from util import Logger, Util

logger = Logger()

class GFSRunner(Runner):
    def evaluate(self, features: List[str]) -> float:
        """
        指定した特徴量リストに対してクロスバリデーションの平均AUCスコアを返す。
        学習時は self.train_x を選択した特徴量のみでサブセットし評価する。

        :param features: 評価対象の特徴量リスト
        :return: クロスバリデーションによる平均AUCスコア（高いほど良い）
        """
        # 元の学習データを保持
        orig_train_x = self.train_x.copy()
        # 指定した特徴量のみに絞る
        self.train_x = self.train_x[features]
        
        scores = []
        for i_fold in range(self.n_fold):
            # 各foldでのスコアを取得
            _, _, _, score = self.train_fold(i_fold)
            scores.append(score)
        avg_score = np.mean(scores)
        
        # 学習データを元に戻す
        self.train_x = orig_train_x
        return avg_score

    def run_gfs(self) -> List[str]:
        """
        グリーディ・フォワード・セレクションを実行するメソッド。
        すべての特徴量候補に対して、追加したときに最もAUCスコアが向上するものを順次選択する。

        :return: 選択された特徴量のリスト
        """
        best_score = 0.0  # 初期値
        selected = set()
        logger.info('=============== start greedy forward selection ===============')
        
        all_features = list(self.train_x.columns)
        
        while True:
            # すべての特徴が選ばれていれば終了
            if len(selected) == len(all_features):
                break

            scores = []
            for feature in all_features:
                if feature not in selected:
                    # 既に選択された特徴に候補特徴を追加して評価
                    fs = list(selected) + [feature]
                    score = self.evaluate(fs)
                    scores.append((feature, score))
            
            # AUCスコアが最も高い特徴を選択
            b_feature, b_score = sorted(scores, key=lambda tpl: tpl[1], reverse=True)[0]
            if b_score > best_score:
                selected.add(b_feature)
                best_score = b_score
                logger.info(f'=============== selected: {b_feature} ===============')
                logger.info(f'=============== selected features: {selected} ===============')
                logger.info(f'=============== score: {b_score} ===============')
            else:
                # どの特徴を追加してもAUCが向上しなければ終了
                break
        
        logger.info(f'=============== selected features: {selected} ===============')
        return list(selected)
