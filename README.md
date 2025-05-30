# base
このリポジトリは、Titanicデータセットを用いた生存予測モデルの作成を目的とした機械学習パイプラインです。LightGBMやXGBoost、CatBoost、ニューラルネットワークなど複数のモデル実装と、Optunaによるハイパーパラメータチューニング、特徴量選択のためのグリーディ・フォワード・セレクションなどのコードを含んでいます。


## ディレクトリ構成
- `code/` - モデル学習・評価用のPythonスクリプト
- `code-analysis/` - Notebookなどの分析用ファイル
- `input/` - 学習・評価に用いるデータセット
- `model/` - 学習済みモデルやログを保存するディレクトリ
- `submission/` - 予測結果を保存しKaggleなどへ提出するファイル

各種ファイルの保存先は `code/config.py` に定義された `MODEL_DIR` や `PRED_DIR` で設定できます。

## 使い方
1. 依存ライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
2. `input/` ディレクトリに `train.csv` と `test.csv` を配置します。
3. Notebook もしくは Python スクリプトでデータを読み込み、必要に応じて特徴量エンジニアリングを行います。
4. `Runner` クラスを用いて学習と予測を実行します。以下は LightGBM を用いた例です。
```python
from code.runner import Runner
from code.model_lgb import ModelLGB
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,
    "num_iterations": 5000,
}
runner = Runner(run_name="lgb", model_cls=ModelLGB, params=params,
                train_x=X_train, train_y=y_train, test_x=X_test)
runner.run_train_cv()
runner.run_predict_cv()
```
4. 予測結果は `submission/` 以下に保存されます。`Submission.create_submission('lgb-cv')` を実行すると Kaggle 提出用の CSV が得られます。
5. `OptunaRunner` や `GFSRunner` を利用することでハイパーパラメータ探索や特徴量選択も行えます。
6. コマンドラインから学習・推論を実行する場合は `python code-analysis/main.py --model lgb --run-name lgb` のように実行します。
