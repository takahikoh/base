import os
import numpy as np
import random
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU, Input, Masking
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from model import Model
from util import Util


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    

class ModelNN(Model):
    def __init__(self, run_fold_name: str, params: dict):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        
        set_seed(42)
        
        # データのセット・スケーリング
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        pipeline = Pipeline(steps=[('imputer', imputer), ('scaler', scaler)])

        tr_x = pipeline.fit_transform(tr_x)
        if va_x is not None:
            va_x = pipeline.transform(va_x)

        # パラメータ
        layers = self.params['layers']
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']
        learning_rate = self.params['learning_rate']

        # モデル構築
        inputs = Input(shape=(tr_x.shape[1],))
        x = Dense(units)(inputs)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        for _ in range(layers - 1):
            x = Dense(units)(x)
            x = PReLU()(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)

        outputs = Dense(1, activation='sigmoid')(x)
        
        model = KerasModel(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)

        # モデルをコンパイル
        model.compile(loss=BinaryCrossentropy(), optimizer=optimizer)  

        # 学習
        callbacks = []
        if va_x is not None:
            es = EarlyStopping(monitor='val_loss', patience=patience,
                                           verbose=1, restore_best_weights=True)
            callbacks.append(es)
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2,
                      validation_data=(va_x, va_y), callbacks=callbacks)
        else:
            model.fit(tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=2)

        # モデルとパイプラインの保存
        self.model = model
        self.pipeline = pipeline

        
    def predict(self, te_x):
        te_x = self.pipeline.transform(te_x)
        pred = self.model.predict(te_x)
        return pred.ravel()

    
    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.keras')
        pipeline_path = os.path.join('../model/model', f'{self.run_fold_name}-pipeline.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)  
        Util.dump(self.pipeline, pipeline_path)

        
    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.keras')
        pipeline_path = os.path.join('../model/model', f'{self.run_fold_name}-pipeline.pkl')
        self.model = tf.keras.models.load_model(model_path)
        self.pipeline = Util.load(pipeline_path)

