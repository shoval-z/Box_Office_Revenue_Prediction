from hpsklearn import HyperoptEstimator
from hpsklearn import any_regressor
from hpsklearn import xgboost_regression
from hyperopt import hp
from hyperopt import tpe
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from main import data_prep
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
# Load and split dataset
train_df = pd.read_csv('hw1_data/train.tsv', sep="\t")

x, y, runtime_median, budget_median,top_p,top_producer,top_e_producer,top_director,top_actors = data_prep(train_df)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
y = np.log1p(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)
xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.01, 0.2, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_log_error(y, pred))
# model = HyperoptEstimator(regressor=any_regressor('reg'), loss_fn=mean_squared_log_error, algo=tpe.suggest, max_evals=50, trial_timeout=30)
model = HyperoptEstimator(regressor=xgboost_regression('xgb',xgb_reg_params), loss_fn=mean_squared_log_error, algo=tpe.suggest, max_evals=50, trial_timeout=30)
# preprocessing=any_preprocessing('pre'),
# algo=tpe.suggest,
# max_evals=10,
# trial_timeout=30)
# Training
model.fit(x_train, y_train)
# Results
print(f"Train score: {model.score(x_train, y_train)}")
print(f"Test score: {model.score(x_test, y_test)}")
# Best model
print(f"Optimal configuration: {model.best_model()}")
