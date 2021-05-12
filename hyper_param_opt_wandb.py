from sklearn.model_selection import train_test_split
from main import data_prep
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
import wandb
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def main():
    run = wandb.init()
    alpha = run.config.alpha
    lr = run.config.learning_rate
    max_depth = run.config.max_depth
    subsample = run.config.subsample
    min_child_weight = run.config.min_child_weight
    colsample_bytree = run.config.colsample_bytree

    train_df = pd.read_csv('hw1_data/train.tsv', sep="\t")
    test_df = pd.read_csv('hw1_data/test.tsv', sep="\t")

    x, y, runtime_median, budget_median, top_p, top_producer, top_e_producer, top_director, top_actors = data_prep(
        train_df)
    x_test, y_test = data_prep(test_df, budget_val=budget_median, runtime_val=runtime_median,top_p=top_p,
                               top_producer=top_producer,top_e_producer=top_e_producer,top_director=top_director,
                               top_actors=top_actors)
    scaler = MinMaxScaler()
    scaler.fit(x)
    new_train_x = scaler.transform(x)
    new_test_x = scaler.transform(x_test)
    y_train = np.log1p(y)

    X_train_mini, X_val, y_train_mini, y_val = train_test_split(new_train_x, y_train, test_size=0.2, random_state=42)

    train_set = xgb.DMatrix(X_train_mini, label=y_train_mini)
    val_set = xgb.DMatrix(X_val, label=y_val)
    test_set = xgb.DMatrix(new_test_x, label=y_test)

    parameters_xgb = dict()
    parameters_xgb['alpha'] = alpha
    parameters_xgb['subsample'] = subsample
    parameters_xgb['learning_rate'] = lr
    parameters_xgb['max_depth'] = max_depth
    parameters_xgb['min_child_weight'] = min_child_weight
    parameters_xgb['colsample_bytree'] = colsample_bytree

    clf_xgb = xgb.train(params=parameters_xgb,
                        dtrain=train_set,
                        num_boost_round=1000,
                        evals=[(val_set, "Test")],
                        early_stopping_rounds=100)
    # importance = clf_xgb.get_score(importance_type='gain')
    # importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    # print(importance)
    y_pred = clf_xgb.predict(test_set)
    y_pred = np.expm1(y_pred)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print('xgboost RMSLE: ', RMSLE)
    print('xgboost RMSE: ', RMSE)
    wandb.log({"RMSLE": RMSLE, "RMSE": RMSE})

if __name__ == '__main__':

    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'Val Accuracy', 'goal': 'maximize'},
        'parameters': {
            'alpha': {'values': [0,1,2]},
            'learning_rate': {'values': [0.1, 0.2, 0.05]},
            'max_depth': {'values': [3, 6, 9, 12]},
            'subsample': {'values': [1, 0.5, 0.3]},
            'min_child_weight':{'values': [1,2,3,4,5,6,7,8]},
            'colsample_bytree':{'values':[0,5,0.6,0.7,0.8,0.9]}
        }}

    # create new sweep
    sweep_id = wandb.sweep(sweep_config, entity="shovalz", project="revenue_predict_xgboost")

    # run the agent to execute the code
    wandb.agent(sweep_id, function=main)
