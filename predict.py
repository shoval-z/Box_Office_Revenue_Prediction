import argparse
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import os
import sys
from main import data_prep


def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1 / len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

# # read extra files:
# # trained model
xgb_model_loaded = pickle.load(open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'data/xgb_model.pkl'), "rb"))
# # min-max scaler trained on the train set
scaler = pickle.load(open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'data/min_max_scaler.pkl'), "rb"))

# creat the test set- create feature
x_test, y_test = data_prep(data, mode='test')
# sanity check
x_test = x_test.fillna(0)
new_test_x = x_test.drop('id', axis=1)
new_test_x = scaler.transform(new_test_x)
test_set = xgb.DMatrix(new_test_x)
y_pred = xgb_model_loaded.predict(test_set)
y_pred = np.expm1(y_pred)
# sanity check
y_pred = np.maximum(y_pred,np.zeros(len(y_pred)))

prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = x_test['id']
prediction_df['revenue'] = y_pred

rmsle = rmsle(y_test, prediction_df['revenue'])
print("RMSLE is: {:.6f}".format(rmsle))

####
prediction_df.to_csv("prediction.csv", index=False, header=False)
