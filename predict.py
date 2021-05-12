import argparse
import numpy as np
import pandas as pd
import pickle
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
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

#####read files
xgb_model_loaded = pickle.load(open('xgb_model.pkl', "rb"))
object_dict = pickle.load(open('relevant_data.pkl', "rb"))
scaler = pickle.load(open('min_max_scaler.pkl', "rb"))

x_test, y_test = data_prep(data, mode='test')
new_test_x = scaler.transform(x_test)
y_pred = xgb_model_loaded.predict(new_test_x)
y_pred = np.expm1(y_pred)
y_pred = [max(item, 0) for item in y_pred]
rmsle = rmsle(y_test,y_pred)
print('RMSLE:', rmsle)

prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = x_test['id']
prediction_df['revenue'] = y_pred
####

prediction_df.to_csv("prediction.csv", index=False, header=False)




#
# ### Example - Calculating RMSLE
# res = rmsle(data['revenue'], prediction_df['revenue'])
# print("RMSLE is: {:.6f}".format(res))


