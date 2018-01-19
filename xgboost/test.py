import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data
# dtrain = xgb.DMatrix('data/agaricus.txt.train')
# dtest = xgb.DMatrix('data/agaricus.txt.test')
# specify parameters via map
# param = {'max_depth':500, 'eta':0.1, 'silent':0, 'objective':'binary:logistic' }
# num_round = 2000
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)

# print(preds)



# data importing
data = pd.read_csv('data/mnist/kaggle_train.csv')
data_x = data.drop('label',1).values
data_y = data['label'].values

num = len(data.index)
trainset = np.random.choice(num, int(num * 0.9), replace=False)

train_X = data_x[trainset,:].astype(np.float32)
valid_X = np.delete(data_x, trainset, axis=0).astype(np.float32)
train_y = data_y[trainset]
valid_y = np.delete(data_y, trainset, axis=0)
test_X = pd.read_csv('data/mnist/kaggle_test.csv').values

param = {'max_depth':5, 'eta':0.1, 'silent':0, 'num_boost_round':20, 'nfold':10, 'metrics': "auc"}
mdata = xgb.DMatrix(data = train_X, label = train_y)
model = xgb.train(param, dtrain = mdata)
xgb.plot_tree(model, num_trees=10)
plt.show()
print(model.get_fscore())

tdata = xgb.DMatrix(data = test_X)
pred = model.predict(data = tdata)
print(pred)
