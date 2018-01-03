import tensorflow as tf
import tensorlayer as tl
import pandas as pd
import numpy as np
import csv

# data importing
data = pd.read_csv('data/mnist/kaggle_train.csv')
data_x = data.drop('label',1).values
data_y = data['label'].values


trainset = np.random.choice(42000, 37800, replace=False)


train_x = data_x[trainset,:].astype(np.float32)
valid_x = np.delete(data_x, trainset, axis=0)
train_y = data_y[trainset].astype(np.float32)
valid_y = np.delete(data_y, trainset, axis=0)

print(valid_x.shape)
print(valid_y.shape)
print(train_x.shape)
print(train_y.shape)