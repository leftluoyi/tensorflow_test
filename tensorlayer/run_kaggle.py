import tensorflow as tf
import tensorlayer as tl
import pandas as pd
import numpy as np
import csv

# data importing
data = pd.read_csv('data/mnist/kaggle_test.csv')

print(data.shape)