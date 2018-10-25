#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import numpy as np
import math

from numpy import genfromtxt
from sklearn.model_selection import KFold

def read_dataset(filePath,delimiter=','):
  data = genfromtxt(filePath, delimiter=delimiter)
  data = np.delete(data, (0), axis=0)
  data = np.delete(data, (0, 9), axis=1)
  labels = data[:,0] # category as y
  features = data[:, [2,3,7]] # reviews, size, genre as x
  return features, labels

def feature_normalize(dataset):
  mu = np.mean(dataset,axis=0)
  sigma = np.std(dataset,axis=0)
  return (dataset - mu)/sigma

def append_bias_reshape(features,labels):
  n_training_samples = features.shape[0]
  n_dim = features.shape[1]
  f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
  l = np.reshape(labels,[n_training_samples,1])
  return f, l

features,labels = read_dataset('../dataset.preprocessed/cleaned.csv')
normalized_features = feature_normalize(features)
f, l = append_bias_reshape(normalized_features,labels)
n_dim = f.shape[1]

rnd_indices = np.random.rand(len(f)) < 0.70

train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[~rnd_indices]
test_y = l[~rnd_indices]

k = 4
batch_size=1000

# Placeholders
X_data_train = tf.placeholder(shape=[None, n_dim], dtype=tf.float32)
X_data_test = tf.placeholder(shape=[None, n_dim], dtype=tf.float32)
Y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare distance metric
# L1
distance = tf.reduce_sum(tf.abs(tf.subtract(X_data_train, tf.expand_dims(X_data_test,1))), axis=2)

# Predict: Get min distance index (Nearest neighbor)
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

top_k_yvals = tf.gather(Y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), axis=[1])

# Calculate MSE
rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(prediction, Y_target_test))))


# Calculate how many loops over training data
num_loops = int(np.ceil(len(test_x)/batch_size))

with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  results = []
  for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size,len(train_x))

    batch_x = test_x[min_index:max_index]
    batch_y = test_y[min_index:max_index]

    predictions = session.run(prediction, feed_dict={X_data_train: train_x, X_data_test: batch_x,
      Y_target_train: train_y, Y_target_test: batch_y})

    batch_rmse = session.run(rmse, feed_dict={X_data_train: train_x, X_data_test: batch_x,
      Y_target_train: train_y, Y_target_test: batch_y})

    results.append(batch_rmse)

  print("Batch rmse: %s" % results)

  # fig, ax = plt.subplots()
  # ax.scatter(test_y, pred_y)
  # ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
  # ax.set_xlabel('Measured')
  # ax.set_ylabel('Predicted')
  # fig.savefig('pred.png')
  # plt.show()
