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
  data = np.delete(data, (0), axis=1)
  labels = data[:,1] # ratings as y
  features = data[:, [2]] # reviews as x
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

features,labels = read_dataset('../../dataset.preprocessed/cleaned.csv')
normalized_features = feature_normalize(features)
f, l = append_bias_reshape(normalized_features,labels)
n_dim = f.shape[1]

rnd_indices = np.random.rand(len(f)) < 0.70

train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[~rnd_indices]
test_y = l[~rnd_indices]

learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([n_dim,1]))


# LR
y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

def run_train(session, train_x, train_y):
  global training_step
  global cost
  global X
  global Y
  global training_epochs
  global cost_history
  global learning_rate

  for epoch in range(training_epochs):
    session.run(training_step,feed_dict={X:train_x,Y:train_y})
    cost_history = np.append(cost_history,session.run(cost,feed_dict={X: train_x,Y: train_y}))

def cross_validate(session, split_size=5):
  global train_x
  global train_y
  global X
  global Y

  results = []
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_x, train_y):
    _train_x = train_x[train_idx]
    _train_y = train_y[train_idx]
    val_x = train_x[val_idx]
    val_y = train_y[val_idx]
    run_train(session, _train_x, _train_y)
    pred_y = session.run(y_, feed_dict={X: val_x})
    mse = tf.reduce_mean(tf.square(pred_y - val_y))
    results.append(session.run(mse, feed_dict={X: val_x, Y: val_y}))
  return results

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  result = cross_validate(session)
  print("Cross-validation result: %s" % result)
  pred_y = session.run(y_, feed_dict={X: test_x})
  mse = tf.reduce_mean(tf.square(pred_y - test_y))
  rmse = tf.sqrt(tf.reduce_mean(tf.square(pred_y - test_y)))
  print("Test rmse: %f" % session.run(rmse, feed_dict={X: test_x, Y: test_y}))
  # squared_difference

  fig, ax = plt.subplots()
  ax.scatter(test_y, pred_y)
  ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  fig.savefig('pred.png')
