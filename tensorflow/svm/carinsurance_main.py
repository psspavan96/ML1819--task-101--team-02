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
  labels = data[:,7] # Carloan as y
  labels[labels == 0] = -1
  features = data[:, [5]] # Balance as x
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

features,labels = read_dataset('../../dataset.preprocessed/carinsurance/cleaned.csv')
normalized_features = feature_normalize(features)
f, l = append_bias_reshape(normalized_features,labels)
n_dim = f.shape[1]

rnd_indices = np.random.rand(len(f)) < 0.70

train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[~rnd_indices]
test_y = l[~rnd_indices]

learning_rate = 0.01
lambd = tf.constant([0.1])
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([n_dim,1]))
b = tf.Variable(tf.zeros([1]))

# SVM
# y_ = tf.nn.softmax(tf.matmul(X,W))
y_ = tf.matmul(X,W) + b
regularization_loss = 0.5*tf.reduce_sum(tf.square(W))
hinge_loss = tf.reduce_sum(tf.maximum(0., 1. - Y*y_))
cost = tf.add(regularization_loss, tf.multiply(lambd, hinge_loss))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
cost_history = np.empty(shape=[1],dtype=float)


def run_train(session, train_x, train_y):
  global optimizer
  global cost
  global X
  global Y
  global training_epochs
  global cost_history

  for epoch in range(training_epochs):
    session.run(optimizer,feed_dict={X:train_x,Y:train_y})
    cost_history = np.append(cost_history,session.run(cost,feed_dict={X: train_x,Y: train_y}))

def cross_validate(session, split_size=5):
  global train_x
  global train_y
  global y_
  global X
  global Y

  results = []
  kf = KFold(n_splits=split_size, shuffle=True)
  for train_idx, val_idx in kf.split(train_x, train_y):
    _train_x = train_x[train_idx]
    _train_y = train_y[train_idx]
    val_x = train_x[val_idx]
    val_y = train_y[val_idx]
    run_train(session, _train_x, _train_y)

    predicted_class = tf.sign(y_)
    correct_prediction = tf.equal(Y, predicted_class)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    results.append(session.run(accuracy, feed_dict={X: val_x, Y: val_y}))
  return results

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  result = cross_validate(session)
  # print("Cross-validation result: %s" % result)

  predicted_class = tf.sign(y_)
  correct_prediction = tf.equal(Y, predicted_class)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # print("Test accuracy: %f" % session.run(accuracy, feed_dict={X: test_x, Y: test_y}))
  print("accuracy,%f" % session.run(accuracy, feed_dict={X: test_x, Y: test_y}))

