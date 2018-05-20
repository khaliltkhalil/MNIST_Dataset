import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split


def y_to_hot_dot(y):
    #X = X.astype(float)
    #y = y.astype(int)

    nk = max(y)+1
    y_hotdot = np.zeros((y.shape[0],nk))
    for (i,x) in enumerate(y):
        y_hotdot[i,x] = 1
    return y_hotdot

def get_mini_batches(X,y,mini_batch_size,seed,shuffled = True):

    np.random.seed(seed)
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    if shuffled:
        X_shuffled = X[permutation, :]
        y_shuffled = y[permutation, :]
    else:
        X_shuffled = X
        y_shuffled = y

    mini_batches_list = []
    number_complete_mini_batches = int(math.floor(m/mini_batch_size))
    for i in range(number_complete_mini_batches):
        X_mini_batch = X_shuffled[i*mini_batch_size:mini_batch_size*(i+1), :]
        y_mini_batch = y_shuffled[i*mini_batch_size:mini_batch_size*(i+1), :]
        mini_batches_list.append((X_mini_batch, y_mini_batch))

    if m % mini_batch_size != 0:
        X_mini_batch = X_shuffled[number_complete_mini_batches*mini_batch_size:, :]
        y_mini_batch = y_shuffled[number_complete_mini_batches*mini_batch_size:, :]
        mini_batches_list.append((X_mini_batch, y_mini_batch))

    return mini_batches_list



def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0],"X")
    Y = tf.placeholder(tf.float32,[None, n_y], "Y")



    return X, Y


def initialize_parameters():

    W1 = tf.get_variable("W1",[3,3,1,16],tf.float32,tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [3, 3, 16, 32], tf.float32, tf.contrib.layers.xavier_initializer())
    # W3 = tf.get_variable("W3",[3,3,16,32],tf.float32,tf.contrib.layers.xavier_initializer())
    # W4 = tf.get_variable("W4", [3, 3, 32, 32], tf.float32, tf.contrib.layers.xavier_initializer())
    parameters = {"W1": W1,
                  "W2": W2,
                  # "W3": W3,
                  # "W4": W4
                  }

    return parameters

def forward_prop(X, parameters,is_training):

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    # W3 = parameters["W3"]
    # W4 = parameters["W4"]

    Z1 = tf.nn.conv2d(X,W1,[1,1,1,1],"SAME")
    Z1_hat = tf.layers.batch_normalization(Z1,axis=-1,training=is_training)
    A1 = tf.nn.relu(Z1_hat)
    Z2 = tf.nn.conv2d(A1, W2, [1, 1, 1, 1], "SAME")
    Z2_hat = tf.layers.batch_normalization(Z2, axis=-1, training=is_training)
    A2 = tf.nn.relu(Z2_hat)
    P2 = tf.nn.max_pool(A2,[1,4,4,1],[1,2,2,1],"SAME")

    # Z3 = tf.nn.conv2d(P2,W3,[1,1,1,1],"SAME")
    # A3 = tf.nn.relu(Z3)
    # Z4 = tf.nn.conv2d(A3, W4, [1, 1, 1, 1], "SAME")
    # A4 = tf.nn.relu(Z4)
    # P4 = tf.nn.max_pool(A4,[1,4,4,1],[1,2,2,1],"SAME")

    P = tf.contrib.layers.flatten(P2)
    Pd = tf.layers.dropout(P,rate=0.5,training= is_training)
    Z5 = tf.contrib.layers.fully_connected(Pd,10, activation_fn =None)
    return Z5

def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=Z3))

    return cost

def model(X_train, Y_train, X_test, Y_test,learning_rate=0.001, num_epoch=200,mini_batch_size=256):

    ops.reset_default_graph()
    Y_train_hot_dot = y_to_hot_dot(Y_train)
    m, n_H0, n_W0, n_C0 = X_train.shape
    n_y = Y_train_hot_dot.shape[1]
    X, Y,  = create_placeholders(n_H0,n_W0,n_C0,n_y)
    is_training = tf.placeholder(tf.bool)
    parameters = initialize_parameters()

    Z3= forward_prop(X, parameters, is_training)

    cost = compute_cost(Z3, Y)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    epoch_costs = []
    with tf.Session() as session:
        session.run(init)
        seed = 0
        for i in range(num_epoch):
            seed += 1
            mini_batches_list = get_mini_batches(X_train,Y_train_hot_dot,mini_batch_size,seed)
            j = 0
            for X_mini_batch, Y_mini_batch in mini_batches_list:
                mini_batch_cost = session.run(cost,feed_dict={X:X_mini_batch, Y:Y_mini_batch, is_training: True})
                session.run(optimizer,feed_dict={X:X_mini_batch, Y:Y_mini_batch, is_training: True})
                if j%10 == 0:
                    print("batch",j)
                j += 1
            print("epoch:", i, mini_batch_cost)
            epoch_costs.append(mini_batch_cost)

        # plt.plot(epoch_costs)
        # plt.show()

        Y_test_hot_dot = y_to_hot_dot(Y_test)
        y_pred_train = np.empty((Y_train.shape[0],0))
        y_pred_test = np.empty((Y_test.shape[0],0))
        mini_batches_list_train = get_mini_batches(X_train, Y_train_hot_dot, mini_batch_size, seed, shuffled=False)
        mini_batches_list_test = get_mini_batches(X_test, Y_test_hot_dot, mini_batch_size, seed, shuffled=False)
        i = 0
        for X_mini_batch, Y_mini_batch in mini_batches_list_train:
            y_pred_train_batch = session.run(tf.argmax(Z3, axis=1), feed_dict={X: X_mini_batch, is_training: True})
            if i%10 == 0:
                print("running prediction on training data, batch:",i)
            y_pred_train = np.append(y_pred_train, y_pred_train_batch)
            i += 1
        print("train accuracy", accuracy_score(Y_train, y_pred_train))

        i = 0
        for X_mini_batch, Y_mini_batch in mini_batches_list_test:
            y_pred_test_batch = session.run(tf.argmax(Z3, axis=1), feed_dict={X: X_mini_batch, is_training: False})
            if i % 10 == 0:
                print("running prediction on testing data, batch:", i)
            y_pred_test = np.append(y_pred_test, y_pred_test_batch)
            i += 1
        print("test accuracy", accuracy_score(Y_test, y_pred_test))

        # print("train accuracy",accuracy_score(Y_train,y_pred_train.T))
        #
        # y_pred_test = session.run(tf.argmax(Z3, axis=1), feed_dict={X: X_test, is_training: False})
        # print("test accuracy", accuracy_score(Y_test, y_pred_test.T))

digits = pd.read_csv("train.csv")
data = digits.values
X = data[:,1:]
m = X.shape[0]
y = data[:,0]
X = np.reshape(X,(m,28,28,1))
X = X/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model(X_train,y_train,X_test,y_test,learning_rate=0.001,num_epoch=50,mini_batch_size=256)












