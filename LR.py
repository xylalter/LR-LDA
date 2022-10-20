from statistics import mean
from tokenize import Double
from sklearn.datasets import load_breast_cancer
import math
import numpy as np
from sklearn.model_selection import KFold 
data = load_breast_cancer()
x = data.data
y = data.target[:, np.newaxis]
def init_params(train_dim):
	w = np.zeros((train_dim,1))
	b = 0
	return w,b
def sigmoid(x):
    z=1/(1+np.exp(-x))
    return z

def logistics(X,y,w,b):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = sigmoid(np.dot(X,w) + b)
    loss = -1/num_train * np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)) 
    dw = np.dot(X.T,(y_hat-y))/num_train
    db = np.sum((y_hat-y))/num_train
    loss = np.squeeze(loss)
    return y_hat, loss, dw, db

def train(X, y, learning_rate=0.01, epochs=5000):
    loss_his = []
    w, b = init_params(X.shape[1])
    for i in range(epochs):
        y_hat, loss, dw, db = logistics(X, y, w, b)
        w = w -learning_rate*dw
        b = b -learning_rate*db
        loss_his.append(loss)
    params = {'w':w, 'b':b}
    grads = {'dw':dw,'db':db}
    return loss_his, params, grads

def predict(X, params):
    w = params['w']
    b = params['b']
    pre = sigmoid(np.dot(X, w) + b)
    for i in range(len(pre)):
        if pre[i]>0.5:
            pre[i]=1
        else:
            pre[i]=0
    return pre

def accuracy(x_train, y_train, x_test, y_test):
    loss, params, grads = train(x_train, y_train)
    pre = predict(x_test, params)
    for i in range(len(pre)):
        pre[i] = 1 - abs(pre[i]-y_test[i])
    return mean(pre.T[0]), params['w'] + params['b']

def cross_validation_logic(x, y, k=5):
    kf = KFold(n_splits=k, random_state=None)
    B = []
    ACC = []
    for train_idx, test_idx in kf.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        acc, beta = accuracy(x_train, y_train, x_test, y_test)
        B.append(beta)
        ACC.append(acc)
    beta_mean = np.mean(B, axis=0)
    acc_mean = np.mean(ACC)
    return beta_mean.T, acc_mean

print(cross_validation_logic(x, y))