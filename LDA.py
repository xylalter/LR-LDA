from statistics import mean
from telnetlib import SE
from tokenize import Double
from sklearn.datasets import load_breast_cancer
import math
import numpy as np
from sklearn.model_selection import KFold
data = load_breast_cancer()
x = data.data
y = data.target
def train(x, y):
    x0 = []
    x1 = []
    for i in range(len(y)):
        if y[i] == 0:
            x0.append(x[i])
        else:
            x1.append(x[i])
    x0 = np.array(x0)
    x1 = np.array(x1)
    u0 = np.mean(x0, axis=0)
    u1 = np.mean(x1, axis=0)
    Sw = sum(map(lambda x: np.dot((x - u0)[:, np.newaxis], [x - u0]), x0)) + sum(map(lambda x: np.dot((x - u1)[:, np.newaxis], [x - u1]), x1))
    Sb = np.dot((u0 - u1)[:, np.newaxis], [u0 - u1])
    # S = np.linalg.inv(Sw) * Sb
    # lamda = np.linalg.eig(S)
    # index = np.argmax(lamda[0])
    # w = lamda[1][:,index]
    w = np.dot(np.linalg.inv(Sw), (u0 - u1))
    cluster0 = np.dot(w, u0)
    cluster1 = np.dot(w, u1)
    mid = (cluster1 + cluster0) / 2.0
    if cluster0 < cluster1:
        myflag = False
    else:
        myflag = True
    return w, mid, myflag
def test(w, mid, myflag, x, y):
    y_hat = []
    for i in range(len(x)):
        tmp = np.dot(w, x[i])
        if tmp < mid:
            y_hat.append(int(myflag))
        else:
            y_hat.append(int(not myflag))
    return sum(map(lambda x, y: int(x == y), y_hat, y)) / len(y)
def cross_validation_LDA(x, y, k=5):
    kf = KFold(n_splits=k, random_state=None)
    W = []
    ACC = []
    SELF_ACC = []
    for train_idx, test_idx in kf.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w, mid, myflag = train(x_train, y_train)
        acc = test(w, mid, myflag, x_test, y_test)
        self_acc = test(w, mid, myflag, x_train, y_train)
        W.append(w)
        ACC.append(acc)
        SELF_ACC.append(self_acc)
    w_mean = np.mean(W, axis=0)
    acc_mean = np.mean(ACC)
    self_acc_mean  = np.mean(SELF_ACC)
    return w_mean, acc_mean
print(cross_validation_LDA(x, y))
    
    
