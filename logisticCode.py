import numpy as np
import matplotlib.pyplot as plt
from DataPlot import *


def splitData(X, y):
    X_pos = X[y == 1]
    np.random.shuffle(X_pos)
    X_pos_train, X_pos_test = X_pos[:400], X_pos[400:]

    X_neg = X[y == 0]
    np.random.shuffle(X_neg)
    X_neg_train, X_neg_test = X_neg[:400], X_neg[400:]

    X_train = np.concatenate((X_pos_train, X_neg_train), axis=0)
    y_train = np.concatenate((np.ones((400, 1)), np.zeros((400, 1))), axis=0)

    X_test = np.concatenate((X_pos_test, X_neg_test), axis=0)
    y_test = np.concatenate((np.ones((94, 1)), np.zeros((106, 1))), axis=0)
    return X_train, y_train, X_test, y_test


class Logistic(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.X_aug = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)

    def sigmoid(self, weights):  # also is prediction
        s = 1.0 / (1.0 + np.exp(-np.dot(self.X_aug, weights)))
        s[s == 1] = 1 - 0.1e-15
        s[s == 0] = 0.1e-15
        return s

    def gradAscent(self, weights, learnRate):
        s = self.sigmoid(weights)
        gradient = np.dot(self.X_aug.T, np.subtract(self.labels, s))
        weights = weights + learnRate * gradient
        return weights

    def train(self, learnRate, iterations):
        weights = 1 * np.ones((self.X_aug.shape[1], 1))  # initial weights
        record = np.zeros((iterations, self.X_aug.shape[1]))
        for i in range(iterations):
            weights = self.gradAscent(weights, learnRate)
            record[i] = weights.T
        return weights, record

    def logLikelihood(self, record):
        ll = np.zeros(record.shape[0])
        for i in range(record.shape[0]):
            s = self.sigmoid(record[i].reshape(record[i].size, 1))
            ll_a = np.log(np.power(s, self.labels)) + np.log(np.power((1 - s), (1 - self.labels)))
            # ll_a = np.zeros(s.shape[0])
            # for j in range(s.shape[0]):
            # if (s[j]==1 and self.labels[j]==0) or (s[j]==0 and self.labels[j]==1):
            # ll_a[j] = -np.abs(np.dot(self.X_aug[j], record[i].reshape(record[i].size,1)))
            # else:
            # ll_a[j] = np.log(np.power(s[j], self.labels[j])) + np.log(np.power((1 - s[j]),(1-self.labels[j])))
            ll[i] = np.mean(ll_a)
        return ll


def sigmoid(X, weights): #also is prediction
    X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    s = 1.0 / (1.0 + np.exp(-np.dot(X_aug, weights)))
    s[s==1] = 1 - 0.1e-15
    s[s==0] = 0.1e-15
    return s

def gradient(X, y, weights):
    X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    s = sigmoid(X, weights)
    return np.dot(X_aug.T, np.subtract(y, s))

def gradAscent(X, y, learnRate, iterations):
    X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    weights = 1 * np.ones((X_aug.shape[1], 1))  # initial weights
    record = np.zeros((iterations, X_aug.shape[1]))
    for i in range(iterations):
        weights = weights + learnRate * gradient(X, y, weights)
        record[i] = weights.T
    return weights, record

def confMatrix(test, label, weights, thres):
    prediction = sigmoid(test, weights)
    prediction[prediction > thres] = 1
    prediction[prediction <= thres] = 0
    conf = np.zeros((2,2))
    conf[0, 0] = 1 - np.count_nonzero(prediction[label== 0]) / prediction[label== 0].size
    conf[0, 1] = np.count_nonzero(prediction[label== 0]) / prediction[label== 0].size
    conf[1, 0] = 1 - np.count_nonzero(prediction[label== 1]) / prediction[label== 1].size
    conf[1, 1] = np.count_nonzero(prediction[label== 1]) / prediction[label== 1].size
    return conf

# joint probability to compute MAP
def func(w, X, y, sigma):
    s = sigmoid(X, w)
    LogL = np.sum(np.log(np.power(s, y)) + np.log(np.power((1 - s),(1-y))))
    LogP = np.sum(-np.power(w, 2) / 2) * sigma
    return -LogL - LogP

# gradient of func
def grad(w, X, y, sigma):
    X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    s = sigmoid(X, w)
    gradient = np.dot(X_aug.T, np.subtract(y, s)) - w*sigma
    return -gradient


def auc(x, y):
    dx = np.diff(x)
    if np.any(dx < 0):
        # reorder the data point according to the x axis
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    area = np.trapz(y, x)
    return area


def plot_ROC(test, label, weights):
    roc = np.zeros((1000, 2))
    thres = np.linspace(1, 0, 1000)
    for i in range(1000):
        conf = confMatrix(test, label, weights, thres[i])
        roc[i, 0] = conf[0, 1]
        roc[i, 1] = conf[1, 1]
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.plot(roc[:, 0], roc[:, 1], 'r-', label='ROC curve (area = %0.3f)' % auc(roc[:, 0], roc[:, 1]))
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operator Characteristic(ROC)')
    plt.legend(loc='lower right')
    plt.show



class Logistic_MAP(Logistic):
    def gradAscent(self, weights, learnRate):
        s = self.sigmoid(weights)
        gradient = np.dot(self.X_aug.T, np.subtract(self.labels, s))
        weights = weights + learnRate * (gradient - weights)
        return weights

def sig(x):
    return 1.0 / (1.0 + np.exp(- x))

def confMatrix_prd(prediction, label, thres):
    #prediction = sigmoid(test, weights)
    prediction[prediction > thres] = 1
    prediction[prediction <= thres] = 0
    conf = np.zeros((2,2))
    conf[0, 0] = 1 - np.count_nonzero(prediction[label== 0]) / prediction[label== 0].size
    conf[0, 1] = np.count_nonzero(prediction[label== 0]) / prediction[label== 0].size
    conf[1, 0] = 1 - np.count_nonzero(prediction[label== 1]) / prediction[label== 1].size
    conf[1, 1] = np.count_nonzero(prediction[label== 1]) / prediction[label== 1].size
    return conf
