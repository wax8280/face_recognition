# coding:utf-8
import cPickle
import os
import time
from functools import wraps

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import *

DIMENSION = 40
DATA_PATH = '..\\data\\raw\\'
EACH_PIC_OF_SOURCE = 300


def log_wrapper(content):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print content
            _ = time.time()
            result = func(*args, **kwargs)
            # print time.time()-_
            print 'Use %.2fs.' % (time.time() - _)
            return result

        return wrapper

    return decorate


class Eigenfaces(object):
    """docstring for Eigenfaces"""

    def __init__(self):
        self.eps = 1.0e-16
        self.X = []
        self.y = []
        self.Mat = []
        self.eig_v = 0
        self.eig_vect = 0
        self.mu = 0
        self.projections = []
        self.dist_metric = self.distEclud

    @log_wrapper('Start loading image...')
    def loadimags(self, path):
        classlabel = 0
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                sub_path = os.path.join(dirname, subdirname)
                count = 0
                for filename in os.listdir(sub_path):
                    if count < EACH_PIC_OF_SOURCE:
                        im = Image.open(os.path.join(sub_path, filename))
                        im = im.convert("L")
                        self.X.append(np.asarray(im, dtype=np.uint8))
                        self.y.append(classlabel)
                        count += 1
                    break
                classlabel += 1

    def genRowMatrix(self):
        self.Mat = np.empty((0, self.X[0].size), dtype=self.X[0].dtype)
        for row in self.X:
            self.Mat = np.vstack((self.Mat, np.asarray(row).reshape(1, -1)))

    def PCA(self, k=DIMENSION):
        self.genRowMatrix()
        [n, d] = shape(self.Mat)
        if k > n:
            k = n
        self.mu = self.Mat.mean(axis=0)
        self.Mat = self.Mat - self.mu
        if n > d:
            XTX = np.dot(self.Mat.T, self.Mat)
            [self.eig_v, self.eig_vect] = linalg.eigh(XTX)
        else:
            XTX = np.dot(self.Mat, self.Mat.T)
            [self.eig_v, self.eig_vect] = linalg.eigh(XTX)
        self.eig_vect = np.dot(self.Mat.T, self.eig_vect)
        for i in xrange(n):
            self.eig_vect[:, i] = self.eig_vect[:, i] / linalg.norm(self.eig_vect[:, i])
        idx = np.argsort(-self.eig_v)
        self.eig_v = self.eig_v[idx]
        self.eig_vect = self.eig_vect[:, idx]
        self.eig_v = self.eig_v[0:k].copy()
        self.eig_vect = self.eig_vect[:, 0:k].copy()

    @log_wrapper('Start Compute...')
    def compute(self):
        self.PCA()
        for xi in self.X:
            self.projections.append(self.project(xi.reshape(1, -1)))

    def distEclud(self, vecA, vecB):
        return linalg.norm(vecA - vecB) + self.eps

    def cosSim(self, vecA, vecB):
        return (dot(vecA, vecB.T) / ((linalg.norm(vecA) * linalg.norm(vecB)) + self.eps))[0, 0]

    def project(self, XI):
        if self.mu is None:
            return np.dot(XI, self.eig_vect)
        return np.dot(XI - self.mu, self.eig_vect)

    def subplot(self, title, images):
        fig = plt.figure()
        fig.text(.5, .95, title, horizontalalignment='center')
        for i in xrange(len(images)):
            ax0 = fig.add_subplot(4, 4, (i + 1))
            plt.imshow(asarray(images[i]), cmap="gray")
            plt.xticks([]), plt.yticks([])
        plt.show()

    # @log_wrapper('Start predict...')
    def predict(self, XI):
        minDist = np.finfo('float').max
        minClass = -1
        Q = self.project(XI.reshape(1, -1))
        for i in xrange(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]
        return minClass


@log_wrapper('Start dump...')
def save_instance(instance, path=r"..\data\trained"):
    np.savez(path + r'\mist', instance.Mat, instance.eig_v, instance.eig_vect, instance.mu)
    np.savez(path + r'\X', [i for i in instance.X])
    np.savez(path + r'\projections', [i for i in instance.projections])
    cPickle.dump(instance.y, open(path + r'\y.plk', "wb"))


def load_instance(path=r"..\data\trained"):
    _ = Eigenfaces()
    _.y = cPickle.load(open(path + r'\y.plk', "rb"))
    mist = np.load(path + r'\mist.npz')
    _.Mat, _.eig_v, _.eig_vect, _.mu = mist['arr_0'], mist['arr_1'], mist['arr_2'], mist['arr_3']
    _.X = np.load(path + r'\X.npz')['arr_0']
    _.projections = np.load(path + r'\projections.npz')['arr_0']
    return _


def display(ef):
    E = []
    X = mat(zeros((10, 10304)))
    for i in xrange(16):
        X = ef.Mat[i * EACH_PIC_OF_SOURCE:(i + 1) * EACH_PIC_OF_SOURCE, :].copy()
        X = X.mean(axis=0)
        imgs = X.reshape(200, 200)
        E.append(imgs)

    ef.subplot(title="AT&T Eigen Facedatabase", images=E)


def train():
    ef = Eigenfaces()
    ef.dist_metric = ef.distEclud

    ef.loadimags(DATA_PATH)
    ef.compute()

    save_instance(ef)
    display(ef)


if __name__ == '__main__':
    train()