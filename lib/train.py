# coding:utf-8
import cPickle
import time
import os
from functools import wraps

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import *

DIMENSION = 40
DATA_PATH = '..\\data\\raw\\'
EACH_PIC_OF_SOURCE = 60


def log_wrapper(content):
    """
    a decorate to print content & process time
    :param content:             string              a context to print
    :return:
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print content
            _ = time.time()
            result = func(*args, **kwargs)
            print 'Process time: %.2fs.' % (time.time() - _)
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
        self.dist_metric = self.dist_eclud

    @log_wrapper('Start loading image...')
    def loadimags(self, path):
        """
        look for all image folder by folder
        :param path:            string          the path of source images
        :return:
        """
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                sub_path = os.path.join(dirname, subdirname)
                count = 0
                for filename in os.listdir(sub_path):
                    if count < EACH_PIC_OF_SOURCE:
                        im = Image.open(os.path.join(sub_path, filename))
                        im = im.convert("L")
                        self.X.append(np.asarray(im, dtype=np.uint8))
                        self.y.append(subdirname)
                        count += 1
                    else:
                        break

    def gen_row_matrix(self):
        self.Mat = np.empty((0, self.X[0].size), dtype=self.X[0].dtype)
        for row in self.X:
            self.Mat = np.vstack((self.Mat, np.asarray(row).reshape(1, -1)))

    def pca(self, k=DIMENSION):
        """
        thr PCA algorithm
        :param k:               int       the aim dimension to reduce
        :return:
        """
        self.gen_row_matrix()
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
        self.pca()
        for xi in self.X:
            self.projections.append(self.project(xi.reshape(1, -1)))

    def dist_eclud(self, vecA, vecB):
        return linalg.norm(vecA - vecB) + self.eps

    def cos_sim(self, vecA, vecB):
        return (dot(vecA, vecB.T) / ((linalg.norm(vecA) * linalg.norm(vecB)) + self.eps))[0, 0]

    def project(self, XI):
        if self.mu is None:
            return np.dot(XI, self.eig_vect)
        return np.dot(XI - self.mu, self.eig_vect)

    @staticmethod
    def subplot(title, images):
        fig = plt.figure()
        fig.text(.5, .95, title, horizontalalignment='center')
        for i in xrange(len(images)):
            ax0 = fig.add_subplot(4, 4, (i + 1))
            plt.imshow(asarray(images[i]), cmap="gray")
            plt.xticks([]), plt.yticks([])
        plt.show()

    def predict(self, XI):
        """
        :param XI:          PIL.Image       the image to predict
        :return:            string          the name of image which is predicted
        """
        minDist = np.finfo('float').max
        minClass = -1
        Q = self.project(XI.reshape(1, -1))
        for i in xrange(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]
        return minClass


@log_wrapper('Start dumping...')
def save_instance(instance, path=r"..\data\trained"):
    """
    to dump the result
    :param instance:        Eigenfaces      a Eigenfaces object to dump
    :param path:            string          the dumped data path
    :return:
    """
    np.savez(os.path.join(path, 'mist'), instance.Mat, instance.eig_v, instance.eig_vect, instance.mu)
    np.savez(os.path.join(path, 'X'), [i for i in instance.X])
    np.savez(os.path.join(path, 'projections'), [i for i in instance.projections])
    cPickle.dump(instance.y, open(os.path.join(path, 'y.plk'), "wb"))


@log_wrapper('Start loading...')
def load_instance(path=r"..\data\trained"):
    """
    to load the result
    :param path:             string          the dumped data path
    :return:                Eigenfaces
    """
    _ = Eigenfaces()
    _.y = cPickle.load(open(os.path.join(path, 'y.plk'), "rb"))
    mist = np.load(os.path.join(path, 'mist.npz'))
    _.Mat, _.eig_v, _.eig_vect, _.mu = mist['arr_0'], mist['arr_1'], mist['arr_2'], mist['arr_3']
    _.X = np.load(os.path.join(path, 'X.npz'))['arr_0']
    _.projections = np.load(os.path.join(path, 'projections.npz'))['arr_0']
    return _


def display(ef):
    """
    display the eigen face
    :param ef:              Eigenfaces          a Eigenfaces objects had trained
    :return:
    """
    E = []
    X = mat(zeros((10, 10304)))
    for i in xrange(16):
        X = ef.Mat[i * EACH_PIC_OF_SOURCE:(i + 1) * EACH_PIC_OF_SOURCE, :].copy()
        X = X.mean(axis=0)
        imgs = X.reshape(200, 200)
        E.append(imgs)

    ef.subplot(title="My Eigen Facedatabase", images=E)


def train():
    """
    train and display
    :return:                Eigenfaces          a Eigenfaces objects
    """
    ef = Eigenfaces()
    ef.loadimags(DATA_PATH)
    ef.compute()

    save_instance(ef)

    return ef


if __name__ == '__main__':
    ef = train()
    display(ef)
