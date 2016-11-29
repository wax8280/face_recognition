# coding:utf-8

import os

import PIL.Image as Image
import numpy as np

from train import load_instance, EACH_PIC_OF_SOURCE

DATA_PATH = r'..\data\raw'
TEST_COUNT = 200


class Test(object):
    def __init__(self):
        self.X = []
        self.y = []
        self._ef = load_instance()
        self.load_image(DATA_PATH)

    def load_image(self, path):
        for dirname, dirnames, filenames in os.walk(DATA_PATH):
            for subdirname in dirnames:
                sub_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(sub_path)[EACH_PIC_OF_SOURCE:EACH_PIC_OF_SOURCE + TEST_COUNT]:
                    im = Image.open(os.path.join(sub_path, filename))
                    im = im.convert("L")
                    self.X.append(np.asarray(im, dtype=np.uint8))
                    self.y.append(subdirname)

    def test(self):
        d = {}
        for index, name in enumerate(self.y):
            if not d.has_key(name):
                d.update({name: {'T': 0, 'F': 0}})

            predict_result = self._ef.predict(self.X[index])
            if predict_result == name:
                d[name]['T'] += 1
            else:
                d[name]['F'] += 1

        return d

if __name__ == '__main__':
    test = Test()
    print test.test()
