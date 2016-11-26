# coding:utf-8
import os
import time
from collections import Counter
from functools import partial

import cv2
import numpy

from train import load_instance

DATA_PATH = r'..\data\raw'


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        """
        :param capture:                                         a cv2.VideoCapture object
        :param previewWindowManager:        WindowManager       the windows
        :param shouldMirrorPreview:         Bool                is reversal the frame?
        :return:
        """
        self.previewWindowManager = previewWindowManager
        # 镜像旋转（在窗口中问不是在文件中），bool
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._enteredFrame = False
        self._frame = None
        self._startTime = None
        # count of frame from begin to now
        self._framesElapsed = long(0)
        # OpenCV cant get FPS，if need we can use time.time() to cal
        self._fpsEstimate = None

    def __enter__(self):
        """get next frame,if exist """
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._frame is None:
            self._enteredFrame = False
            return

        # release
        self._frame = None
        self._enteredFrame = False

    def capture(self, callback=None):

        if self._enteredFrame and self._frame is None:
            if callback != None:
                _, frame = self._capture.retrieve()
                self._frame = callback(frame)
            else:
                _, self._frame = self._capture.retrieve()

        # get FPS
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        return self._frame

    def draw_windows(self):
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)


class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)


class Dector(object):
    face_cascade = cv2.CascadeClassifier('../data/cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../data/cascades/haarcascade_eye.xml')

    def __init__(self, mode, people_name=None):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(1), self._windowManager, True)
        self._ef = load_instance()
        self._predict_result = []
        self._mode = mode
        self.count_to_print = 10

        self._people_name = people_name

        # make dir if not exit
        if self._people_name:
            if not os.path.exists(os.path.join(DATA_PATH, self._people_name)):
                os.mkdir(os.path.join(DATA_PATH, self._people_name))

    def print_result(self):
        """
        print the result
        :return:
        """
        if len(self._predict_result) > self.count_to_print:
            most = Counter(self._predict_result).most_common(1)
            print 'Is {} !!!'.format(str(most[0][0]))
            print 'Presentage: %.4f' % (float(most[0][1]) / len(self._predict_result) * 100) + '%'
            self._predict_result = []

    def face_recoginze(self, image):
        """
        face_recoginze
        :param image:               IMAGE           a image object to predict
        :return:
        """
        return self._ef.predict(image)

    def face_solution(self, frame, mode='rec'):
        """
        face detect & recogize or collect
        :param frame:               a               a source frame
        :param mode:                string          a string that set func to collect mode or recoginze mode
        :return:
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = Dector.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

            if mode == 'rec':
                self._predict_result.append(self.face_recoginze(f))
            elif mode == 'col':
                PATH_ = os.path.join(os.path.join(DATA_PATH, self._people_name), '%s.pgm' % str(time.time()))
                print PATH_
                cv2.imwrite(PATH_, f)
        return frame

    def run(self):
        """begin loop"""
        self._windowManager.createWindow()
        particle_face_detect = partial(self.face_solution, mode=self._mode)

        while self._windowManager.isWindowCreated:
            with self._captureManager:
                self._captureManager.capture(callback=particle_face_detect)
                self._captureManager.draw_windows()
            self._windowManager.processEvents()

            self.print_result()

    def onKeypress(self, keycode):
        """
        handle pressed key
        escape -> exit
        """
        if keycode == 27:  # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Dector(mode='col', people_name='1').run()
