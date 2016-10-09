# coding:utf-8

import time
from functools import partial

import cv2
import numpy


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        # 绘制窗口，bool
        self.previewWindowManager = previewWindowManager
        # 镜像旋转（在窗口中问不是在文件中），bool
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._enteredFrame = False
        self._frame = None
        self._startTime = None
        # 从开始到现在帧数
        self._framesElapsed = long(0)
        # OpenCV没办法获取FPS，如果需要可以用time.time()计算
        self._fpsEstimate = None

    def __enter__(self):
        """捕获下一帧，如果有的话"""
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._frame is None:
            self._enteredFrame = False
            return

        # 释放
        self._frame = None
        self._enteredFrame = False

    def capture(self, callback=None):

        if self._enteredFrame and self._frame is None:
            if callback != None:
                _, frame = self._capture.retrieve()
                self._frame = callback(frame)
            else:
                _, self._frame = self._capture.retrieve()

        # 获取FPS
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        return self._frame

    def draw_windows(self):
        # 绘制窗口
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

    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(1), self._windowManager, True)

    def face_detect(self, frame, is_generate=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = Dector.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if is_generate:
                f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                cv2.imwrite('../data/%s.pgm' % str(time.time()), f)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = Dector.eye_cascade.detectMultiScale(roi_gray, 1.3, 6, 0, (40, 40))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 2)
        return frame

    def run(self):
        """开始循环"""
        self._windowManager.createWindow()
        particle_face_detect = partial(self.face_detect, is_generate=True)

        while self._windowManager.isWindowCreated:
            with self._captureManager:
                self._captureManager.capture(callback=particle_face_detect)
                self._captureManager.draw_windows()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        处理案件
        escape -> 退出
        """
        if keycode == 27:  # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Dector().run()