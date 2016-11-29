# coding:utf-8
import os
import time
from collections import Counter
from functools import partial

import cv2
import numpy

from train import load_instance

DATA_PATH = r'..\data\raw'
FACE_CASCADE=r'..\data\cascades\haarcascade_frontalface_default.xml'


class CaptureManager(object):
    def __init__(self, capture, preview_window_manager=None, should_mirror_preview=False):
        """
        :param capture:                                           a cv2.VideoCapture object
        :param preview_window_manager:        WindowManager       the windows
        :param should_mirror_preview:         Bool                is reversal the frame?
        :return:
        """
        self.previewWindowManager = preview_window_manager
        # mirror preview(not in file,just preview)
        self.shouldMirrorPreview = should_mirror_preview
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
            if callback is not None:
                _, frame = self._capture.retrieve()
                self._frame = callback(frame)
            else:
                _, self._frame = self._capture.retrieve()

        # get FPS
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            time_elapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / time_elapsed
        self._framesElapsed += 1

        return self._frame

    def draw_windows(self):
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirrored_frame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirrored_frame)
            else:
                self.previewWindowManager.show(self._frame)


class WindowManager(object):
    def __init__(self, window_name, keypress_callback=None):
        self.keypressCallback = keypress_callback
        self._windowName = window_name
        self._isWindowCreated = False

    @property
    def is_window_created(self):
        return self._isWindowCreated

    def create_window(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def process_events(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)


class Dector(object):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE)

    def __init__(self, mode, people_name=None):
        self._windowManager = WindowManager('Cameo', self.on_keypress)
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
                path_ = os.path.join(os.path.join(DATA_PATH, self._people_name), '%s.pgm' % str(time.time()))
                cv2.imwrite(path_, f)
        return frame

    def run(self):
        """begin loop"""
        self._windowManager.create_window()
        particle_face_detect = partial(self.face_solution, mode=self._mode)

        while self._windowManager.is_window_created:
            with self._captureManager:
                self._captureManager.capture(callback=particle_face_detect)
                self._captureManager.draw_windows()
            self._windowManager.process_events()

            self.print_result()

    def on_keypress(self, keycode):
        """
        handle pressed key
        escape -> exit
        """
        if keycode == 27:  # escape
            self._windowManager.destroy_window()


if __name__ == "__main__":
    Dector(mode='rec', people_name='Teacher').run()
