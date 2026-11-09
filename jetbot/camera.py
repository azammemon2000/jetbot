import traitlets
from traitlets.config.configurable import SingletonConfigurable
import atexit
import cv2
import threading
import numpy as np

class Camera(SingletonConfigurable):
    """
    Hardware-accelerated CSI Camera for NVIDIA Jetson Nano using GStreamer nvarguscamerasrc.
    """
    value = traitlets.Any()
    width = traitlets.Integer(default_value=224).tag(config=True)
    height = traitlets.Integer(default_value=224).tag(config=True)
    fps = traitlets.Integer(default_value=30).tag(config=True)
    capture_width = traitlets.Integer(default_value=1280).tag(config=True)
    capture_height = traitlets.Integer(default_value=720).tag(config=True)
    capture_device = traitlets.Integer(default_value=0).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__(*args, **kwargs)
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._running = False
        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            re, image = self.cap.read()
            if re:
                self.value = image
                self.start()
            else:
                self.cap = None
        except Exception:
            self.cap = None
        atexit.register(self.stop)

    def _gst_str(self):
        return (
            f"nvarguscamerasrc sensor-id={self.capture_device} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.capture_width}, height=(int){self.capture_height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, width=(int){self.width}, height=(int){self.height}, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        )

    def start(self):
        if self._running or self.cap is None:
            return
        self._running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def _capture_frames(self):
        while self._running:
            re, image = self.cap.read()
            if re:
                self.value = image

    def stop(self):
        self._running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
