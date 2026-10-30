"""Thin wrapper around Ultralytics YOLO + BoT-SORT, shared by live mode and evaluation."""
import numpy as np
from ultralytics import YOLO


class BotSortTracker:
    def __init__(self, weights, device, imgsz=640, classes=None, det_conf=0.1):
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.classes = classes or None  # None = all classes
        # Keep low-confidence detections: BoT-SORT's second association pass uses them.
        self.det_conf = det_conf
        self.cfg = None

    def set_config(self, cfg_path):
        self.cfg = str(cfg_path)
        self.reset()

    def reset(self):
        """Drop all track state (a new predictor builds a fresh tracker on the next frame)."""
        self.model.predictor = None

    def warmup(self, frame):
        self.model.predict(frame, device=self.device, imgsz=self.imgsz, verbose=False)
        self.reset()

    def update(self, frame):
        """Track one BGR frame. Returns (ultralytics result, xyxy boxes, int ids, confidences)."""
        res = self.model.track(
            frame, persist=True, tracker=self.cfg, classes=self.classes, device=self.device,
            imgsz=self.imgsz, conf=self.det_conf, verbose=False,
        )[0]
        b = res.boxes
        if b is None or b.id is None or len(b) == 0:
            return res, np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
        return res, b.xyxy.cpu().numpy(), b.id.int().cpu().numpy(), b.conf.cpu().numpy()
