"""
detector.py — YOLOv5 detector (hub + ultralytics).
"""

import torch
import numpy as np
from typing import List, Tuple


class Detector:

    def __init__(self, backend: str, model_path: str,
                 conf: float = 0.4,    
                 iou: float = 0.5,    
                 imgsz: int = 1600,     
                 device: str = None):
        self.backend = backend
        self.device  = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = None
        self.conf    = conf
        self.iou     = iou
        self.imgsz   = imgsz

        if backend == "yolov5_hub":
            self._load_yolov5_hub(model_path)
        elif backend == "yolov5":
            self._load_yolov5_ultralytics(model_path)
        else:
            raise ValueError(
                f"Backend '{backend}' không hỗ trợ. Dùng 'yolov5' hoặc 'yolov5_hub'."
            )

    def _load_yolov5_hub(self, model_path: str):
        import warnings, logging
        logging.getLogger("yolov5").setLevel(logging.ERROR)
        YOLOV5_REPO = r"D:\UITs subject\Năm 3\Nhận dạng\SoccerMOT_Baseline\yolov5"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.hub.load(
                YOLOV5_REPO, "custom", path=model_path,
                source="local", force_reload=False, verbose=False,
            )
        self.model.to(self.device)
        self.model.conf = self.conf
        self.model.iou  = self.iou
        self.model.eval()
        print(f"[Detector] YOLOv5-hub loaded | model={model_path} | device={self.device} | conf={self.conf} iou={self.iou} imgsz={self.imgsz}")

    def _load_yolov5_ultralytics(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path).to(self.device)
        print(f"[Detector] YOLOv5-ultralytics loaded | model={model_path} | device={self.device} | conf={self.conf} iou={self.iou} imgsz={self.imgsz}")

    def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        if self.backend == "yolov5_hub":
            return self._detect_hub(image)
        else:
            return self._detect_ultralytics(image)

    def _detect_hub(self, image: np.ndarray) -> List[Tuple]:
        import cv2
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=self.imgsz)
        boxes   = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append([x1, y1, x2, y2, float(conf)])
        return boxes

    def _detect_ultralytics(self, image: np.ndarray) -> List[Tuple]:
        result = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            save=False,
            verbose=False,
        )[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append([x1, y1, x2, y2, conf])
        return boxes

    def set_conf(self, conf: float):
        self.conf = conf
        if self.backend == "yolov5_hub" and self.model is not None:
            self.model.conf = conf
        print(f"[Detector] conf updated → {conf}")