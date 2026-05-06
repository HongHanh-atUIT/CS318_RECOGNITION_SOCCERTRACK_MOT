import torch
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
import cv2


class Detector:
    """
    Phát hiện cầu thủ trên sân với 2 backend:
      - yolo26 : conf=0.4, iou=0.5, imgsz=1600
      - rf_detr: placeholder
    """

    def __init__(self, backend: str, model_path: str):
        """
        Args:
            backend    : "yolo26" | "rf_detr"
            model_path : đường dẫn tới file weight
        """
        self.backend = backend
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        if backend == "yolo26":
            self._load_yolo(model_path)
        elif backend == "rf_detr":
            self._load_detr(model_path)

    # ------------------------------------------------------------------ #
    #  Load models                                                         #
    # ------------------------------------------------------------------ #

    def _load_yolo(self, model_path: str):
        self.model = YOLO(model_path).to(self.device)
        self.conf = 0.4
        self.iou = 0.5
        self.imgsz = 1600
        print(f"[Detector] YOLO loaded on {self.device}")

    def _load_detr(self, model_path: str):
        from rfdetr import RFDETRMedium
        self.model      = RFDETRMedium(
            num_classes=1,
            pretrain_weights=model_path,
            resolution=576,
        )
        self.conf       = 0.4
        self.detr_size  = 576  
        print(f"[Detector] RF-DETR loaded")

    # ------------------------------------------------------------------ #
    #  Detect                                                              #
    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Args:
            image: ảnh BGR numpy array (H, W, 3)
        Returns:
            list of [x1, y1, x2, y2, score] với top left, bottom right
        """
        if self.backend == "yolo26":
            return self._detect_yolo(image)
        elif self.backend == "rf_detr":
            return self._detect_detr(image)

    def _detect_yolo(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
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
    
    def _detect_detr(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        orig_h, orig_w = image.shape[:2]

        img_resized = cv2.resize(image, (self.detr_size, self.detr_size))

        tmp_path = "__tmp_detr_input__.jpg"
        cv2.imwrite(tmp_path, img_resized)

        det   = self.model.predict(tmp_path, threshold=self.conf)
        preds = det.xyxy        
        confs = det.confidence

        scale_x = orig_w / self.detr_size
        scale_y = orig_h / self.detr_size

        boxes = []
        for i in range(len(preds)):
            x1, y1, x2, y2 = preds[i]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            conf = float(confs[i])
            boxes.append([x1, y1, x2, y2, conf])

        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return boxes