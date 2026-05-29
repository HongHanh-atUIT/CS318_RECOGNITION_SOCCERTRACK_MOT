"""
feature_extractor.py — OSNet pretrained feature extractor.
Backend: "osnet" — trả về L2-normalized feature vector (512,).
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T


class FeatureExtractor:
    def __init__(
        self,
        backend:     str = "osnet",
        model_name:  str = "osnet_x1_0",
        device:      str = "auto",
        **kwargs,
    ):
        self.backend = backend.lower()

        if self.backend != "osnet":
            raise ValueError(f"Unknown backend '{backend}'. Only 'osnet' is supported.")

        import importlib.util, sys, os
        _osnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osnet.py")
        _spec = importlib.util.spec_from_file_location("bl_osnet", _osnet_path)
        _osnet_mod = importlib.util.module_from_spec(_spec)
        sys.modules["bl_osnet"] = _osnet_mod
        _spec.loader.exec_module(_osnet_mod)
        osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25 = (
            _osnet_mod.osnet_x1_0, _osnet_mod.osnet_x0_75,
            _osnet_mod.osnet_x0_5, _osnet_mod.osnet_x0_25,
        )

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else device

        _models = {
            "osnet_x1_0":  osnet_x1_0,
            "osnet_x0_75": osnet_x0_75,
            "osnet_x0_5":  osnet_x0_5,
            "osnet_x0_25": osnet_x0_25,
        }
        if model_name not in _models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(_models)}")

        print(f"[FeatureExtractor] Loading {model_name} pretrained ...")
        self.model = _models[model_name](num_classes=1000, pretrained=True, loss="softmax")
        self.model.eval().to(self.device)
        print(f"[FeatureExtractor] Ready on {self.device}")

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        crop : BGR numpy (H, W, 3)
        Returns: L2-normalized feature (512,) float32
        """
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        feat   = self.model(tensor)
        feat   = feat.cpu().numpy().squeeze()
        norm   = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm
        return feat.astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, crops: list) -> np.ndarray:
        """
        crops : list of BGR numpy arrays
        Returns: (N, 512) float32
        """
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        tensors, valid = [], []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensors.append(self._transform(rgb))
                valid.append(i)

        result = np.zeros((len(crops), 512), dtype=np.float32)
        if not tensors:
            return result

        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / (norms + 1e-8)

        for out_i, src_i in enumerate(valid):
            result[src_i] = feats[out_i]
        return result