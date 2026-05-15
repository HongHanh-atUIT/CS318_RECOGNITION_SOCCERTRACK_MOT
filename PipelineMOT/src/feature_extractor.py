import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from osnet import osnet_x1_0


class FeatureExtractor:
    """
    Trích xuất đặc trưng ảnh crop người (ReID) với 3 backend:
      - osnet           : 512-dim vector (ImageNet pretrained)
      - solider         : 768-dim vector (placeholder)
      - color_histogram : 512-dim HSV histogram, không cần model
    """

    _OSNET_TRANSFORM = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    _SOLIDER_TRANSFORM = T.Compose([
        T.Resize((384, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])

    def __init__(
        self,
        backend: str = "color_histogram",
        solider_model_path: str = None,
        solider_arch: str = "swin_small",
        solider_semantic_weight: float = 0.2,
    ):
        self.backend = backend
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        if backend == "osnet":
            self._load_osnet()
        elif backend == "solider":
            self._load_solider(solider_model_path, solider_arch, solider_semantic_weight)

    # ------------------------------------------------------------------ #
    #  Load models                                                         #
    # ------------------------------------------------------------------ #

    def _load_osnet(self):
        self.model = osnet_x1_0(num_classes=1, pretrained=True)
        self.model.eval().to(self.device)
        print(f"[FeatureExtractor] OSNet loaded on {self.device}")

    def _load_solider(self, model_path: str, arch: str, semantic_weight: float):
        from swin_transformer import (
            swin_tiny_patch4_window7_224,
            swin_small_patch4_window7_224,
            swin_base_patch4_window7_224,
        )
        assert model_path is not None
        assert Path(model_path).exists(), f"Không tìm thấy: {model_path}"

        arch_map = {
            "swin_tiny" : swin_tiny_patch4_window7_224,
            "swin_small": swin_small_patch4_window7_224,
            "swin_base" : swin_base_patch4_window7_224,
        }

        swin = arch_map[arch](
            img_size=(384, 128),
            convert_weights=False,
            semantic_weight=semantic_weight,
        )

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        new_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("backbone."):
                new_state_dict[k[9:]] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = swin.load_state_dict(new_state_dict, strict=False)
        print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        swin.eval()
        swin.to(self.device)

        self.model    = swin
        self.feat_dim = swin.num_features[-1]
        print(f"[FeatureExtractor] SOLIDER ({arch}) loaded on {self.device}")
        print(f"[FeatureExtractor] Feature dim: {self.feat_dim}")
        
    # ------------------------------------------------------------------ #
    #  Single-image extraction (giữ lại để tương thích ngược)             #
    # ------------------------------------------------------------------ #

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Trích xuất đặc trưng từ 1 ảnh crop.
        Args:
            crop: ảnh BGR numpy array (H, W, 3)
        Returns:
            feature vector 1D float32
        """
        if self.backend == "color_histogram":
            return self._extract_color_histogram(crop)
        return self._extract_deep(crop)

    # ------------------------------------------------------------------ #
    #  Batch extraction (MỚI – dùng trong pipeline chính)                 #
    # ------------------------------------------------------------------ #

    # batch_size mặc định theo backend: SOLIDER nặng hơn OSNet
    _DEFAULT_BATCH_SIZE = {
        "osnet"           : 128,
        "solider"         : 64,
        "color_histogram" : 200,
    }

    def extract_batch(self, crops: list[np.ndarray], batch_size: int = None) -> list[np.ndarray]:
        """
        Trích xuất đặc trưng cho nhiều crop cùng lúc (batch inference).

        Args:
            crops      : list các ảnh BGR numpy array (H, W, 3)
            batch_size : số crops tối đa mỗi lần forward. None → dùng default
                         theo backend (osnet=64, solider=32).
                         Giới hạn này tránh CUDA OOM khi có quá nhiều detections.
        Returns:
            list các feature vector 1D float32, cùng thứ tự với crops.
        """
        if not crops:
            return []

        if self.backend == "color_histogram":
            return [self._extract_color_histogram(c) for c in crops]

        if batch_size is None:
            batch_size = self._DEFAULT_BATCH_SIZE.get(self.backend, 32)

        transform = (
            self._OSNET_TRANSFORM if self.backend == "osnet"
            else self._SOLIDER_TRANSFORM
        )

        all_feats = []
        for i in range(0, len(crops), batch_size):
            chunk = crops[i : i + batch_size]

            tensors = [
                transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                for c in chunk
            ]
            batch = torch.stack(tensors).to(self.device)   # (B, 3, H, W)

            with torch.no_grad():
                if self.backend == "osnet":
                    feats = self.model(batch)               # (B, 512)
                elif self.backend == "solider":
                    feats, _ = self.model(batch, semantic_weight=None)  # (B, D)

            feats = F.normalize(feats, p=2, dim=1)
            all_feats.extend(f.cpu().numpy().astype(np.float32) for f in feats)

        return all_feats

    # ------------------------------------------------------------------ #
    #  Internal single-image helpers                                       #
    # ------------------------------------------------------------------ #

    def _extract_deep(self, crop: np.ndarray) -> np.ndarray:
        transform = (
            self._OSNET_TRANSFORM if self.backend == "osnet"
            else self._SOLIDER_TRANSFORM
        )
        inp = self._preprocess(crop, transform)

        with torch.no_grad():
            if self.backend == "osnet":
                feat = self.model(inp)
            elif self.backend == "solider":
                return self._extract_solider(crop)

        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)
    
    def _extract_solider(self, crop: np.ndarray) -> np.ndarray:
        inp = self._preprocess(crop, self._SOLIDER_TRANSFORM)
        with torch.no_grad():
            feat, _ = self.model(inp, semantic_weight=None)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _preprocess(self, crop: np.ndarray, transform: T.Compose) -> torch.Tensor:
        """BGR numpy (H,W,3) → tensor batch (1,3,H,W)."""
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return transform(img).unsqueeze(0).to(self.device)

    @staticmethod
    def _extract_color_histogram(crop: np.ndarray) -> np.ndarray:
        """
        HSV histogram 512-dim từ dải giữa (1/3 width) của crop.
        bins: H=32, S=16, V=1 → 32×16×1 = 512, L2-normalized.
        """
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        W = crop.shape[1]
        strip = crop[:, W // 3: 2 * W // 3]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            histSize=[32, 16, 1],
            ranges=[0, 180, 0, 256, 0, 256],
        ).flatten().astype(np.float32)

        l2 = np.linalg.norm(hist)
        return hist / l2 if l2 > 0 else hist