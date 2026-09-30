from __future__ import annotations

from typing import Iterable

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1e-12 else vector / norm


def mean_embedding(vectors: Iterable[np.ndarray]) -> np.ndarray | None:
    vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
    if not vectors:
        return None
    return l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))


class AppearanceEmbedder:
    """Full-person appearance encoder; no face recognition is performed."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config

        self.torch = torch
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.model_name = model_name
        except Exception:
            fallback = "resnet50.a1_in1k"
            self.model = timm.create_model(fallback, pretrained=True, num_classes=0)
            self.model_name = fallback

        self.model.eval().to(self.device)
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)

    def extract(self, bgr_crop: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = bgr_crop[:, :, ::-1]
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            features = self.model(tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.flatten(1)[0].detach().float().cpu().numpy()
        return l2_normalize(features)
