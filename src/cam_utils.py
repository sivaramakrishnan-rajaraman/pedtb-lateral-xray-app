# src/cam_utils.py

from __future__ import annotations
from typing import Tuple, Optional, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import cv2

# PyTorch-Grad-CAM (Jacob Gil)
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM,
    LayerCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    import matplotlib.cm as cm
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# ---------------------------
# CAM method registry
# ---------------------------
CAM_CLASSES: Dict[str, type] = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "hirescam": HiResCAM,
}

# ---------------------------
# Per-method colormap mapping
#  - first 5 use Matplotlib: Reds, Blues, Greens, Oranges, Purples
#  - last 5 use OpenCV colormaps
# ---------------------------
METHOD_TO_CMAP_KIND: Dict[str, Tuple[str, Union[str, int]]] = {
    "gradcam":      ("mpl", "Reds"),
    "gradcam++":    ("mpl", "Blues"),
    "scorecam":     ("mpl", "Greens"),
    "ablationcam":  ("mpl", "Oranges"),
    "xgradcam":     ("mpl", "Purples"),
    "layercam":     ("cv2", cv2.COLORMAP_HOT),
    "fullgrad":     ("cv2", cv2.COLORMAP_BONE),
    "eigencam":     ("cv2", cv2.COLORMAP_INFERNO),
    "eigengradcam": ("cv2", cv2.COLORMAP_MAGMA),
    "hirescam":     ("cv2", cv2.COLORMAP_PINK),
}

# ---------------------------
# Target layer discovery:
# prefer model.cam_target if set; else last Conv2d in backbone; else last Conv2d anywhere
# ---------------------------
def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def discover_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "cam_target") and isinstance(model.cam_target, nn.Module):
        return model.cam_target
    if hasattr(model, "backbone"):
        last = _find_last_conv(model.backbone)
        if last is not None:
            return last
    last = _find_last_conv(model)
    if last is None:
        raise RuntimeError("No Conv2d layer found for CAM target.")
    return last

# ---------------------------
# Main CAM: returns a [0,1] float mask at input spatial size
# ---------------------------
@torch.enable_grad()
def compute_cam_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,          # (1,3,H,W) on correct device
    class_index: int,
    method: str = "gradcam",
    aug_smooth: bool = True,
    eigen_smooth: bool = True,
) -> np.ndarray:
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must be (1, C, H, W)")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    mname = method.lower()
    if mname not in CAM_CLASSES:
        raise ValueError(f"Unknown CAM method: {method}")
    CAMClass = CAM_CLASSES[mname]

    target_layer = discover_target_layer(model)
    cam = CAMClass(model=model, target_layers=[target_layer])
    cam.batch_size = 1

    # class-specific target
    targets = [ClassifierOutputTarget(int(class_index))]

    mask = cam(
        input_tensor=input_tensor.float(),
        targets=targets,
        aug_smooth=aug_smooth,
        eigen_smooth=eigen_smooth
    )[0]  # HxW float

    # normalize [0,1]
    mask = np.asarray(mask, dtype=np.float32)
    mmin, mmax = float(mask.min()), float(mask.max())
    if mmax > mmin:
        mask = (mask - mmin) / (mmax - mmin)
    else:
        mask = np.zeros_like(mask, dtype=np.float32)

    # release hooks
    try:
        cam.activations_and_grads.release()
    except Exception:
        pass
    del cam

    return mask

# ---------------------------
# Overlay: supports Matplotlib or OpenCV colormaps based on cmap_kind
# ---------------------------
def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,                # uint8 (H, W, 3) BGR
    cam_mask: np.ndarray,                # float [0,1], (H, W)
    alpha: float = 0.2,
    cmap_kind: Tuple[str, Union[str, int]] = ("cv2", cv2.COLORMAP_HOT),
) -> np.ndarray:
    H, W = base_bgr.shape[:2]
    cam_u8 = np.uint8(np.clip(cam_mask, 0, 1) * 255)
    cam_u8 = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)

    kind, spec = cmap_kind
    if kind == "mpl":
        if not _HAVE_MPL:
            # Fallback to a stable OpenCV map if Matplotlib isn't available
            heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        else:
            # spec like "Reds", "Blues", ...
            m = cm.get_cmap(str(spec))
            heat_rgb = (m(cam_u8 / 255.0)[..., :3] * 255.0).astype(np.uint8)  # HxWx3 RGB
            heat = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    else:
        heat = cv2.applyColorMap(cam_u8, int(spec))

    overlay = cv2.addWeighted(heat, float(alpha), base_bgr, 1.0 - float(alpha), 0.0)
    return overlay

