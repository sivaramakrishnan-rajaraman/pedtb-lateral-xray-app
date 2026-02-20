# src/tb_model.py

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from torchvision import models
from torchmetrics.classification import Accuracy, F1Score


# -------------------------
# Utilities to find last conv
# -------------------------
class _ConvSpy:
    """
    Attaches forward hooks to every Conv2d in `module` and stores the shapes
    of their outputs during a single dummy forward pass. We can then identify
    the Conv2d(s) that actually wrote the final-stage [B, C, H_last, W_last] map.
    """
    def __init__(self):
        self.records: List[Tuple[str, nn.Module, Tuple[int, int, int, int]]] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str, mod: nn.Conv2d):
        def _h(m, x, y):
            if isinstance(y, torch.Tensor) and y.ndim == 4:
                self.records.append((name, mod, tuple(y.shape)))
        return _h

    def attach(self, module: nn.Module):
        for n, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                self.hooks.append(m.register_forward_hook(self._hook(n, m)))

    def detach(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()


def _discover_laststage_conv_k(
    backbone: nn.Module, img_size: int
) -> Tuple[Optional[nn.Conv2d], Optional[Tuple[int, int]]]:
    """
    Return (last_conv_that_writes_final_stage, kernel_size) or (None, None).
    Runs a CPU dummy forward with features_only to identify the final feature map size.
    """
    spy = _ConvSpy()
    spy.attach(backbone)
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
        feats = backbone(x)  # timm features_only → list[tensor]; torchvision VGG → Tensor
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        last_hw = tuple(last.shape[-2:])
    spy.detach()

    # Keep only convs whose output H,W match the final feature map H,W
    matches: List[Tuple[str, nn.Module, Tuple[int, int, int, int]]] = []
    for (n, m, shp) in spy.records:
        if tuple(shp[-2:]) == last_hw:
            matches.append((n, m, shp))
    if not matches:
        return None, None
    # Execution order is preserved in records; take the last one
    _, last_conv, _ = matches[-1]
    ks = last_conv.kernel_size
    k = ks if isinstance(ks, tuple) else (ks, ks)
    return last_conv, (int(k[0]), int(k[1]))


def _resolve_timm_name(user_name: str, table: Dict[str, str]) -> str:
    """
    Normalize aliases into exact timm IDs.
    Falls through to the original string if already a valid timm name.
    """
    key = user_name.strip().lower()
    norm = (
        key.replace("-", "_")
           .replace(" ", "")
           .replace("rw224", "rw_224")
    )
    return table.get(norm, table.get(key, user_name))


def _last_conv_any(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# -------------------------
# The model
# -------------------------
class TBModel(pl.LightningModule):
    def __init__(self, h: Dict):
        super().__init__()
        # Save hyperparams for Lightning checkpoints
        self.h: Dict = h
        self.save_hyperparameters("h")

        name: str        = h["model"]
        img_size: int    = int(h["img_size"])
        drop_rate: float = float(h["dropout"])
        num_classes: int = int(h["num_classes"])
        self.num_classes = num_classes

        # Flags
        self.is_transformer: bool = False
        self.cam_target: Optional[nn.Module] = None  # exposed for CAM

        # --------------- 1) Backbone selection ---------------
        if name in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            weights = getattr(models, f"{name.upper()}_Weights").IMAGENET1K_V1
            vgg = getattr(models, name)(weights=weights)
            self.backbone = vgg.features
            feat_dim = 512
            # VGG already ends with convs—keep as-is
            self.post3x3 = nn.Identity()

            # Target: last Conv2d in the VGG conv stack
            self.cam_target = _last_conv_any(self.backbone)

        elif name in [
            "resnet18", "inceptionv3_new", "xception", "resnet34", "resnet50",
            "densenet121", "efficientnet_b0", "convnext_nano_new", "convnext_small",
            "dpn68_new", "hrnet32_new"
        ]:
            timm_map = {
                "resnet18":           "resnet18",
                "inceptionv3_new":    "inception_v3",
                "xception":           "xception",
                "resnet34":           "resnet34",
                "resnet50":           "resnet50",
                "densenet121":        "densenet121",
                "efficientnet_b0":    "efficientnet_b0",
                "convnext_nano_new":  "convnext_nano.in12k",
                "convnext_small":     "convnext_small",
                "dpn68_new":          "dpn68.mx_in1k",
                "hrnet32_new":        "hrnet_w32.ms_in1k",
            }
            base = timm.create_model(_resolve_timm_name(name, timm_map), pretrained=True, features_only=True)
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]

            # Discover last-stage conv + kernel
            last_conv, k = _discover_laststage_conv_k(self.backbone, img_size)
            need_3x3 = (k is None) or (k != (3, 3))

            if need_3x3:
                # Add trainable 3×3 + BN + SiLU
                self.post3x3 = nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                    nn.SiLU(inplace=True),
                )
                # CAM target = the newly added conv (index 0)
                self.cam_target = self.post3x3[0]
                print(f"[CAM] Added 3×3 conv; cam_target=post3x3[0] for '{name}'")
            else:
                self.post3x3 = nn.Identity()
                # CAM target = the discovered last-stage conv
                self.cam_target = last_conv
                print(f"[CAM] Using last-stage conv (k={k}); cam_target set for '{name}'")

        elif name == "efficientnetv2s_new":
            base = timm.create_model("tf_efficientnetv2_s.in21k", pretrained=True, features_only=True)
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]

            last_conv, k = _discover_laststage_conv_k(self.backbone, img_size)
            need_3x3 = (k is None) or (k != (3, 3))
            if need_3x3:
                self.post3x3 = nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                    nn.SiLU(inplace=True),
                )
                self.cam_target = self.post3x3[0]
                print(f"[CAM] Added 3×3 conv; cam_target=post3x3[0] for 'efficientnetv2s_new'")
            else:
                self.post3x3 = nn.Identity()
                self.cam_target = last_conv
                print(f"[CAM] Using last-stage conv (k={k}); cam_target set for 'efficientnetv2s_new'")

        elif any(s in name.lower() for s in ["coatnet", "maxvit"]):
            coat_maxvit_map = {
                # CoAtNet-0
                "coatnet0": "coatnet_0_rw_224.sw_in1k",
                "coatnet_0": "coatnet_0_rw_224.sw_in1k",
                "coatnet_0_rw_224": "coatnet_0_rw_224.sw_in1k",
                "coatnet_0_rw_224.sw_in1k": "coatnet_0_rw_224.sw_in1k",
                # CoAtNet-nano
                "coatnet_nano": "coatnet_nano_rw_224.sw_in1k",
                "coatnetnano": "coatnet_nano_rw_224.sw_in1k",
                "coatnet_nano_rw_224": "coatnet_nano_rw_224.sw_in1k",
                "coatnet_nano_rw_224.sw_in1k": "coatnet_nano_rw_224.sw_in1k",
                # MaxViT small/tiny (TF port)
                "maxvit_small": "maxvit_small_tf_224.in1k",
                "maxvit_small_tf_224.in1k": "maxvit_small_tf_224.in1k",
                "maxvit_tiny": "maxvit_tiny_tf_224.in1k",
                "maxvit_tiny_tf_224.in1k": "maxvit_tiny_tf_224.in1k",
            }
            resolved = _resolve_timm_name(name, coat_maxvit_map)
            print(f"[Backbone] resolved '{name}' → timm id '{resolved}'")
            base = timm.create_model(resolved, pretrained=True, features_only=True)
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]

            last_conv, k = _discover_laststage_conv_k(self.backbone, img_size)
            need_3x3 = (k is None) or (k != (3, 3))
            if need_3x3:
                self.post3x3 = nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                    nn.SiLU(inplace=True),
                )
                self.cam_target = self.post3x3[0]
                print(f"[CAM] Added 3×3 conv; cam_target=post3x3[0] for '{name}'")
            else:
                self.post3x3 = nn.Identity()
                self.cam_target = last_conv
                print(f"[CAM] Using last-stage conv (k={k}); cam_target set for '{name}'")

        elif any(s in name.lower() for s in ["vit", "deit", "beit"]):
            vt_map = {
                "vit_base": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "vitb16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "vit_b16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "vit_base_patch16_224": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "vit_base_patch16_224.augreg2_in21k_ft_in1k": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                "deit3_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
                "deit3s": "deit3_small_patch16_224.fb_in22k_ft_in1k",
                "deit3_small_patch16_224": "deit3_small_patch16_224.fb_in22k_ft_in1k",
                "deit3_small_patch16_224.fb_in22k_ft_in1k": "deit3_small_patch16_224.fb_in22k_ft_in1k",
                "beit3_base": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
                "beit_v2_base": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
                "beitv2_base_patch16_224": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
                "beitv2_base_patch16_224.in1k_ft_in22k_in1k": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
            }
            resolved = _resolve_timm_name(name, vt_map)
            print(f"[Backbone] resolved '{name}' → timm id '{resolved}'")
            base = timm.create_model(
                resolved,
                pretrained=True,
                features_only=True,   # ensures we get a spatial (B,C,H,W) for head
                img_size=img_size,
                drop_rate=0.0,
                drop_path_rate=0.0,
                pretrained_strict=False,
            )
            self.backbone = base
            feat_dim = base.feature_info[-1]["num_chs"]
            self.is_transformer = True
            self.post3x3 = nn.Identity()

            # For ViT-like models, Grad-CAM typically uses token attention maps,
            # but since we focus on CNN now (DPN-68), we leave CAM target unset.
            # If needed later, we can set cam_target to the stem conv, if present:
            self.cam_target = _last_conv_any(self.backbone)

        else:
            raise ValueError(f"Unsupported model '{name}'")

        # --------------- 2) Classification head ---------------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,H,W)->(B,C,1,1)
            nn.Flatten(),             # (B,C)
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # --------------- 3) Metrics ---------------
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1  = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1    = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    # Forward is identical to Biowulf version (with post3x3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = self.post3x3(feats)
        logits = self.head(feats)
        return logits

    # ---- Lightning plumbing (unchanged) ----
    @staticmethod
    def _nll_from_onehot_logits(logits: torch.Tensor, labels_oh: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        return -(labels_oh * logp).sum(dim=1).mean()

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        labels_oh = F.one_hot(labels, num_classes=self.num_classes).float() if labels.ndim == 1 else labels.float()
        logits = self(imgs)
        loss   = self._nll_from_onehot_logits(logits, labels_oh)
        preds  = logits.argmax(dim=1)
        tgt    = labels_oh.argmax(dim=1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc",  self.train_acc(preds, tgt), on_epoch=True, prog_bar=True)
        self.log("train_f1",   self.train_f1(preds, tgt),  on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        labels_oh = F.one_hot(labels, num_classes=self.num_classes).float() if labels.ndim == 1 else labels.float()
        logits = self(imgs)
        loss   = self._nll_from_onehot_logits(logits, labels_oh)
        preds  = logits.argmax(dim=1)
        tgt    = labels_oh.argmax(dim=1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc",  self.val_acc(preds, tgt), on_epoch=True, prog_bar=True)
        self.log("val_f1",   self.val_f1(preds, tgt),  on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.h.get("lr", 5e-5), weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.h.get("max_epochs", 64))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    # ---- Convenience: auto-strict checkpoint loader ----
    @classmethod
    def load_from_ckpt_auto_strict(
        cls,
        ckpt_path: str,
        *,
        h: Dict,
        map_location: Optional[Union[str, torch.device]] = None,
    ):
        """
        Try strict=True first, fallback to strict=False.
        Returns (model, info_dict).
        """
        tried_err: Optional[Exception] = None
        for strict_flag in (True, False):
            try:
                model = cls.load_from_checkpoint(
                    ckpt_path,
                    h=h,
                    strict=strict_flag,
                    map_location=map_location,
                )
                return model, {"strict_used": strict_flag, "missing_keys": [], "unexpected_keys": []}
            except Exception as e:
                tried_err = e
                continue
        raise tried_err or RuntimeError(f"Failed to load checkpoint: {ckpt_path}")


