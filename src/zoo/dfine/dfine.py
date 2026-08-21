"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = [
    "DFINE",
]


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, return_intermediates=False):
        # Default behavior is unchanged.
        # Set return_intermediates=True only when extracting visualization features.
        backbone_feats = self.backbone(x)

        if return_intermediates:
            encoder_feats, encoder_aux = self.encoder(
                backbone_feats, return_intermediates=True
            )
            outputs = self.decoder(encoder_feats, targets)
            aux = {
                "backbone_feats": backbone_feats,
                "encoder_feats": encoder_feats,
                "wavelet": encoder_aux.get("wavelet", None),
            }
            return outputs, aux

        encoder_feats = self.encoder(backbone_feats)
        outputs = self.decoder(encoder_feats, targets)
        return outputs

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
