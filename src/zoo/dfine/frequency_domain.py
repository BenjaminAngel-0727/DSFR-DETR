import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        return self._fuse_bn_tensor()

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvBNAct(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size=3,
        stride=1,
        groups=1,
        padding=None,
        dilation=1,
        bias=False,
        act="silu",
    ):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = (
                    ((kernel_size[0] - 1) * dilation) // 2,
                    ((kernel_size[1] - 1) * dilation) // 2,
                )
            else:
                padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class HaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        high = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        ll = torch.outer(low, low)
        lh = torch.outer(high, low)
        hl = torch.outer(low, high)
        hh = torch.outer(high, high)
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "H and W must be even for HaarDWT2D."
        weight = self.filt.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        y = y.view(b, c, 4, h // 2, w // 2)
        ll = y[:, :, 0].contiguous()
        lh = y[:, :, 1].contiguous()
        hl = y[:, :, 2].contiguous()
        hh = y[:, :, 3].contiguous()
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        low = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        high = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        ll = torch.outer(low, low)
        lh = torch.outer(high, low)
        hl = torch.outer(low, high)
        hh = torch.outer(high, high)
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt)

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).view(b, 4 * c, h, w)
        weight = self.filt.to(dtype=ll.dtype, device=ll.device).repeat(c, 1, 1, 1)
        return F.conv_transpose2d(y, weight, stride=2, padding=0, groups=c)


class DeepSemanticGuidanceModule(nn.Module):
    def __init__(self, channels=256, act="silu"):
        super().__init__()
        self.proj = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        )
        self.align = nn.Sequential(
            ConvNormLayer_fuse(channels * 2, channels, 3, 1, act=act),
            ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        )
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out = ConvNormLayer_fuse(channels, channels, 3, 1, act=act)

    def forward(self, p5, low_ref):
        guide = self.proj(p5)
        guide = self.up1(guide)
        guide = self.up2(guide)
        if guide.shape[-2:] != low_ref.shape[-2:]:
            guide = F.interpolate(guide, size=low_ref.shape[-2:], mode="nearest")
        delta = self.align(torch.cat([guide, low_ref], dim=1))
        gate = torch.sigmoid(self.gate(delta))
        guide = guide + gate * delta
        return self.out(guide)


class FrequencyStructureEnhancementModule(nn.Module):
    def __init__(self, channels=256, reduction=4, act="silu"):
        super().__init__()
        inter_channels = max(channels // reduction, 16)
        self.fuse = nn.Sequential(
            ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act),
            ConvNormLayer_fuse(channels, channels, 3, 1, act=act),
        )
        self.lowpass = nn.Sequential(
            ConvBNAct(channels, channels, 5, 1, groups=channels, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        )
        self.spatial_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=True),
            get_activation(act),
            nn.Conv2d(inter_channels, channels, kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, low, guide):
        fused = self.fuse(torch.cat([low, guide], dim=1))
        base = self.lowpass(low)
        spatial = torch.sigmoid(self.spatial_gate(fused))
        channel = torch.sigmoid(self.channel_gate(guide))
        update = spatial * channel * (base - low)
        return low + self.alpha * update


class FrequencyDetailEnhancementModule(nn.Module):
    def __init__(self, channels=256, act="silu", mid_ratio=0.5, gate_ratio=0.25):
        super().__init__()
        mid_channels = max(int(channels * mid_ratio), 32)
        gate_channels = max(int(channels * gate_ratio), 32)
        self.shared_stem = nn.Sequential(
            ConvNormLayer_fuse(channels, mid_channels, 1, 1, act=act),
            ConvBNAct(mid_channels, mid_channels, 3, 1, groups=mid_channels, act=act),
        )
        self.lh_branch = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, (1, 5), 1, groups=mid_channels, act=act),
            ConvBNAct(mid_channels, mid_channels, 3, 1, groups=mid_channels, act=act),
        )
        self.hl_branch = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, (5, 1), 1, groups=mid_channels, act=act),
            ConvBNAct(mid_channels, mid_channels, 3, 1, groups=mid_channels, act=act),
        )
        self.hh_branch = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, 3, 1, groups=mid_channels, dilation=2, act=act),
            ConvBNAct(mid_channels, mid_channels, 3, 1, groups=mid_channels, act=act),
        )
        self.shared_expand = ConvNormLayer_fuse(mid_channels, channels, 1, 1, act=act)
        self.guide_reduce = ConvNormLayer_fuse(channels, gate_channels, 1, 1, act=act)
        self.gate = nn.Sequential(
            ConvNormLayer_fuse(gate_channels + 3, gate_channels, 1, 1, act=act),
            ConvBNAct(gate_channels, gate_channels, 3, 1, groups=gate_channels, act=act),
            nn.Conv2d(gate_channels, 3, kernel_size=1, bias=True),
        )
        self.alpha_h = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _magnitude_map(x):
        return x.abs().mean(dim=1, keepdim=True)

    def forward(self, lh, hl, hh, guide):
        s_lh = self.shared_stem(lh)
        s_hl = self.shared_stem(hl)
        s_hh = self.shared_stem(hh)
        r_lh = self.shared_expand(self.lh_branch(s_lh))
        r_hl = self.shared_expand(self.hl_branch(s_hl))
        r_hh = self.shared_expand(self.hh_branch(s_hh))
        guide_feat = self.guide_reduce(guide)
        m_lh = self._magnitude_map(lh)
        m_hl = self._magnitude_map(hl)
        m_hh = self._magnitude_map(hh)
        gate_in = torch.cat([guide_feat, m_lh, m_hl, m_hh], dim=1)
        gate = torch.sigmoid(self.gate(gate_in))
        g_lh = gate[:, 0:1]
        g_hl = gate[:, 1:2]
        g_hh = gate[:, 2:3]
        out_lh = lh + self.alpha_h[0] * g_lh * r_lh
        out_hl = hl + self.alpha_h[1] * g_hl * r_hl
        out_hh = hh + self.alpha_h[2] * g_hh * r_hh
        return out_lh, out_hl, out_hh


class DSFRModule(nn.Module):
    def __init__(self, channels=256, act="silu"):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.guide_gen = DeepSemanticGuidanceModule(channels, act=act)
        self.low_enhance = FrequencyStructureEnhancementModule(channels, act=act)
        self.high_enhance = FrequencyDetailEnhancementModule(channels, act=act)
        self.recon_post = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        )
        self.alpha_rec = nn.Parameter(torch.zeros(1))

    def forward(self, p2, p5, return_intermediates=False):
        ll, lh, hl, hh = self.dwt(p2)
        guide = self.guide_gen(p5, ll)
        ll_hat = self.low_enhance(ll, guide)
        lh_hat, hl_hat, hh_hat = self.high_enhance(lh, hl, hh, guide)
        p2_rec = self.idwt(ll_hat, lh_hat, hl_hat, hh_hat)
        residual = self.recon_post(p2_rec - p2)
        out = p2 + self.alpha_rec * residual

        if return_intermediates:
            aux = {
                "p2": p2,
                "p5": p5,
                "ll": ll,
                "lh": lh,
                "hl": hl,
                "hh": hh,
                "guide": guide,
                "ll_hat": ll_hat,
                "lh_hat": lh_hat,
                "hl_hat": hl_hat,
                "hh_hat": hh_hat,
                "p2_rec": p2_rec,
                "residual": residual,
                "out": out,
                "diff_ll": (ll_hat - ll).abs(),
                "diff_lh": (lh_hat - lh).abs(),
                "diff_hl": (hl_hat - hl).abs(),
                "diff_hh": (hh_hat - hh).abs(),
                "diff_p2": (out - p2).abs(),
            }
            return out, aux

        return out


LFGuideGenerator = DeepSemanticGuidanceModule
LowFrequencyEnhancer = FrequencyStructureEnhancementModule
HighFrequencyEnhancer = FrequencyDetailEnhancementModule
WaveletGuidedP2Enhancer = DSFRModule
