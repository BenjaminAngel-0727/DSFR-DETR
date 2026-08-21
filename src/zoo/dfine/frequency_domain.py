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
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

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
    """
    输入:  x  [B, C, H, W]
    输出:  ll, lh, hl, hh  [B, C, H/2, W/2]
    """
    def __init__(self):
        super().__init__()

        low = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        high = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])

        ll = torch.outer(low, low)
        lh = torch.outer(high, low)
        hl = torch.outer(low, high)
        hh = torch.outer(high, high)

        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer("filt", filt)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "H and W must be even for HaarDWT2D."

        weight = self.filt.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)  # [4C,1,2,2]
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)  # [B, 4C, H/2, W/2]
        y = y.view(b, c, 4, h // 2, w // 2)

        ll = y[:, :, 0, :, :].contiguous()
        lh = y[:, :, 1, :, :].contiguous()
        hl = y[:, :, 2, :, :].contiguous()
        hh = y[:, :, 3, :, :].contiguous()
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """
    输入:  ll, lh, hl, hh  [B, C, H, W]
    输出:  x_rec [B, C, 2H, 2W]
    """
    def __init__(self):
        super().__init__()

        low = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        high = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])

        ll = torch.outer(low, low)
        lh = torch.outer(high, low)
        hl = torch.outer(low, high)
        hh = torch.outer(high, high)

        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer("filt", filt)

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).view(b, 4 * c, h, w)  # [B,4C,H,W]
        weight = self.filt.to(dtype=ll.dtype, device=ll.device).repeat(c, 1, 1, 1)  # [4C,1,2,2]
        x = F.conv_transpose2d(y, weight, stride=2, padding=0, groups=c)
        return x


class LFGuideGenerator(nn.Module):
    """
    用 AIFI 后的 P5 生成低频引导 G@80x80
    不是直接 bilinear 上采样，而是逐级语义抬升 + LL-aware 对齐
    """
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

    def forward(self, p5, ll_ref):
        g = self.proj(p5)     # 20x20
        g = self.up1(g)       # 40x40
        g = self.up2(g)       # 80x80

        if g.shape[-2:] != ll_ref.shape[-2:]:
            g = F.interpolate(g, size=ll_ref.shape[-2:], mode="nearest")

        delta = self.align(torch.cat([g, ll_ref], dim=1))
        gate = torch.sigmoid(self.gate(delta))
        g = g + gate * delta
        g = self.out(g)
        return g


class LowFrequencyEnhancer(nn.Module):
    """
    低频结构增强:
    - 用 guide 产生低频调制
    - 对 LL 做平滑/一致性增强
    """
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

    def forward(self, ll, guide):
        fuse_feat = self.fuse(torch.cat([ll, guide], dim=1))
        base = self.lowpass(ll)

        spatial = torch.sigmoid(self.spatial_gate(fuse_feat))
        channel = torch.sigmoid(self.channel_gate(guide))

        update = spatial * channel * (base - ll)
        out = ll + self.alpha * update
        return out


# class HighFrequencyEnhancer(nn.Module):
#     """
#     高频细节增强:
#     - LH / HL / HH 分支分别建模
#     - 用低频 guide + LL_hat + 高频能量做门控
#     """
#     def __init__(self, channels=256, act="silu"):
#         super().__init__()
#
#         self.lh_branch = nn.Sequential(
#             ConvBNAct(channels, channels, (1, 5), 1, act=act),
#             ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
#             ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
#         )
#         self.hl_branch = nn.Sequential(
#             ConvBNAct(channels, channels, (5, 1), 1, act=act),
#             ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
#             ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
#         )
#         self.hh_branch = nn.Sequential(
#             ConvBNAct(channels, channels, 3, 1, groups=channels, dilation=2, act=act),
#             ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
#         )
#
#         self.gate = nn.Sequential(
#             ConvNormLayer_fuse(channels * 5, channels * 2, 1, 1, act=act),
#             ConvNormLayer_fuse(channels * 2, channels * 2, 3, 1, act=act),
#             nn.Conv2d(channels * 2, channels * 3, kernel_size=1, bias=True),
#         )
#
#         self.alpha_h = nn.Parameter(torch.zeros(3))
#
#     def forward(self, lh, hl, hh, guide, ll_hat):
#         r_lh = self.lh_branch(lh)
#         r_hl = self.hl_branch(hl)
#         r_hh = self.hh_branch(hh)
#
#         energy = torch.cat([lh.abs(), hl.abs(), hh.abs()], dim=1)
#         gate = torch.sigmoid(self.gate(torch.cat([guide, ll_hat, energy], dim=1)))
#         g_lh, g_hl, g_hh = gate.chunk(3, dim=1)
#
#         out_lh = lh + self.alpha_h[0] * g_lh * r_lh
#         out_hl = hl + self.alpha_h[1] * g_hl * r_hl
#         out_hh = hh + self.alpha_h[2] * g_hh * r_hh
#         return out_lh, out_hl, out_hh

class HighFrequencyEnhancer(nn.Module):
    """
    Lite 高频细节增强:
    - 三路高频共享 stem，减少重复参数
    - 分支只保留轻量方向 depthwise 建模
    - gate 改为压缩 guide / ll_hat + 3 路能量图，避免原来 5C 的重门控
    """
    def __init__(self, channels=256, act="silu", mid_ratio=0.5, gate_ratio=0.25):
        super().__init__()

        mid_channels = max(int(channels * mid_ratio), 32)
        gate_channels = max(int(channels * gate_ratio), 32)

        # 1) 共享高频 stem：先统一降维，再做轻量局部建模
        self.shared_stem = nn.Sequential(
            ConvNormLayer_fuse(channels, mid_channels, 1, 1, act=act),
            ConvBNAct(mid_channels, mid_channels, 3, 1, groups=mid_channels, act=act),
        )

        # 2) 三路方向分支：只保留非常轻的 depthwise 方向卷积
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

        # 3) 共享回升维：避免每一路都单独做一套 mid -> C 的 pointwise
        self.shared_expand = ConvNormLayer_fuse(mid_channels, channels, 1, 1, act=act)

        # 4) 轻量 gate：guide / ll_hat 先压缩，高频只取能量图
        self.guide_reduce = ConvNormLayer_fuse(channels, gate_channels, 1, 1, act=act)
        self.ll_reduce = ConvNormLayer_fuse(channels, gate_channels, 1, 1, act=act)

        # 输入通道 = gate_channels + gate_channels + 3
        self.gate = nn.Sequential(
            ConvNormLayer_fuse(gate_channels * 2 + 3, gate_channels, 1, 1, act=act),
            ConvBNAct(gate_channels, gate_channels, 3, 1, groups=gate_channels, act=act),
            nn.Conv2d(gate_channels, 3, kernel_size=1, bias=True),
        )

        # 残差缩放，保持和你原版一致的稳训练风格
        self.alpha_h = nn.Parameter(torch.zeros(3))

    def _energy_map(self, x):
        # [B, C, H, W] -> [B, 1, H, W]
        return x.abs().mean(dim=1, keepdim=True)

    def forward(self, lh, hl, hh, guide, ll_hat):
        # 共享 stem
        s_lh = self.shared_stem(lh)
        s_hl = self.shared_stem(hl)
        s_hh = self.shared_stem(hh)

        # 轻量方向分支
        r_lh = self.shared_expand(self.lh_branch(s_lh))
        r_hl = self.shared_expand(self.hl_branch(s_hl))
        r_hh = self.shared_expand(self.hh_branch(s_hh))

        # 轻量 gate
        g_feat = self.guide_reduce(guide)
        l_feat = self.ll_reduce(ll_hat)

        e_lh = self._energy_map(lh)
        e_hl = self._energy_map(hl)
        e_hh = self._energy_map(hh)

        gate_in = torch.cat([g_feat, l_feat, e_lh, e_hl, e_hh], dim=1)
        gate = torch.sigmoid(self.gate(gate_in))   # [B, 3, H, W]

        g_lh = gate[:, 0:1, :, :]
        g_hl = gate[:, 1:2, :, :]
        g_hh = gate[:, 2:3, :, :]

        out_lh = lh + self.alpha_h[0] * g_lh * r_lh
        out_hl = hl + self.alpha_h[1] * g_hl * r_hl
        out_hh = hh + self.alpha_h[2] * g_hh * r_hh

        return out_lh, out_hl, out_hh


class WaveletGuidedP2Enhancer(nn.Module):
    """
    P2(160x160) 作为母特征做 DWT
    AIFI 后的 P5(20x20) 生成低频引导
    最终重建出增强后的 P2
    """
    def __init__(self, channels=256, act="silu"):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.guide_gen = LFGuideGenerator(channels, act=act)
        self.low_enhance = LowFrequencyEnhancer(channels, act=act)
        self.high_enhance = HighFrequencyEnhancer(channels, act=act)

        self.recon_post = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, groups=channels, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
        )

        self.alpha_rec = nn.Parameter(torch.zeros(1))

    def forward(self, p2, p5, return_intermediates=False):
        # p2: [B, C, 160, 160]
        # p5: [B, C, 20, 20]  (已过 AIFI/Transformer)
        #
        # When return_intermediates=True, this function returns:
        #   out: enhanced P2 feature
        #   aux: intermediate wavelet-domain tensors for visualization
        # The default behavior is unchanged: only out is returned.

        # 1) Wavelet decomposition of shallow P2 feature
        ll, lh, hl, hh = self.dwt(p2)   # -> [B, C, 80, 80]

        # 2) Deep semantic guidance generated from P5 and aligned with LL
        guide = self.guide_gen(p5, ll)  # -> [B, C, 80, 80]

        # 3) Low-frequency structure enhancement
        ll_hat = self.low_enhance(ll, guide)

        # 4) High-frequency detail enhancement
        lh_hat, hl_hat, hh_hat = self.high_enhance(lh, hl, hh, guide, ll_hat)

        # 5) Inverse wavelet reconstruction and residual update
        p2_rec = self.idwt(ll_hat, lh_hat, hl_hat, hh_hat)   # -> [B, C, 160, 160]
        residual = self.recon_post(p2_rec - p2)
        out = p2 + self.alpha_rec * residual

        if return_intermediates:
            aux = {
                # original projected features
                "p2": p2,
                "p5": p5,

                # original wavelet subbands
                "ll": ll,
                "lh": lh,
                "hl": hl,
                "hh": hh,

                # semantic guidance map
                "guide": guide,

                # enhanced wavelet subbands
                "ll_hat": ll_hat,
                "lh_hat": lh_hat,
                "hl_hat": hl_hat,
                "hh_hat": hh_hat,

                # reconstruction-related tensors
                "p2_rec": p2_rec,
                "residual": residual,
                "out": out,

                # absolute enhancement differences, useful for paper visualization
                "diff_ll": (ll_hat - ll).abs(),
                "diff_lh": (lh_hat - lh).abs(),
                "diff_hl": (hl_hat - hl).abs(),
                "diff_hh": (hh_hat - hh).abs(),
                "diff_p2": (out - p2).abs(),
            }
            return out, aux

        return out
