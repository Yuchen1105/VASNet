"""VASNet journal release, derived from the verified local MambaUXV32 source.

Parameter and buffer names retain historical checkpoint compatibility.
Paper names: ux_block -> RASE, BAMBlock3D_Improved -> GSA,
PATSkipFusion3D -> SCA. See docs/IMPLEMENTATION.md for implementation details.
"""
import math
import random
from functools import partial
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from ._layers import DropPath
try:
    from mamba_ssm import Mamba
except (ImportError, OSError) as exc:
    raise ImportError(
        "VASNet requires the CUDA Mamba implementation. Install the pinned "
        "mamba-ssm/causal-conv1d dependencies on Linux; see docs/INSTALL.md."
    ) from exc

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class ux_block(nn.Module):

    def __init__(self, dim, slice, drop_path=0.0, layer_scale_init_value=1e-06):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-06)
        self.norm2 = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Conv3d(dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mamba1 = Mamba(d_model=dim // 2, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=dim // 2, d_state=16, d_conv=4, expand=2)
        self.AMFWFusion = DualBranchFusion_KAN(in_channels=dim, out_channels=dim)
        self.onion_scan_indices = self._generate_onion_scan_indices(slice, slice, slice)

    def _snake_scan_2d(self, height, width, reverse=False):
        positions = []
        for i in range(height):
            if i % 2 == 0 and (not reverse) or (i % 2 != 0 and reverse):
                for j in range(width):
                    positions.append((i, j))
            else:
                for j in range(width - 1, -1, -1):
                    positions.append((i, j))
        return positions

    def _perimeter_scan_2d(self, height, width, start_position=None, counter_clockwise=False):
        positions = []
        if not counter_clockwise:
            for j in range(width):
                positions.append((0, j))
            for i in range(1, height):
                positions.append((i, width - 1))
            for j in range(width - 2, -1, -1):
                positions.append((height - 1, j))
            for i in range(height - 2, 0, -1):
                positions.append((i, 0))
        else:
            for j in range(width - 1, -1, -1):
                positions.append((0, j))
            for i in range(1, height):
                positions.append((i, 0))
            for j in range(1, width):
                positions.append((height - 1, j))
            for i in range(height - 2, 0, -1):
                positions.append((i, width - 1))
        if start_position:
            (start_i, start_j) = start_position
            start_idx = None
            for (idx, (i, j)) in enumerate(positions):
                if i == start_i and j == start_j:
                    start_idx = idx
                    break
            if start_idx is not None:
                positions = positions[start_idx:] + positions[:start_idx]
        return positions

    def _generate_onion_scan_indices(self, D, H, W):
        all_indices = []
        num_shells = min(D, H, W) // 2
        if min(D, H, W) % 2 == 1:
            num_shells += 1
        for shell in range(num_shells):
            shell_indices = []
            (min_z, max_z) = (shell, D - 1 - shell)
            (min_y, max_y) = (shell, H - 1 - shell)
            (min_x, max_x) = (shell, W - 1 - shell)
            if min_z > max_z or min_y > max_y or min_x > max_x:
                break
            if min_z == max_z:
                plane_positions = self._snake_scan_2d(max_y - min_y + 1, max_x - min_x + 1)
                for (i, j) in plane_positions:
                    shell_indices.append([min_z, min_y + i, min_x + j])
                all_indices.extend(shell_indices)
                continue
            top_positions = self._snake_scan_2d(max_y - min_y + 1, max_x - min_x + 1)
            for (i, j) in top_positions:
                shell_indices.append([min_z, min_y + i, min_x + j])
            for z in range(min_z + 1, max_z):
                counter_clockwise = (z - min_z) % 2 == 1
                perimeter_positions = self._perimeter_scan_2d(max_y - min_y + 1, max_x - min_x + 1, counter_clockwise=counter_clockwise)
                for (i, j) in perimeter_positions:
                    shell_indices.append([z, min_y + i, min_x + j])
            bottom_positions = self._snake_scan_2d(max_y - min_y + 1, max_x - min_x + 1, reverse=True)
            for (i, j) in bottom_positions:
                shell_indices.append([max_z, min_y + i, min_x + j])
            all_indices.extend(shell_indices)
        return torch.tensor(all_indices, dtype=torch.long)

    def scan_tensor(self, tensor, indices=None):
        if indices is None:
            indices = self.onion_scan_indices.to(tensor.device)
        (d_idx, h_idx, w_idx) = (indices[:, 0], indices[:, 1], indices[:, 2])
        out = tensor[:, :, d_idx, h_idx, w_idx]
        return out

    def restore_tensor(self, seq_tensor, D, H, W, indices=None):
        (B, C, _) = seq_tensor.shape
        device = seq_tensor.device
        if indices is None:
            indices = self.onion_scan_indices.to(device)
        else:
            indices = indices.to(device)
        flat_idx = indices[:, 0] * (H * W) + indices[:, 1] * W + indices[:, 2]
        out = torch.zeros((B, C, D * H * W), dtype=seq_tensor.dtype, device=device)
        out.scatter_(2, flat_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1), seq_tensor)
        out = out.view(B, C, D, H, W)
        return out

    def forward(self, x):
        input = x
        (B, C, D, H, W) = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm1(x)
        x = x.permute(0, 4, 1, 2, 3)
        all_views = [[2, 3, 4], [3, 2, 4], [4, 2, 3]]
        perm_orders = (random.choice(all_views) if self.training or getattr(self, 'inference_scan', 'canonical') == 'legacy_random' else all_views[0])

        def get_inverse_perm(perm):
            return [perm.index(i) + 2 for i in range(2, 5)]
        xs = x.permute(0, 1, *perm_orders)
        (x1, x2) = torch.chunk(xs, 2, dim=1)
        seq_x1 = self.scan_tensor(x1, self.onion_scan_indices)
        proc_x1 = self.mamba1(seq_x1.permute(0, 2, 1))
        x1_restored = self.restore_tensor(proc_x1.permute(0, 2, 1), D, H, W, indices=self.onion_scan_indices)
        seq_x2 = self.scan_tensor(x2, self.onion_scan_indices)
        N_total = seq_x2.shape[2]
        n_inner = (D // 2) ** 3
        outer_len = N_total - n_inner
        seq_outer = seq_x2[:, :, :outer_len]
        seq_inner = seq_x2[:, :, outer_len:]
        proc_outer = self.mamba2(seq_outer.permute(0, 2, 1))
        proc_inner = self.mamba2(seq_inner.permute(0, 2, 1))
        merged = torch.cat([proc_outer, proc_inner], dim=1)
        x2_restored = self.restore_tensor(merged.permute(0, 2, 1), D, H, W, indices=self.onion_scan_indices)
        x = torch.cat([x1_restored, x2_restored], dim=1)
        restore_order = get_inverse_perm(perm_orders)
        x = x.permute(0, 1, *restore_order)
        x = self.AMFWFusion(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm2(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x

class uxnet_conv(nn.Module):

    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], drop_path_rate=0.0, layer_scale_init_value=1e-06, out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3), LayerNorm(dims[0], eps=1e-06, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-06, data_format='channels_first'), nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        slice = [64, 32, 16, 8]
        for i in range(4):
            stage = nn.Sequential(*[ux_block(dim=dims[i], slice=slice[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.out_indices = out_indices
        norm_layer = partial(LayerNorm, eps=1e-06, data_format='channels_first')
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class ProjectionHead(nn.Module):

    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()
        Logger.info('proj_dim: {}'.format(proj_dim))
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(nn.Conv3d(dim_in, dim_in, kernel_size=1), ModuleHelper.BNReLU(dim_in, bn_type=bn_type), nn.Conv3d(dim_in, proj_dim, kernel_size=1))

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

class DepthwiseSeparableConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DualBranchFusion_KAN(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=4):
        super(DualBranchFusion_KAN, self).__init__()
        self.extractor1 = DepthwiseSeparableConv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.extractor2 = DepthwiseSeparableConv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveMaxPool3d(1)
        self.kan_alpha_1 = KANLinear(out_channels, out_channels // reduction)
        self.kan_alpha_2 = KANLinear(out_channels // reduction, out_channels)
        self.kan_beta_1 = KANLinear(out_channels, out_channels // reduction)
        self.kan_beta_2 = KANLinear(out_channels // reduction, out_channels)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, F_i):
        F_s = self.extractor1(F_i)
        F_v = self.extractor2(F_i)
        F_g = (self.gap(F_s) + self.gap(F_v)) / 2
        F_g = F_g.view(F_g.size(0), F_g.size(1))
        alpha = self.kan_alpha_2(self.act(self.kan_alpha_1(F_g)))
        beta = self.kan_beta_2(self.act(self.kan_beta_1(F_g)))
        alpha = self.softmax(alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.softmax(beta).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        F_s_prime = F_s * alpha
        F_v_prime = F_v * beta
        F_o = F_s_prime + F_v_prime
        return F_o * F_i

class KANLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer('grid', grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

class ChannelAttention3D_KAN(nn.Module):

    def __init__(self, channel: int, reduction: int=16, kan_kwargs: Optional[dict]=None):
        super().__init__()
        if kan_kwargs is None:
            kan_kwargs = dict(grid_size=5, spline_order=3, scale_noise=0.01, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        self.channel = channel
        self.reduction = reduction
        C_red = max(1, channel // reduction)
        self.reduce = nn.Conv3d(channel, C_red, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm3d(C_red)
        self.act = nn.ReLU(inplace=True)
        self.kan = KANLinear(in_features=2 * C_red, out_features=C_red, **kan_kwargs)
        self.expand = nn.Conv3d(C_red, channel, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, D, H, W) = x.shape
        C_red = self.reduce.out_channels
        y = self.reduce(x)
        y = self.bn_reduce(y)
        y = self.act(y)
        avg = F.adaptive_avg_pool3d(y, 1).view(B, C_red)
        maxp = F.adaptive_max_pool3d(y, 1).view(B, C_red)
        pooled = torch.cat([avg, maxp], dim=1)
        kan_out = self.kan(pooled)
        kan_out = kan_out.view(B, C_red, 1, 1, 1)
        out = self.expand(kan_out)
        out = out.expand_as(x)
        return out

class SepConv3D(nn.Module):

    def __init__(self, channels: int, kernel_size: int=5, dilation: int=1, bias: bool=False):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels, bias=bias)
        self.bn_dw = nn.BatchNorm3d(channels)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=bias)
        self.bn_pw = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.pw(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        return x

class SpatialAttention3D_Sep(nn.Module):

    def __init__(self, channel: int, reduction: int=16, kernel_size: int=5, dilation: int=1):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        C_red = max(1, channel // reduction)
        self.reduce = nn.Conv3d(channel, C_red, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm3d(C_red)
        self.relu = nn.ReLU(inplace=True)
        self.sep = SepConv3D(channels=C_red, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.project = nn.Conv3d(C_red, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, D, H, W) = x.shape
        y = self.reduce(x)
        y = self.bn_reduce(y)
        y = self.relu(y)
        y = self.sep(y)
        y = self.project(y)
        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode='trilinear', align_corners=False)
        y = y.expand_as(x)
        return y

class BAMBlock3D_Improved(nn.Module):

    def __init__(self, channel: int=64, reduction: int=16, kan_kwargs: Optional[dict]=None, sa_kernel: int=5, sa_dilation: int=1):
        super().__init__()
        self.ca = ChannelAttention3D_KAN(channel=channel, reduction=reduction, kan_kwargs=kan_kwargs)
        self.sa = SpatialAttention3D_Sep(channel=channel, reduction=reduction, kernel_size=sa_kernel, dilation=sa_dilation)
        self.sigmoid = nn.Sigmoid()
        self.pwcconv = nn.Conv3d(channel, channel * 2, kernel_size=1, groups=channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1.0 + weight) * x
        out = self.pwcconv(out)
        return out

class SRM3D(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.cfc = nn.Conv3d(channel, channel, kernel_size=(1, 1, 2), bias=False)
        self.bn = nn.BatchNorm3d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        (b, c, d, h, w) = x.shape
        mean = x.view(b, c, -1).mean(-1).view(b, c, 1, 1, 1)
        std = x.view(b, c, -1).std(-1).view(b, c, 1, 1, 1)
        u = torch.cat([mean, std], dim=-1)
        z = self.cfc(u)
        z = self.bn(z)
        g = self.sigmoid(z)
        return x * g.expand_as(x)

class PartialSpatialAttn3D(nn.Module):

    def __init__(self, dim, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.conv_branch = nn.Conv3d(self.dim_conv, self.dim_conv, kernel_size=1, bias=False)
        self.conv_attn = nn.Conv3d(self.dim_untouched, 1, kernel_size=1, bias=False)
        self.norm_attn = nn.BatchNorm3d(self.dim_untouched)
        self.norm_conv = nn.BatchNorm3d(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        (x1, x2) = torch.split(x, [self.dim_untouched, self.dim_conv], dim=1)
        weight = self.act(self.conv_attn(x1))
        x1 = x1 * weight
        x1 = self.norm_attn(x1)
        x2 = self.norm_conv(x2)
        x2 = self.conv_branch(x2)
        out = torch.cat((x1, x2), dim=1)
        return out

class HighLevelGuidance3D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channel_attn = SRM3D(channels)
        self.spatial_attn = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False), nn.BatchNorm3d(channels), nn.GELU(), nn.Conv3d(channels, 1, kernel_size=1, bias=True), nn.Hardsigmoid())
        self.channel_weight = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(channels, channels, kernel_size=1, bias=True), nn.Hardsigmoid())

    def forward(self, x):
        x_ch = self.channel_attn(x)
        ch_weight = self.channel_weight(x_ch)
        sp_weight = self.spatial_attn(x_ch)
        return (x_ch, ch_weight, sp_weight)

class LowLevelCalibration3D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm3d(channels), nn.GELU())

    def forward(self, x_low, ch_weight, sp_weight):
        x_low = self.proj(x_low)
        x_low = x_low * (1.0 + ch_weight) * (1.0 + sp_weight)
        return x_low

class PATSkipFusion3D(nn.Module):

    def __init__(self, high_channels, low_channels, out_channels=None, partial=0.5):
        super().__init__()
        if out_channels is None:
            out_channels = low_channels
        self.out_channels = out_channels
        self.high_compress = nn.Sequential(nn.Conv3d(high_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm3d(out_channels), nn.GELU())
        self.low_proj = nn.Sequential(nn.Conv3d(low_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm3d(out_channels), nn.GELU())
        self.high_guidance = HighLevelGuidance3D(out_channels)
        self.low_calibration = LowLevelCalibration3D(out_channels)
        self.low_channel_refine = SRM3D(out_channels)
        self.low_spatial_refine = PartialSpatialAttn3D(out_channels, partial=partial)
        self.fuse = nn.Sequential(nn.Conv3d(out_channels * 2, out_channels * 2, kernel_size=1, bias=False), nn.BatchNorm3d(out_channels * 2), nn.GELU())

    def forward(self, x_high, x_low):
        x_high = self.high_compress(x_high)
        x_high = F.interpolate(x_high, size=x_low.shape[2:], mode='trilinear', align_corners=False)
        x_low = self.low_proj(x_low)
        (x_high_guided, ch_weight, sp_weight) = self.high_guidance(x_high)
        x_low = self.low_calibration(x_low, ch_weight, sp_weight)
        x_low = self.low_channel_refine(x_low)
        x_low = self.low_spatial_refine(x_low)
        cat_feat = torch.cat([x_high_guided, x_low], dim=1)
        out = self.fuse(cat_feat)
        return out

class VASNet(nn.Module):

    def __init__(self, in_chans=1, out_chans=13, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384], drop_path_rate=0, layer_scale_init_value=1e-06, hidden_size: int=768, norm_name: Union[Tuple, str]='instance', conv_block: bool=True, res_block: bool=True, spatial_dims=3, inference_scan='canonical') -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)
        self.spatial_dims = spatial_dims
        self.uxnet_3d = uxnet_conv(in_chans=self.in_chans, depths=self.depths, dims=self.feat_size, drop_path_rate=self.drop_path_rate, layer_scale_init_value=1e-06, out_indices=self.out_indice)
        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.in_chans, out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder2 = PATSkipFusion3D(high_channels=self.feat_size[1], low_channels=self.feat_size[0], out_channels=self.feat_size[0], partial=0.5)
        self.encoder3 = PATSkipFusion3D(high_channels=self.feat_size[2], low_channels=self.feat_size[1], out_channels=self.feat_size[1], partial=0.5)
        self.encoder4 = PATSkipFusion3D(high_channels=self.feat_size[3], low_channels=self.feat_size[2], out_channels=self.feat_size[2], partial=0.5)
        self.encoder5 = BAMBlock3D_Improved(feat_size[-1])
        self.decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.hidden_size, out_channels=self.feat_size[3], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.feat_size[2], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[1], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[0], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        if inference_scan not in ('canonical', 'legacy_random'):
            raise ValueError('inference_scan must be canonical or legacy_random')
        if feat_size[0] != 48 or hidden_size != feat_size[-1] * 2:
            raise ValueError('Checkpoint-compatible topology requires feat_size[0]=48 and hidden_size=2*feat_size[-1]')
        for stage in self.uxnet_3d.stages:
            for block in stage:
                block.inference_scan = inference_scan


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple((d + 1 for d in range(len(feat_size))))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        if x_in.ndim != 5 or tuple(x_in.shape[2:]) != (128, 128, 128):
            raise ValueError('VASNet expects N,C,128,128,128 patches; use sliding-window inference for full volumes.')
        outs = self.uxnet_3d(x_in)
        enc_hidden = self.encoder5(outs[3])
        enc4 = self.encoder4(outs[3], outs[2])
        enc3 = self.encoder3(outs[2], outs[1])
        enc2 = self.encoder2(outs[1], outs[0])
        enc1 = self.encoder1(x_in)
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        return self.out(out)

# Aliases do not register additional modules or alter state_dict keys.
UXNET = VASNet
RASEBlock = ux_block
GSABlock = BAMBlock3D_Improved
SCABlock = PATSkipFusion3D
