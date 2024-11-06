from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from monai.utils import UpsampleMode
from monai.networks.layers.factories import Dropout
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer

from .CDVim import Backbone_VSSM
from .block import LGF_Mamba, SS3D
# from .resnet import extract_features, ResNetFeatureExtractor
# from .changevit import ChangeVitEncoder


class MBFEMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.ss3d = SS3D(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2"
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.ss3d(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mbfem_layer(spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1):
    
    mbfem_layer = MBFEMLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mbfem_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return mbfem_layer


class MBFEMBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mbfem_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
        self.conv2 = get_mbfem_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class LGF(nn.Module):
    def __init__(self, dim, conv_mode="deepwise", resdiual=False, act="silu"):
        super(LGF, self).__init__()

        factory_kwargs = {"device": None, "dtype": None}
        self.fusionencoder1 = LGF_Mamba(dim, bimamba_type="v2", conv_mode=conv_mode, act=act)
        self.fusionencoder2 = LGF_Mamba(dim, bimamba_type="v2", conv_mode=conv_mode, act=act)
        self.act = nn.ReLU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.out_proj1 = nn.Linear(dim*2, dim, bias=False, **factory_kwargs)
        self.out_proj2 = nn.Linear(dim*2, dim, bias=False, **factory_kwargs)
        self.resdiual = resdiual
        self.skip_scale = nn.Parameter(torch.ones(1))
    def forward(self, x1, x2):
        id1 = x1
        id2 = x2

        b, c, h, w = x1.shape
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        s6_x1, conv_x1 = self.fusionencoder1(x1)
        s6_x2, conv_x2 = self.fusionencoder1(x2)
        
        out_G1 = F.linear(rearrange(s6_x1 * self.act(s6_x2), "b d l -> b l d"), self.out_proj1.weight, self.out_proj1.bias)
        out_L1 = F.linear(rearrange(s6_x1 * self.act(conv_x2), "b d l -> b l d"), self.out_proj1.weight, self.out_proj1.bias)
        out_G2 = F.linear(rearrange(s6_x2 * self.act(s6_x1), "b d l -> b l d"), self.out_proj2.weight, self.out_proj2.bias)
        out_L2 = F.linear(rearrange(s6_x2 * self.act(conv_x1), "b d l -> b l d"), self.out_proj2.weight, self.out_proj2.bias)

        Gx1 = rearrange(out_G1, 'b (h w) c -> b c h w', h=h)
        Lx1 = rearrange(out_L1, 'b (h w) c -> b c h w', h=h)
        Gx2 = rearrange(out_G2, 'b (h w) c -> b c h w', h=h)
        Lx2 = rearrange(out_L2, 'b (h w) c -> b c h w', h=h)

        if self.resdiual:
            Gx1 = Gx1 + self.skip_scale*id1
            Lx1 = Lx1 + self.skip_scale*id1
            Gx2 = Gx2 + self.skip_scale*id2
            Lx2 = Lx2 + self.skip_scale*id2
        return Gx1, Lx1, Gx2, Lx2


class AdaptiveGate(nn.Module):
    def __init__(self, in_dim, num_expert=2):
        super().__init__()

        self.gate = nn.Linear(in_dim*2, num_expert, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_l, x_g):
        x_l = rearrange(x_l, 'b c h w -> b (h w) c')
        x_g = rearrange(x_g, 'b c h w -> b (h w) c')
        x_l = torch.mean(x_l, dim=1)
        x_g = torch.mean(x_g, dim=1)
        x_l_g = torch.cat([x_l, x_g], dim=-1)
        gate_score = self.gate(x_l_g)
        gate_score_n = self.softmax(gate_score)
        return gate_score_n


class Mamba3D(nn.Module):
    def __init__(
            self,
            pretrained: str,
            spatial_dims: int = 3,
            init_filters: int = 16,
            in_channels: int = 1,
            out_channels: int = 2,
            conv_mode: str = "deepwise",
            local_query_model = "orignal_dinner",
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            up_mode="ResMamba",
            resdiual=False,
            stage = 4,
            diff_abs="later", # "later" or "early"
            mamba_act = "silu",
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.stage = stage
        self.mamba_act = mamba_act
        self.resdiual = resdiual
        self.up_mode = up_mode
        self.diff_abs = diff_abs
        self.conv_mode = conv_mode
        self.local_query_model = local_query_model
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.channels_list = [self.init_filters, self.init_filters*2, self.init_filters*4, self.init_filters*8]
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        self.encode = Backbone_VSSM(out_indices=(0,1,2,3), pretrained=pretrained)
        # self.encode = ResNetFeatureExtractor(pretrained)
        # self.encode = ChangeVitEncoder(pretrained)

        self.mbfem_decoder_layers, self.up_samples = self._make_decoder()
        
        self.conv_final1 = self._make_final_conv(out_channels, self.init_filters*4)
        self.conv_final2 = self._make_final_conv(out_channels, self.init_filters*2)
        self.conv_final3 = self._make_final_conv(out_channels, self.init_filters)
        self.conv_final = nn.Sequential(self.conv_final1, self.conv_final2, self.conv_final3)

        self.lgf1 = LGF(self.channels_list[0], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.lgf2 = LGF(self.channels_list[1], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.lgf3 = LGF(self.channels_list[2], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.lgf4 = LGF(self.channels_list[3], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        self.lgf = nn.Sequential(self.lgf1, self.lgf2, self.lgf3, self.lgf4)

        self.ag1 = AdaptiveGate(self.channels_list[0])
        self.ag2 = AdaptiveGate(self.channels_list[1])
        self.ag3 = AdaptiveGate(self.channels_list[2])
        self.ag4 = AdaptiveGate(self.channels_list[3])
        self.ag = nn.Sequential(self.ag1, self.ag2, self.ag3, self.ag4)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)


    def _make_decoder(self):
        mbfem_decoder_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            mbfem_decoder_layers.append(
                nn.Sequential(
                    *[
                        MBFEMBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return mbfem_decoder_layers, up_samples

    def _make_final_conv(self, out_channels: int, channels):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=channels),
            self.act_mod,
            get_conv_layer(self.spatial_dims, channels, out_channels, kernel_size=1, bias=True),
        )

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        up_x = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.mbfem_decoder_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            up_x.append(x)

        if self.use_conv_final:
            for i in range(3):
                up_x[i] = self.conv_final[i](up_x[i])
                up_x[i] = F.interpolate(up_x[i], size=(256, 256), mode='bilinear', align_corners=False)
        
        return up_x

    def forward(self, t1: torch.Tensor, t2:torch.Tensor) -> torch.Tensor:
        b, c, h, w = t1.shape
        down_x1 = self.encode(t1)
        down_x2 = self.encode(t2)
        down_x = []

        for i in range(len(down_x1)):
            x1, x2 = down_x1[i], down_x2[i]
            if i < self.stage:
                x1_g, x1_l, x2_g, x2_l = self.lgf[i](x1, x2)
                x1_gate = self.ag[i](x1_l, x1_g)
                x2_gate = self.ag[i](x2_l, x2_g)
                x1 = x1_gate[:, 0:1].view(b, 1, 1, 1)*x1_l + x1_gate[:, 1:2].view(b, 1, 1, 1)*x1_g
                x2 = x2_gate[:, 0:1].view(b, 1, 1, 1)*x2_l + x2_gate[:, 1:2].view(b, 1, 1, 1)*x2_g
            down_x.append(torch.abs(x1-x2))
        down_x.reverse()

        x = self.decode(down_x[0], down_x)
        return x, x[2]
