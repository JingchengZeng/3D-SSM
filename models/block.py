import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None


class LightweightModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightModel, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class LGF_Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        conv_mode = "deepwise",
        act = "silu"
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.conv_mode = conv_mode
        self.act = act
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type

        if self.conv_mode == "orignal":
            self.lcoal_relation = nn.Sequential(
                nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=self.d_model, out_channels=self.d_inner, kernel_size=3, stride=1, padding=1),
            )
        elif self.conv_mode == "orignal_dinner":
            self.lcoal_relation = nn.Sequential(
                nn.Conv2d(in_channels=self.d_model, out_channels=self.d_inner, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=3, stride=1, padding=1),
            )
        elif self.conv_mode == "orignal_1_5_dmodel":
            self.lcoal_relation = nn.Sequential(
                nn.Conv2d(in_channels=self.d_model, out_channels=int(1.5*self.d_model), kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=int(1.5*self.d_model), out_channels=self.d_inner, kernel_size=3, stride=1, padding=1),
            )
        elif self.conv_mode == "deepwise":
            self.lcoal_relation = nn.Sequential(
                LightweightModel(in_channels=self.d_model, out_channels=self.d_model),
                nn.SiLU(),
                LightweightModel(in_channels=self.d_model, out_channels=self.d_inner),
            )
        elif self.conv_mode == "deepwise_dinner":
            self.lcoal_relation = nn.Sequential(
                LightweightModel(in_channels=self.d_model, out_channels=self.d_inner),
                nn.SiLU(),
                LightweightModel(in_channels=self.d_inner, out_channels=self.d_inner),
            )

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        if self.act == "silu":
            self.act = nn.SiLU()
        elif self.act == "relu":
            self.act = nn.ReLU()
        elif self.act == "swish":
            self.act = nn.Hardswish()
        elif self.act == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.act == "lekyrelu":
            self.act = nn.LeakyReLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        assert bimamba_type == "v2"

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        # channel scan
        self.c_inner = 3072 // self.d_model
        self.conv1ch = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_ch = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_ch = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_ch = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_ch_log = torch.log(A_ch)
        self.A_ch_log = nn.Parameter(A_ch_log)
        self.A_ch_log._no_weight_decay = True

        self.D_ch = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_ch._no_weight_decay = True

        self.conv1ch_b = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_chb = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_chb = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_chb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_chb_log = torch.log(A_chb)
        self.A_chb_log = nn.Parameter(A_chb_log)
        self.A_chb_log._no_weight_decay = True

        self.D_chb = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_chb._no_weight_decay = True

        self.conv1cw = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_cw = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_cw = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_cw = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_cw_log = torch.log(A_cw)
        self.A_cw_log = nn.Parameter(A_cw_log)
        self.A_cw_log._no_weight_decay = True

        self.D_cw = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_cw._no_weight_decay = True

        self.conv1cw_b = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_cwb = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_cwb = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_cwb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_cwb_log = torch.log(A_cwb)
        self.A_cwb_log = nn.Parameter(A_cwb_log)
        self.A_cwb_log._no_weight_decay = True

        self.D_cwb = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_cwb._no_weight_decay = True


    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        another_hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        h = int(math.sqrt(seqlen))

        another_xz = hidden_states.clone()
        another_xz = self.lcoal_relation(rearrange(another_xz, "b (h w) d -> b d h w", h=h))
        out_conv = rearrange(another_xz, "b d h w -> b d (h w)")

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        
        channel_h = rearrange(xz, "b d (h w) -> b h (d w)", h=h)
        channel_w = rearrange(xz, "b d (h w) -> b w (d h)", h=h)

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v2":
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                A_b = -torch.exp(self.A_b_log.float())
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                A_ch = -torch.exp(self.A_ch_log.float())
                out_ch = mamba_inner_fn_no_out_proj(
                    channel_h,
                    self.conv1ch.weight,
                    self.conv1ch.bias,
                    self.x_proj_ch.weight,
                    self.dt_proj_ch.weight,
                    A_ch,
                    None,
                    None,
                    self.D_ch.float(),
                    delta_bias=self.dt_proj_ch.bias.float(),
                    delta_softplus=True,
                )
                out_ch = rearrange(out_ch, "b h (d w) -> b d (h w)", w=h*2)
                A_ch_b = -torch.exp(self.A_chb_log.float())
                out_chb = mamba_inner_fn_no_out_proj(
                    channel_h.flip([-1]),
                    self.conv1ch_b.weight,
                    self.conv1ch_b.bias,
                    self.x_proj_chb.weight,
                    self.dt_proj_chb.weight,
                    A_ch_b,
                    None,
                    None,
                    self.D_chb.float(),
                    delta_bias=self.dt_proj_chb.bias.float(),
                    delta_softplus=True,
                )
                out_chb = rearrange(out_chb, "b h (d w) -> b d (h w)", w=h*2)
                A_cw = -torch.exp(self.A_cw_log.float())
                out_cw = mamba_inner_fn_no_out_proj(
                    channel_w,
                    self.conv1cw.weight,
                    self.conv1cw.bias,
                    self.x_proj_cw.weight,
                    self.dt_proj_cw.weight,
                    A_cw,
                    None,
                    None,
                    self.D_cw.float(),
                    delta_bias=self.dt_proj_cw.bias.float(),
                    delta_softplus=True,
                )
                out_cw = rearrange(out_cw, "b h (d w) -> b d (h w)", w=h*2)
                A_cwb = -torch.exp(self.A_cwb_log.float())
                out_cwb = mamba_inner_fn_no_out_proj(
                    channel_w.flip([-1]),
                    self.conv1cw_b.weight,
                    self.conv1cw_b.bias,
                    self.x_proj_cwb.weight,
                    self.dt_proj_cwb.weight,
                    A_cwb,
                    None,
                    None,
                    self.D_cwb.float(),
                    delta_bias=self.dt_proj_cwb.bias.float(),
                    delta_softplus=True,
                )
                out_cwb = rearrange(out_cwb, "b h (d w) -> b d (h w)", w=h*2)
                out_s6 = out + out_b.flip([-1]) + out_ch + out_chb.flip([-1]) + out_cw + out_cwb.flip([-1])
                # out_s6 = out_ch + out_chb.flip([-1]) + out_cw + out_cwb.flip([-1])

            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_s6 = out
        
        return out_s6, out_conv

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class SS3D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none"
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type

        self.local_relation = nn.Sequential(
            LightweightModel(in_channels=self.d_model, out_channels=self.d_model),
            nn.SiLU(),
            LightweightModel(in_channels=self.d_model, out_channels=self.d_inner),
        )

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        assert bimamba_type == "v2"

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        # channel scan
        self.c_inner = 3072 // self.d_model
        self.conv1ch = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_ch = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_ch = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_ch = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_ch_log = torch.log(A_ch)
        self.A_ch_log = nn.Parameter(A_ch_log)
        self.A_ch_log._no_weight_decay = True

        self.D_ch = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_ch._no_weight_decay = True

        self.conv1ch_b = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_chb = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_chb = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_chb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_chb_log = torch.log(A_chb)
        self.A_chb_log = nn.Parameter(A_chb_log)
        self.A_chb_log._no_weight_decay = True

        self.D_chb = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_chb._no_weight_decay = True

        self.conv1cw = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_cw = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_cw = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_cw = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_cw_log = torch.log(A_cw)
        self.A_cw_log = nn.Parameter(A_cw_log)
        self.A_cw_log._no_weight_decay = True

        self.D_cw = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_cw._no_weight_decay = True

        self.conv1cw_b = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_cwb = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_cwb = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_cwb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_cwb_log = torch.log(A_cwb)
        self.A_cwb_log = nn.Parameter(A_cwb_log)
        self.A_cwb_log._no_weight_decay = True

        self.D_cwb = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_cwb._no_weight_decay = True

        # out
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # FFT
        self.project_in = nn.Conv2d(self.d_model, self.d_model * 6, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(self.d_model * 6, self.d_model * 6, kernel_size=3, stride=1, padding=1,
                                groups=self.d_model * 6, bias=bias)
        self.fft = nn.Parameter(torch.ones((self.d_model * 6, 1, 1, 8, 8 // 2 + 1)))
        self.project_out = nn.Conv2d(self.d_model * 3, self.d_inner, kernel_size=1, bias=bias)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        h = int(math.sqrt(seqlen))

        # 卷积分支
        local_relation = self.local_relation(rearrange(hidden_states, "b (h w) d -> b d h w", h=h))
        local_relation = rearrange(local_relation, "b d h w -> b d (h w)")

        # fft分支
        frequency = rearrange(hidden_states, "b (h w) d -> b d h w", h=h)
        frequency = self.project_in(frequency)
        x_patch = rearrange(frequency, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', 
                            patch1=8, patch2=8)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(8, 8))
        frequency = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', 
                              patch1=8, patch2=8)
        x1, x2 = self.dwconv(frequency).chunk(2, dim=1)
        frequency = F.gelu(x1) * x2
        frequency = self.project_out(frequency)
        frequency = rearrange(frequency, "b d h w -> b d (h w)")

        # mamba分支
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLD -> DBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        channel_h = rearrange(xz, "b d (h w) -> b h (d w)", h=h)
        channel_w = rearrange(xz, "b d (h w) -> b w (d h)", h=h)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # out_all1 = []
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v2":
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                A_b = -torch.exp(self.A_b_log.float())
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                A_ch = -torch.exp(self.A_ch_log.float())
                out_ch = mamba_inner_fn_no_out_proj(
                    channel_h,
                    self.conv1ch.weight,
                    self.conv1ch.bias,
                    self.x_proj_ch.weight,
                    self.dt_proj_ch.weight,
                    A_ch,
                    None,
                    None,
                    self.D_ch.float(),
                    delta_bias=self.dt_proj_ch.bias.float(),
                    delta_softplus=True,
                )
                out_ch = rearrange(out_ch, "b h (d w) -> b d (h w)", w=h*2)
                A_ch_b = -torch.exp(self.A_chb_log.float())
                out_chb = mamba_inner_fn_no_out_proj(
                    channel_h.flip([-1]),
                    self.conv1ch_b.weight,
                    self.conv1ch_b.bias,
                    self.x_proj_chb.weight,
                    self.dt_proj_chb.weight,
                    A_ch_b,
                    None,
                    None,
                    self.D_chb.float(),
                    delta_bias=self.dt_proj_chb.bias.float(),
                    delta_softplus=True,
                )
                out_chb = rearrange(out_chb, "b h (d w) -> b d (h w)", w=h*2)
                A_cw = -torch.exp(self.A_cw_log.float())
                out_cw = mamba_inner_fn_no_out_proj(
                    channel_w,
                    self.conv1cw.weight,
                    self.conv1cw.bias,
                    self.x_proj_cw.weight,
                    self.dt_proj_cw.weight,
                    A_cw,
                    None,
                    None,
                    self.D_cw.float(),
                    delta_bias=self.dt_proj_cw.bias.float(),
                    delta_softplus=True,
                )
                out_cw = rearrange(out_cw, "b h (d w) -> b d (h w)", w=h*2)
                A_cwb = -torch.exp(self.A_cwb_log.float())
                out_cwb = mamba_inner_fn_no_out_proj(
                    channel_w.flip([-1]),
                    self.conv1cw_b.weight,
                    self.conv1cw_b.bias,
                    self.x_proj_cwb.weight,
                    self.dt_proj_cwb.weight,
                    A_cwb,
                    None,
                    None,
                    self.D_cwb.float(),
                    delta_bias=self.dt_proj_cwb.bias.float(),
                    delta_softplus=True,
                )
                out_cwb = rearrange(out_cwb, "b h (d w) -> b d (h w)", w=h*2)
                out_all = out + out_b.flip([-1]) + out_ch + out_chb.flip([-1]) + out_cw + out_cwb.flip([-1])
                out = F.linear(rearrange(out_all + local_relation + frequency, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                # out = F.linear(rearrange(out_all + local_relation, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class SS3D_Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_state = d_state
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        # channel scan
        self.c_inner = 3072 // self.d_model
        self.conv1ch = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_ch = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_ch = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_ch = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_ch_log = torch.log(A_ch)
        self.A_ch_log = nn.Parameter(A_ch_log)
        self.A_ch_log._no_weight_decay = True

        self.D_ch = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_ch._no_weight_decay = True

        self.conv1ch_b = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_chb = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_chb = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_chb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_chb_log = torch.log(A_chb)
        self.A_chb_log = nn.Parameter(A_chb_log)
        self.A_chb_log._no_weight_decay = True

        self.D_chb = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_chb._no_weight_decay = True

        self.conv1cw = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_cw = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_cw = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_cw = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_cw_log = torch.log(A_cw)
        self.A_cw_log = nn.Parameter(A_cw_log)
        self.A_cw_log._no_weight_decay = True

        self.D_cw = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_cw._no_weight_decay = True

        self.conv1cw_b = nn.Conv1d(
            in_channels=self.c_inner,
            out_channels=self.c_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.c_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_cwb = nn.Linear(
            self.c_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_cwb = nn.Linear(self.dt_rank, self.c_inner, bias=True, **factory_kwargs)

        A_cwb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.c_inner,
        ).contiguous()
        A_cwb_log = torch.log(A_cwb)
        self.A_cwb_log = nn.Parameter(A_cwb_log)
        self.A_cwb_log._no_weight_decay = True

        self.D_cwb = nn.Parameter(torch.ones(self.c_inner, device=device))
        self.D_cwb._no_weight_decay = True
        
        # out
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        h = int(math.sqrt(seqlen))

        # We do matmul and transpose BLD -> DBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        channel_h = rearrange(xz, "b d (h w) -> b h (d w)", h=h)
        channel_w = rearrange(xz, "b d (h w) -> b w (d h)", h=h)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        out = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        A_b = -torch.exp(self.A_b_log.float())
        out_b = mamba_inner_fn_no_out_proj(
            xz.flip([-1]),
            self.conv1d_b.weight,
            self.conv1d_b.bias,
            self.x_proj_b.weight,
            self.dt_proj_b.weight,
            A_b,
            None,
            None,
            self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
        )
        A_ch = -torch.exp(self.A_ch_log.float())
        out_ch = mamba_inner_fn_no_out_proj(
            channel_h,
            self.conv1ch.weight,
            self.conv1ch.bias,
            self.x_proj_ch.weight,
            self.dt_proj_ch.weight,
            A_ch,
            None,
            None,
            self.D_ch.float(),
            delta_bias=self.dt_proj_ch.bias.float(),
            delta_softplus=True,
        )
        out_ch = rearrange(out_ch, "b h (d w) -> b d (h w)", w=h*2)
        A_ch_b = -torch.exp(self.A_chb_log.float())
        out_chb = mamba_inner_fn_no_out_proj(
            channel_h.flip([-1]),
            self.conv1ch_b.weight,
            self.conv1ch_b.bias,
            self.x_proj_chb.weight,
            self.dt_proj_chb.weight,
            A_ch_b,
            None,
            None,
            self.D_chb.float(),
            delta_bias=self.dt_proj_chb.bias.float(),
            delta_softplus=True,
        )
        out_chb = rearrange(out_chb, "b h (d w) -> b d (h w)", w=h*2)
        A_cw = -torch.exp(self.A_cw_log.float())
        out_cw = mamba_inner_fn_no_out_proj(
            channel_w,
            self.conv1cw.weight,
            self.conv1cw.bias,
            self.x_proj_cw.weight,
            self.dt_proj_cw.weight,
            A_cw,
            None,
            None,
            self.D_cw.float(),
            delta_bias=self.dt_proj_cw.bias.float(),
            delta_softplus=True,
        )
        out_cw = rearrange(out_cw, "b h (d w) -> b d (h w)", w=h*2)
        A_cwb = -torch.exp(self.A_cwb_log.float())
        out_cwb = mamba_inner_fn_no_out_proj(
            channel_w.flip([-1]),
            self.conv1cw_b.weight,
            self.conv1cw_b.bias,
            self.x_proj_cwb.weight,
            self.dt_proj_cwb.weight,
            A_cwb,
            None,
            None,
            self.D_cwb.float(),
            delta_bias=self.dt_proj_cwb.bias.float(),
            delta_softplus=True,
        )
        out_cwb = rearrange(out_cwb, "b h (d w) -> b d (h w)", w=h*2)
        out_all = out + out_b.flip([-1]) + out_ch + out_chb.flip([-1]) + out_cw + out_cwb.flip([-1])
        out = F.linear(rearrange(out_all, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        return out
