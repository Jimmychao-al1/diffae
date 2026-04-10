import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import math

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def differentiable_clamp(x, lower, upper):
    x = x + F.relu(lower - x)
    x = x - F.relu(x - upper)
    return x

def normalized_fake_quant(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8):
    """
    Normalized fake quantization: absmax normalize -> round to [-128,127] -> denormalize
    
    Args:
        x: input tensor
        scale: absmax scale (scalar or per-channel)
        eps: small value to avoid division by zero
        
    Returns:
        x_q: fake-quantized tensor (still float, range approximately [-1, 1])
    """
    scale = torch.clamp(scale, min=eps)
    x_norm = differentiable_clamp(x / scale, -1.0, 1.0)
    #x_norm = torch.clamp(x / scale, -1.0, 1.0)
    x_int = round_ste(x_norm * 127.0)
    #x_int = torch.clamp(x_int, -128.0, 127.0)
    x_q = x_int / 127.0  # normalized output in [-1, 1]
    return x_q


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    Normalized fake quantization (absmax-based, symmetric int8: [-128, 127])
    
    Changed from affine quantization to normalized quantization:
    - No longer uses delta/zero_point for [0, 255] mapping
    - Uses absmax scale to normalize to [-1, 1], then quantize to [-128, 127]
    - Returns normalized fake-quantized values in [-1, 1] range
    
    :param n_bits: number of bit for quantization (only 8-bit supported for now)
    :param channel_wise: if True, compute per-channel absmax
    :param scale_method: 'absmax' for normalized quantization
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = True, channel_wise: bool = False, scale_method: str = 'absmax',
                 leaf_param: bool = False, weight_tensor = None, need_init=True):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = True  # Always symmetric for normalized quant
        assert n_bits == 8, 'Only 8-bit supported for normalized quantization'
        self.n_bits = n_bits
        self.n_levels = 256  # [-128, 127]
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = 'absmax'  # Force absmax for normalized quant
        self.eps = 1e-8
        
        # Keep delta/zero_point for compatibility, but not used in new flow
        # scale (absmax) will be computed dynamically in forward
        self.inited = True  # No initialization needed for absmax
        self.delta = None  # Placeholder for compatibility
        self.zero_point = None
        
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x
    
    def forward(self, x: torch.Tensor):
        """
        Normalized fake quantization forward.
        Computes absmax dynamically, then applies normalized fake-quant.
        
        Returns normalized quantized tensor in approximately [-1, 1] range.
        """
        # Compute absmax scale dynamically
        if self.channel_wise:
            # Per-channel absmax (for weights: per out-channel)
            if len(x.shape) == 4:  # Conv weight: [Cout, Cin, H, W]
                scale = x.abs().amax(dim=(1, 2, 3), keepdim=True) + self.eps
            elif len(x.shape) == 2:  # Linear weight: [Cout, Cin]
                scale = x.abs().amax(dim=1, keepdim=True) + self.eps
            else:
                scale = x.abs().max() + self.eps
        else:
            # Per-tensor absmax (for activations)
            scale = x.abs().max() + self.eps
        
        # Apply normalized fake quantization
        x_q = normalized_fake_quant(x, scale, self.eps)
        
        return x_q



    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)

class INT_UniformAffineQuantizer(nn.Module):
    """
    Normalized fake quantization (absmax-based, symmetric int8: [-128, 127])
    
    Changed from affine quantization to normalized quantization:
    - No longer uses delta/zero_point for [0, 255] mapping
    - Uses absmax scale to normalize to [-1, 1], then quantize to [-128, 127]
    - Returns normalized fake-quantized values in [-1, 1] range
    
    :param n_bits: number of bit for quantization (only 8-bit supported for now)
    :param channel_wise: if True, compute per-channel absmax
    :param scale_method: 'absmax' for normalized quantization
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = True, channel_wise: bool = False, scale_method: str = 'absmax',
                 leaf_param: bool = False, weight_tensor = None, need_init=True):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = True  # Always symmetric for normalized quant
        assert n_bits == 8, 'Only 8-bit supported for normalized quantization'
        self.n_bits = n_bits
        self.n_levels = 256  # [-128, 127]
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = 'absmax'  # Force absmax for normalized quant
        self.eps = 1e-8
        
        # Keep delta/zero_point for compatibility, but not used in new flow
        # scale (absmax) will be computed dynamically in forward
        self.inited = True  # No initialization needed for absmax
        self.delta = None  # Placeholder for compatibility
        self.zero_point = None
        
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x
    
    def forward(self, x: torch.Tensor):
        """
        Normalized fake quantization forward.
        Computes absmax dynamically, then applies normalized fake-quant.
        
        Returns normalized quantized tensor in approximately [-1, 1] range.
        """
        # Compute absmax scale dynamically
        if self.channel_wise:
            # Per-channel absmax (for weights: per out-channel)
            if len(x.shape) == 4:  # Conv weight: [Cout, Cin, H, W]
                scale = x.abs().amax(dim=(1, 2, 3), keepdim=True) + self.eps
            elif len(x.shape) == 2:  # Linear weight: [Cout, Cin]
                scale = x.abs().amax(dim=1, keepdim=True) + self.eps
            else:
                scale = x.abs().max() + self.eps
        else:
            # Per-tensor absmax (for activations)
            scale = x.abs().max() + self.eps
        
        # Apply normalized fake quantization
        x_q = normalized_fake_quant(x, scale, self.eps)
        
        return x_q



    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)
    
class TemporalActivationQuantizer(nn.Module):
    """
    Temporal (per-diffusion-step) activation quantizer with trainable scale.
    
    Changed from delta_list/zp_list to scale_list (a_x[k]):
    - a_x[k] is a trainable scalar for each timestep k (0..99)
    - During forward: use a_x[current_step] as absmax scale
    - Calibration initializes a_x[k] from observed absmax
    
    :param n_bits: 8-bit only
    :param num_steps: total diffusion steps (e.g., 100)
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, num_steps = 100):
        super(TemporalActivationQuantizer, self).__init__()
        self.sym = True  # Always symmetric for normalized quant
        assert n_bits == 8, 'Only 8-bit supported for normalized quantization'
        self.n_bits = n_bits
        self.n_levels = 256  # [-128, 127]
        self.eps = 1e-8

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1  # Start from last step (99)

        # Trainable per-step scale: a_x[k] for k=0..num_steps-1
        # Initialize with small positive values (will be calibrated)
        self.scale_list = nn.Parameter(torch.ones(num_steps) * 0.1, requires_grad=True)
        
        self.inited = False  # Will be set True after first calibration pass
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = 'absmax'  # Force absmax
        
        # Compatibility placeholders (not used in normalized quant)
        self.delta = None
        self.zero_point = None
        self.calibration_mode = False
    
    def clipping(self, x, lower, upper):
        """Differentiable clipping for STE."""
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        return x
    
    def calibrate_step(self, x: torch.Tensor, step: int):
        """
        Calibrate scale for a specific timestep using observed absmax.
        
        Args:
            x: activation tensor
            step: timestep index (0..total_steps-1)
        """
        with torch.no_grad():
            absmax = x.detach().abs().max().clamp(min=self.eps)
            self.scale_list.data[step] = absmax.to(self.scale_list.data.dtype)
    
    def forward(self, x: torch.Tensor):
        """
        Temporal normalized fake quantization.
        Uses scale_list[current_step] as absmax scale.

        Returns normalized quantized tensor in approximately [-1, 1] range.
        """
        if self.calibration_mode:
            self.calibrate_step(x, self.current_step)
            self.current_step = self.total_steps - 1 if self.current_step - 1 < 0 else self.current_step - 1
            return x

        if not self.inited:
            # Keep behavior aligned with original implementation:
            # first forward initializes per-step parameters and does NOT decrement current_step.
            self.inited = True
            self.calibrate_step(x, self.current_step)
            with torch.no_grad():
                init_scale = self.scale_list[self.current_step].detach().clamp(min=self.eps)
                self.scale_list.data.copy_(torch.ones_like(self.scale_list.data) * init_scale)

            scale = self.scale_list[self.current_step].clamp(min=self.eps)
            return normalized_fake_quant(x, scale, self.eps)

        # Normal path: use per-step scale and auto-decrement.
        step_idx = max(0, min(self.current_step, self.total_steps - 1))
        scale = self.scale_list[step_idx].clamp(min=self.eps)
        x_q = normalized_fake_quant(x, scale, self.eps)
        self.current_step = self.total_steps - 1 if self.current_step - 1 < 0 else self.current_step - 1
        return x_q

    def bitwidth_refactor(self, refactored_bit: int):
        """Compatibility method (not used in v2)."""
        assert refactored_bit == 8, 'Only 8-bit supported'
        self.n_bits = refactored_bit
        self.n_levels = 256

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method=absmax, symmetric=True, total_steps={total_steps}'
        return s.format(**self.__dict__)
    
class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None, need_init=True):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        if not need_init:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, weight_tensor=self.weight)
        else:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, need_init=need_init)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        # runtime_mode:
        # - train: dynamic a_w every forward
        # - infer: optionally use cached a_w
        self.runtime_mode = 'train'
        self.use_cached_aw = False
        self.cached_a_w = None

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def _compute_a_w(self, weight: torch.Tensor):
        if len(weight.shape) == 4:  # Conv2d: [Cout, Cin, H, W]
            return weight.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        if len(weight.shape) == 2:  # Linear: [Cout, Cin]
            return weight.abs().amax(dim=1, keepdim=True) + 1e-8
        return weight.abs().max() + 1e-8

    def _get_a_w(self, weight: torch.Tensor):
        if self.runtime_mode == 'infer' and self.use_cached_aw:
            if self.cached_a_w is None:
                self.cached_a_w = self._compute_a_w(weight).detach()
            return self.cached_a_w.to(weight.device)
        return self._compute_a_w(weight)

    def forward(self, input: torch.Tensor):
        """
        Forward pass with normalized fake-quant + rescale + FP32 bias.
        
        Flow:
        1. Quantize activation to x_norm (in [-1,1])
        2. Quantize weight to w_norm (in [-1,1])
        3. Compute y_norm = conv/linear(x_norm, w_norm, bias=None)
        4. Rescale: y = y_norm * (a_x * a_w)
        5. Add FP32 bias: y += bias_fp32
        """
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
        
        # Keep original bias in FP32
        bias_fp32 = self.bias if self.bias is not None else None
        
        if self.use_weight_quant and self.use_act_quant:
            # Both quant enabled: normalized quant + rescale
            # Step 1: Compute activation scale (a_x)
            a_x = input.abs().max().clamp(min=1e-8)
            
            # Step 2: Quantize activation to normalized range
            x_norm = normalized_fake_quant(input, a_x, eps=1e-8)
            
            # Step 3: Compute weight scale (a_w)
            a_w = self._get_a_w(self.weight)
            
            # Step 4: Quantize weight to normalized range
            w_norm = normalized_fake_quant(self.weight, a_w, eps=1e-8)
            
            # Step 5: Compute normalized output (bias=None)
            y_norm = self.fwd_func(x_norm, w_norm, None, **self.fwd_kwargs)
            
            # Step 6: Rescale to original scale
            # For conv2d: a_w shape [Cout,1,1,1] -> broadcast to [1,Cout,1,1]
            # For linear: a_w shape [Cout,1] -> broadcast to [1,Cout]
            if len(self.weight.shape) == 4:
                scale_factor = (a_x * a_w).view(1, -1, 1, 1)
            else:
                scale_factor = (a_x * a_w).view(1, -1)
            
            out = y_norm * scale_factor
            
            # Step 7: Add FP32 bias (if exists)
            if bias_fp32 is not None:
                if len(self.weight.shape) == 4:
                    out = out + bias_fp32.view(1, -1, 1, 1)
                else:
                    out = out + bias_fp32.view(1, -1)
        
        elif self.use_weight_quant and not self.use_act_quant:
            # Only weight quant:
            # w_norm = quant(w / a_w) in [-1,1], so output must be rescaled by a_w.
            a_w = self._get_a_w(self.weight)
            w_norm = normalized_fake_quant(self.weight, a_w, eps=1e-8)
            y_norm = self.fwd_func(input, w_norm, None, **self.fwd_kwargs)

            if len(self.weight.shape) == 4:
                scale_factor = a_w.view(1, -1, 1, 1)
            else:
                scale_factor = a_w.view(1, -1)
            out = y_norm * scale_factor

            if bias_fp32 is not None:
                if len(self.weight.shape) == 4:
                    out = out + bias_fp32.view(1, -1, 1, 1)
                else:
                    out = out + bias_fp32.view(1, -1)
        
        elif not self.use_weight_quant and self.use_act_quant:
            # Only act quant:
            # x_norm = quant(x / a_x) in [-1,1], so output must be rescaled by a_x.
            a_x = input.abs().max().clamp(min=1e-8)
            input = normalized_fake_quant(input, a_x, eps=1e-8)
            weight = self.org_weight.to(input.device)
            y_norm = self.fwd_func(input, weight, None, **self.fwd_kwargs)
            out = y_norm * a_x

            if bias_fp32 is not None:
                if len(self.weight.shape) == 4:
                    out = out + bias_fp32.view(1, -1, 1, 1)
                else:
                    out = out + bias_fp32.view(1, -1)
        
        else:
            # No quantization
            weight = self.org_weight.to(input.device)
            bias = self.org_bias.to(input.device) if self.org_bias is not None else None
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_runtime_mode(self, mode: str = 'train', use_cached_aw: bool = False, clear_cached_aw: bool = False):
        """
        Set runtime behavior for a_w strategy.
        """
        if mode not in ('train', 'infer'):
            raise ValueError(f"Unsupported runtime mode: {mode}")
        self.runtime_mode = mode
        self.use_cached_aw = use_cached_aw
        if clear_cached_aw:
            self.cached_a_w = None





class SimpleDequantizer(nn.Module):

    def __init__(self, uaq: UniformAffineQuantizer, weight):
        super(SimpleDequantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        # self.n_bits = uaq.n_bits
        self.n_bits = torch.tensor(uaq.n_bits, dtype=torch.int8)
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.ori_shape = weight.shape

        self.size_scale = int(8 // self.n_bits)

        if len(weight.shape) == 4:
            self.delta = nn.Parameter(torch.randn(size=(weight.shape[0]*self.size_scale, 1, 1, 1)), requires_grad=False)
            self.zero_point = nn.Parameter(torch.randn(size=(weight.shape[0]*self.size_scale, 1, 1, 1)), requires_grad=False)
        elif len(weight.shape) == 2:
            self.delta = nn.Parameter(torch.randn(size=(weight.shape[0]*self.size_scale, 1)), requires_grad=False)
            self.zero_point = nn.Parameter(torch.randn(size=(weight.shape[0]*self.size_scale, 1)), requires_grad=False)
        else:
            print(weight.shape)
            raise ValueError('shape not implemented')

        self.gap = torch.tensor(list(range(0, 8, self.n_bits)), dtype=torch.uint8, device='cuda').unsqueeze(0)


    def forward(self, x_int8):
        ## unpack
        #if len(x_int_pack8.shape) == 4:
        #  pass
        pass


# ============================================================
# INT-path helpers for INT_QuantModule_DiffAE_LoRA forward
#
# ACCUMULATION SEMANTICS:
#   torch._int_mm(int8, int8) → int32   ← TRUE int32 accumulation
#   verified: max_abs_diff vs int64 reference = 0 (PyTorch 2.4.1, CPU)
#
# DTYPE CHAIN (per function):
#   _norm_to_int_code : float32  →  int32
#   _int_linear_accum : int32 → int8 → _int_mm → int32
#   _int_conv2d_accum : int32 → int8 → unfold(float tmp) + _int_mm → int32
#
# Note on unfold:
#   F.unfold does not accept integer tensors; the col extraction is done
#   in float32 (a shape rearrangement, no arithmetic), then immediately
#   cast back to int8 before _int_mm.  The multiply-accumulate itself
#   is int8 × int8 → int32 inside _int_mm.
# ============================================================

def _norm_to_int_code(x_norm: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    """
    float32 normalized [-1,1]  →  int32 code in [-qmax, qmax].

    input  dtype : float32
    output dtype : int32
    """
    return torch.round(torch.clamp(x_norm, -1.0, 1.0) * qmax).to(torch.int32)


def _int_mm_safe(a_i8: torch.Tensor, b_i8: torch.Tensor) -> torch.Tensor:
    """
    Safe wrapper for torch._int_mm(int8, int8) -> int32.

    Some CUDA kernels fail for small M (e.g., "self.size(0) needs to be greater than 16").
    In that case, fall back to exact integer semantics via int64 matmul and cast to int32.
    """
    assert a_i8.dtype == torch.int8, f"a_i8 dtype must be int8, got {a_i8.dtype}"
    assert b_i8.dtype == torch.int8, f"b_i8 dtype must be int8, got {b_i8.dtype}"
    try:
        return torch._int_mm(a_i8, b_i8)
    except RuntimeError as e:
        msg = str(e)
        if "self.size(0) needs to be greater than 16" in msg:
            return (a_i8.to(torch.int64) @ b_i8.to(torch.int64)).to(torch.int32)
        raise


def _int_linear_accum(x_int: torch.Tensor, w_int: torch.Tensor) -> torch.Tensor:
    """
    True int32 accumulation for linear:  y_int = x_int @ w_int.T

    Dtype chain:
      x_int  : int32,  values in [-127, 127]
      w_int  : int32,  values in [-127, 127]
      x_i8   : int8   (safe cast, no overflow)
      w_i8   : int8   (safe cast, no overflow)
      y_flat : int32  (torch._int_mm output)
      return : int32

    torch._int_mm(int8, int8) → int32 guarantees exact integer
    accumulation (verified vs int64 reference, diff = 0).

    x_int : [..., in_features]          int32
    w_int : [out_features, in_features] int32
    returns: [..., out_features]        int32
    """
    assert x_int.dtype == torch.int32, f"x_int dtype must be int32, got {x_int.dtype}"
    assert w_int.dtype == torch.int32, f"w_int dtype must be int32, got {w_int.dtype}"

    orig_shape = x_int.shape
    # [N, Cin]
    x_flat = x_int.reshape(-1, orig_shape[-1])

    # int32 → int8 (values already in [-127,127], no clamp needed)
    x_i8 = x_flat.to(torch.int8).contiguous()          # int8
    w_i8 = w_int.to(torch.int8).t().contiguous()        # int8, [Cin, Cout]

    # TRUE INT32 ACCUMULATION
    y_flat = _int_mm_safe(x_i8, w_i8)                   # int32, [N, Cout]

    assert y_flat.dtype == torch.int32
    return y_flat.reshape(orig_shape[:-1] + (w_int.shape[0],))


def _int_conv2d_accum(
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    stride,
    padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    """
    True int32 accumulation for conv2d via unfold + torch._int_mm.

    Dtype chain (groups=1):
      x_int     : int32  [N, Cin, H, W]
      x_fp      : float32  (F.unfold requires float; shape rearrangement only, no multiply)
      x_col     : float32  [N, K, HW]   where K = Cin*kH*kW
      x_col_2d  : float32  [K, N*HW]
      x_col_i8  : int8     [K, N*HW]   (safe cast back to int8)
      w_flat    : int32    [Cout, K]
      w_i8      : int8     [Cout, K]
      y_2d      : int32    [Cout, N*HW]   ← TRUE INT32 ACCUMULATION
      return    : int32    [N, Cout, Hout, Wout]

    groups > 1: above applied per-group, results cat'd.

    x_int : [N, Cin, H, W]              int32
    w_int : [Cout, Cin//groups, kH, kW] int32
    returns: [N, Cout, Hout, Wout]      int32
    """
    assert x_int.dtype == torch.int32, f"x_int dtype must be int32, got {x_int.dtype}"
    assert w_int.dtype == torch.int32, f"w_int dtype must be int32, got {w_int.dtype}"

    N, Cin, H, W = x_int.shape
    Cout = w_int.shape[0]
    kH, kW = w_int.shape[2], w_int.shape[3]

    _stride = (stride, stride)   if isinstance(stride, int)  else tuple(stride)
    _pad    = (padding, padding) if isinstance(padding, int) else tuple(padding)
    _dil    = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)

    Hout = (H + 2 * _pad[0] - _dil[0] * (kH - 1) - 1) // _stride[0] + 1
    Wout = (W + 2 * _pad[1] - _dil[1] * (kW - 1) - 1) // _stride[1] + 1
    HW   = Hout * Wout

    def _group_int_mm(x_g_int32, w_g_int32, Cout_g):
        """Per-group true int32 conv via unfold + _int_mm."""
        # F.unfold requires float; this is shape rearrangement only (no multiply)
        x_col = F.unfold(                                # float32 [N, K, HW]
            x_g_int32.float(),
            kernel_size=(kH, kW), dilation=_dil, padding=_pad, stride=_stride)
        K = x_col.shape[1]                               # Cin_g * kH * kW

        # [K, N*HW] float32 → int8   (back to int domain before multiply)
        x_col_2d = x_col.permute(1, 0, 2).reshape(K, N * HW)  # float32
        x_col_i8 = x_col_2d.to(torch.int8).contiguous()        # int8

        w_i8 = w_g_int32.reshape(Cout_g, -1).to(torch.int8).contiguous()  # int8 [Cout_g, K]

        # TRUE INT32 ACCUMULATION: int8 × int8 → int32
        y_2d = _int_mm_safe(w_i8, x_col_i8)            # int32 [Cout_g, N*HW]
        return y_2d.reshape(Cout_g, N, HW).permute(1, 0, 2)  # int32 [N, Cout_g, HW]

    if groups == 1:
        y = _group_int_mm(x_int, w_int, Cout)           # int32 [N, Cout, HW]
    else:
        Cin_g  = Cin  // groups
        Cout_g = Cout // groups
        outs = []
        for g in range(groups):
            x_g = x_int[:, g * Cin_g:(g + 1) * Cin_g, :, :]
            w_g = w_int[g * Cout_g:(g + 1) * Cout_g, :, :, :]
            outs.append(_group_int_mm(x_g, w_g, Cout_g))
        y = torch.cat(outs, dim=1)                      # int32 [N, Cout, HW]

    assert y.dtype == torch.int32
    return y.reshape(N, Cout, Hout, Wout)