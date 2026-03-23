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