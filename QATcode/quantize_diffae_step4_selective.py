import os
import sys
import time

from torch.utils.data import DataLoader

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from QATcode.quant_dataset import DiffusionInputDataset
from QATcode.quant_utils import get_default_device
from QATcode.quant_layer import QuantModule
from QATcode.quant_model_selective import (
    SelectiveQuantModel,
    analyze_unet_structure,
    count_quantized_modules,
)
from templates import *
from templates_latent import *

# 量化配置
n_bits_w = 8
n_bits_a = 8
quantize_skip_connections = False  # 設為 True 可量化 skip connections

# ActivationCollector - optimized for CPU multi-core stat computation
import os
import json
import math
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

import os
import json
import time
import math
import numpy as np
import torch
import multiprocessing as mp

# ---------------- Module-level worker: 可被 multiprocessing pickle ----------------
def _compute_layer_stats_cpu(task):
    """
    task: (base, pre_arr, post_arr, meta, quantiles)
    returns: (base, stats_dict) where stats_dict contains 'meta','pre','post','errors'
    """
    import numpy as _np
    base, pre_arr, post_arr, meta, quantiles = task

    pre = _np.asarray(pre_arr, dtype=_np.float64) if pre_arr is not None else _np.asarray([], dtype=_np.float64)
    post = _np.asarray(post_arr, dtype=_np.float64) if post_arr is not None else _np.asarray([], dtype=_np.float64)

    def _stat(arr):
        if arr.size == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "quantiles": {}}
        qmap = {}
        for q in quantiles:
            try:
                qmap[str(q)] = float(_np.quantile(arr, q))
            except Exception:
                qmap[str(q)] = None
        return {"count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()),
                "mean": float(_np.mean(arr)), "std": float(_np.std(arr)), "quantiles": qmap}

    pre_stat = _stat(pre)
    post_stat = _stat(post)

    # errors
    errors = {"mse": None, "max_abs": None, "median_abs": None, "pct_clipped": None, "range_coverage": None}
    if pre.size > 0 and post.size > 0:
        n = min(pre.size, post.size)
        # deterministic sampling by base name hash
        rng = _np.random.default_rng(abs(hash(base)) % (2**32))
        idx_pre = rng.choice(pre.size, size=n, replace=False)
        idx_post = rng.choice(post.size, size=n, replace=False)
        a_pre = pre[idx_pre]
        a_post = post[idx_post]
        dif = a_post - a_pre
        errors["mse"] = float(_np.mean(dif**2))
        errors["max_abs"] = float(_np.max(_np.abs(dif)))
        errors["median_abs"] = float(_np.median(_np.abs(dif)))

    # pct_clipped & range_coverage using meta if available
    try:
        act_delta = meta.get("act_delta", None)
        act_zp = meta.get("act_zero_point", None)
        nb = meta.get("n_bits", None)
        if nb is not None and act_delta is not None:
            # extract scalar delta/zp (if list -> average)
            if isinstance(act_delta, (list, tuple, _np.ndarray)):
                delta_c = float(_np.mean(_np.asarray(act_delta, dtype=_np.float64)))
            else:
                delta_c = float(act_delta)
            if isinstance(act_zp, (list, tuple, _np.ndarray)):
                zp_c = int(_np.round(_np.mean(_np.asarray(act_zp))))
            else:
                zp_c = int(act_zp) if act_zp is not None else 0
            nb = int(nb)
            # infer qmin/qmax
            if zp_c is not None and 0 <= zp_c <= (2**nb - 1):
                qmin = 0; qmax = 2**nb - 1
            else:
                qmin = -(2**(nb-1)); qmax = (2**(nb-1)) - 1
            rep_min_f = (qmin - zp_c) * delta_c
            rep_max_f = (qmax - zp_c) * delta_c
            if post.size > 0:
                errors["pct_clipped"] = float(((post < rep_min_f) | (post > rep_max_f)).sum()) / float(post.size)
            if pre.size > 0:
                try:
                    obs_q = float(_np.quantile(pre, 0.999))
                    if obs_q != 0:
                        errors["range_coverage"] = float(abs(rep_max_f) / abs(obs_q))
                except Exception:
                    errors["range_coverage"] = None
    except Exception:
        # best-effort: keep errors fields as None if fails
        pass

    stats = {"meta": meta, "pre": pre_stat, "post": post_stat, "errors": errors}
    return (base, stats)
# ------------------------------------------------------------------------------------

class ActivationCollector:
    def __init__(self, model,
                 reservoir_size=20000,
                 sample_size_per_activation=2048,
                 per_channel_sample=512,
                 quantiles=(0.5, 0.99, 0.999),
                 out_dir="./analysis_step4",
                 run_tag="step4",
                 topk=10,
                 sample_seed=0):
        self.model = model
        self.reservoir_size = int(reservoir_size)
        self.sample_size = int(sample_size_per_activation)
        self.per_channel_sample = int(per_channel_sample) if per_channel_sample is not None else 0
        self.quantiles = tuple(quantiles)
        self.out_dir = out_dir
        self.run_tag = run_tag
        self.topk = int(topk)
        self.sample_seed = int(sample_seed)

        self.buff_pre = {}
        self.buff_post = {}
        self.meta = {}
        self._mode = "pre"
        self._handles = []
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        for name, mod in self.model.named_modules():
            try:
                if hasattr(mod, "act_quantizer"):
                    h = mod.register_forward_hook(self._make_hook(name))
                    self._handles.append(h)
                    self.meta.setdefault(name, {})
            except Exception:
                continue

    # (hook implementation unchanged from your last accepted version)
    def _make_hook(self, name):
        def _hook(module, inp, out):
            try:
                t = None
                if isinstance(out, (list, tuple)):
                    for v in out:
                        if isinstance(v, torch.Tensor):
                            t = v; break
                elif isinstance(out, torch.Tensor):
                    t = out
                if t is None:
                    return
            except Exception:
                return

            try:
                td = t.detach()
                if self.per_channel_sample and td.dim() >= 3:
                    if td.dim() >= 4:
                        perm = list(range(td.dim()))
                        perm[0], perm[1] = perm[1], perm[0]
                        tc = td.permute(*perm).contiguous().view(td.shape[1], -1)
                        C = tc.shape[0]
                        for c in range(C):
                            vec = tc[c]
                            clen = vec.numel()
                            k = min(self.per_channel_sample, clen, max(1, int(self.sample_size // max(1, C))))
                            if clen <= k:
                                samp = vec.reshape(-1)
                            else:
                                idx = torch.randperm(clen, device=vec.device)[:k]
                                samp = vec.reshape(-1)[idx]
                            arr = samp.cpu().numpy()
                            self._append_reservoir(name + f"__ch{c}", arr, mode=self._mode)
                    else:
                        C = td.shape[0]
                        for c in range(C):
                            vec = td[c].reshape(-1)
                            clen = vec.numel()
                            k = min(self.per_channel_sample, clen, max(1, int(self.sample_size // max(1, C))))
                            if clen <= k:
                                samp = vec
                            else:
                                idx = torch.randperm(clen, device=vec.device)[:k]
                                samp = vec[idx]
                            arr = samp.cpu().numpy()
                            self._append_reservoir(name + f"__ch{c}", arr, mode=self._mode)
                else:
                    numel = td.numel()
                    if numel <= self.sample_size:
                        sampled = td.reshape(-1)
                    else:
                        idx = torch.randperm(numel, device=td.device)[:self.sample_size]
                        sampled = td.reshape(-1)[idx]
                    arr = sampled.cpu().numpy()
                    self._append_reservoir(name, arr, mode=self._mode)
            except Exception:
                pass

            # capture quantizer metadata
            try:
                if hasattr(module, "act_quantizer"):
                    aq = getattr(module, "act_quantizer")
                    delta = getattr(aq, "delta", None)
                    zp = getattr(aq, "zero_point", None)
                    nbits = getattr(aq, "n_bits", None)
                    per_ch = getattr(aq, "channel_wise", getattr(aq, "per_channel", False))
                    self.meta.setdefault(name, {})
                    self.meta[name].update({
                        "act_delta": self._tensor_to_py(delta),
                        "act_zero_point": self._tensor_to_py(zp),
                        "n_bits": int(nbits) if nbits is not None else None,
                        "act_per_channel": bool(per_ch)
                    })
            except Exception:
                pass

            # compact weight_delta summary
            try:
                if hasattr(module, "weight_quantizer"):
                    wq = getattr(module, "weight_quantizer")
                    wdelta = getattr(wq, "delta", None)
                    mmeta = self.meta.setdefault(name, {})
                    if wdelta is not None and "weight_delta_count" not in mmeta:
                        if torch.is_tensor(wdelta):
                            try:
                                wdev = wdelta.detach().to(self._device).float().view(-1)
                                q1 = float(torch.quantile(wdev, torch.tensor(0.25, device=self._device)).detach().cpu().item())
                                q2 = float(torch.quantile(wdev, torch.tensor(0.5, device=self._device)).detach().cpu().item())
                                q3 = float(torch.quantile(wdev, torch.tensor(0.75, device=self._device)).detach().cpu().item())
                                extra_q = {}
                                for q in self.quantiles:
                                    try:
                                        extra_q[str(q)] = float(torch.quantile(wdev, torch.tensor(q, device=self._device)).detach().cpu().item())
                                    except Exception:
                                        extra_q[str(q)] = None
                                mean = float(wdev.mean().detach().cpu().item())
                                std = float(wdev.std().detach().cpu().item())
                                mn = float(wdev.min().detach().cpu().item())
                                mx = float(wdev.max().detach().cpu().item())
                                cnt = int(wdev.numel())
                                mmeta.update({
                                    "weight_delta_q1": q1, "weight_delta_q2": q2, "weight_delta_q3": q3,
                                    "weight_delta_quantiles": extra_q,
                                    "weight_delta_mean": mean, "weight_delta_std": std,
                                    "weight_delta_min": mn, "weight_delta_max": mx,
                                    "weight_delta_count": cnt
                                })
                            except Exception:
                                try:
                                    wf = wdelta.detach().cpu().numpy().ravel()
                                    q25,q50,q75 = np.quantile(wf, [0.25,0.5,0.75]).tolist()
                                    extra_q = {str(q): float(np.quantile(wf, q)) for q in self.quantiles}
                                    mmeta.update({
                                        "weight_delta_q1": float(q25), "weight_delta_q2": float(q50), "weight_delta_q3": float(q75),
                                        "weight_delta_quantiles": extra_q,
                                        "weight_delta_mean": float(np.mean(wf)), "weight_delta_std": float(np.std(wf)),
                                        "weight_delta_min": float(np.min(wf)), "weight_delta_max": float(np.max(wf)),
                                        "weight_delta_count": int(wf.size)
                                    })
                                except Exception:
                                    mmeta.update({"weight_delta_summary_error": True})
                        else:
                            try:
                                arr = np.asarray(wdelta)
                                q25,q50,q75 = np.quantile(arr, [0.25,0.5,0.75]).tolist()
                                extra_q = {str(q): float(np.quantile(arr, q)) for q in self.quantiles}
                                mmeta.update({
                                    "weight_delta_q1": float(q25), "weight_delta_q2": float(q50), "weight_delta_q3": float(q75),
                                    "weight_delta_quantiles": extra_q,
                                    "weight_delta_mean": float(np.mean(arr)), "weight_delta_std": float(np.std(arr)),
                                    "weight_delta_min": float(np.min(arr)), "weight_delta_max": float(np.max(arr)),
                                    "weight_delta_count": int(arr.size)
                                })
                            except Exception:
                                mmeta.update({"weight_delta_summary_error": True})
            except Exception:
                pass

        return _hook

    def _append_reservoir(self, key, arr_np, mode="pre"):
        buf = self.buff_pre if mode == "pre" else self.buff_post
        lst = buf.setdefault(key, [])
        for v in np.asarray(arr_np).ravel():
            if len(lst) < self.reservoir_size:
                lst.append(float(v))
            else:
                idx = np.random.randint(0, len(lst))
                lst[idx] = float(v)

    def _tensor_to_py(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            a = x.detach().cpu().numpy()
            if a.size == 1:
                return float(a.reshape(1)[0])
            else:
                return a.tolist()
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x).tolist()
        return x

    def set_mode(self, mode):
        assert mode in ("pre", "post")
        self._mode = mode

    def dump_results(self, out_dir=None, n_workers=10):
        od = out_dir or self.out_dir
        os.makedirs(od, exist_ok=True)

        # prepare meta (compact)
        meta_clean = {}
        for name, m in self.meta.items():
            mm = {}
            for k in ("act_delta", "act_zero_point", "n_bits", "act_per_channel",
                      "weight_delta_q1", "weight_delta_q2", "weight_delta_q3",
                      "weight_delta_quantiles", "weight_delta_mean", "weight_delta_std",
                      "weight_delta_min", "weight_delta_max", "weight_delta_count"):
                if k in m:
                    mm[k] = m[k]
            meta_clean[name] = mm
        meta_path = os.path.join(od, f"{self.run_tag}_layer_qmeta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_clean, f, indent=2)

        # build tasks (include quantiles in each task)
        keys = set(list(self.buff_pre.keys()) + list(self.buff_post.keys()))
        base_names = set()
        for k in keys.union(set(meta_clean.keys())):
            base = k.split("__ch")[0] if "__ch" in k else k
            base_names.add(base)
        base_names = sorted(list(base_names))
        tasks = []
        for base in base_names:
            pre_list = []
            post_list = []
            if base in self.buff_pre:
                pre_list.append(np.asarray(self.buff_pre.get(base, []), dtype=np.float64))
            if base in self.buff_post:
                post_list.append(np.asarray(self.buff_post.get(base, []), dtype=np.float64))
            # channel keys
            ch_keys_pre = [k for k in self.buff_pre.keys() if k.startswith(base + "__ch")]
            ch_keys_post = [k for k in self.buff_post.keys() if k.startswith(base + "__ch")]
            for k in ch_keys_pre:
                pre_list.append(np.asarray(self.buff_pre.get(k, []), dtype=np.float64))
            for k in ch_keys_post:
                post_list.append(np.asarray(self.buff_post.get(k, []), dtype=np.float64))
            pre_arr = np.concatenate(pre_list) if len(pre_list) > 0 else np.asarray([], dtype=np.float64)
            post_arr = np.concatenate(post_list) if len(post_list) > 0 else np.asarray([], dtype=np.float64)
            meta_for = meta_clean.get(base, {})
            tasks.append((base, pre_arr, post_arr, meta_for, self.quantiles))

        # run module-level worker _compute_layer_stats_cpu
        n_workers = int(n_workers) if n_workers is not None else max(1, (mp.cpu_count() - 1))
        if n_workers > 1 and len(tasks) > 0:
            with mp.Pool(processes=min(n_workers, len(tasks))) as pool:
                results = pool.map(_compute_layer_stats_cpu, tasks)
        else:
            results = list(map(_compute_layer_stats_cpu, tasks))

        stats_all = dict(results)

        # rank top-k by pct_clipped / pre variance
        def _rank_key(item):
            name, obj = item
            err = obj.get("errors", {})
            pct = err.get("pct_clipped", None)
            if pct is not None:
                return (1, float(pct))
            pre = obj.get("pre", {})
            std = pre.get("std", 0.0) or 0.0
            return (0, float(std**2))
        ranked = sorted(stats_all.items(), key=_rank_key, reverse=True)
        topk_layers = [name for name, _ in ranked[:self.topk]]

        # compute per-channel stats for top-K
        per_channel_results = {}
        for base in topk_layers:
            ch_keys = sorted([k for k in set(list(self.buff_pre.keys())+list(self.buff_post.keys())) if k.startswith(base + "__ch")])
            if not ch_keys:
                continue
            per_channel_results[base] = {}
            for ch_key in ch_keys:
                pre = np.asarray(self.buff_pre.get(ch_key, []), dtype=np.float64)
                post = np.asarray(self.buff_post.get(ch_key, []), dtype=np.float64)
                # re-use worker function logic locally (serial small compute)
                def _stat_small(arr):
                    if arr.size == 0:
                        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "quantiles": {}}
                    qm = {}
                    for q in self.quantiles:
                        try:
                            qm[str(q)] = float(np.quantile(arr, q))
                        except Exception:
                            qm[str(q)] = None
                    return {"count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()),
                            "mean": float(np.mean(arr)), "std": float(np.std(arr)), "quantiles": qm}
                pre_stat = _stat_small(pre)
                post_stat = _stat_small(post)
                errors = {"mse": None, "max_abs": None, "median_abs": None, "pct_clipped": None, "range_coverage": None}
                if pre.size > 0 and post.size > 0:
                    n = min(pre.size, post.size)
                    rng = np.random.default_rng(abs(hash(ch_key)) % (2**32))
                    idx_pre = rng.choice(pre.size, size=n, replace=False)
                    idx_post = rng.choice(post.size, size=n, replace=False)
                    a_pre = pre[idx_pre]; a_post = post[idx_post]; dif = a_post - a_pre
                    errors["mse"] = float(np.mean(dif**2))
                    errors["max_abs"] = float(np.max(np.abs(dif)))
                    errors["median_abs"] = float(np.median(np.abs(dif)))
                # pct_clipped using base meta
                meta_base = meta_clean.get(base, {})
                try:
                    act_delta = meta_base.get("act_delta", None)
                    act_zp = meta_base.get("act_zero_point", None)
                    nb = meta_base.get("n_bits", None)
                    if nb is not None and act_delta is not None:
                        if isinstance(act_delta, (list, tuple, np.ndarray)):
                            ch_idx = None
                            try:
                                ch_idx = int(ch_key.split("__ch")[-1])
                            except Exception:
                                ch_idx = None
                            delta_c = float(np.mean(act_delta)) if ch_idx is None else float(act_delta[ch_idx]) if ch_idx < len(act_delta) else float(np.mean(act_delta))
                        else:
                            delta_c = float(act_delta)
                        if isinstance(act_zp, (list, tuple, np.ndarray)):
                            zp_c = int(np.mean(act_zp))
                        else:
                            zp_c = int(act_zp) if act_zp is not None else 0
                        nb = int(nb)
                        if zp_c is not None and 0 <= zp_c <= (2**nb - 1):
                            qmin = 0; qmax = 2**nb - 1
                        else:
                            qmin = -(2**(nb-1)); qmax = (2**(nb-1))-1
                        rep_min_f = (qmin - zp_c) * delta_c
                        rep_max_f = (qmax - zp_c) * delta_c
                        if post.size > 0:
                            errors["pct_clipped"] = float(((post < rep_min_f) | (post > rep_max_f)).sum()) / float(post.size)
                        if pre.size > 0:
                            try:
                                obs_q = float(np.quantile(pre, 0.999))
                                if obs_q != 0:
                                    errors["range_coverage"] = float(abs(rep_max_f) / abs(obs_q))
                            except Exception:
                                errors["range_coverage"] = None
                except Exception:
                    pass
                per_channel_results[base][ch_key.split("__ch")[-1]] = {"pre": pre_stat, "post": post_stat, "errors": errors}

        # combine
        final_stats = dict(stats_all)
        final_stats["_per_channel_topk"] = per_channel_results

        stats_path = os.path.join(od, f"{self.run_tag}_layer_act_stats_prepost.json")
        # clean to JSON-serializable
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (np.floating, float)):
                if math.isnan(o): return None
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, list):
                return [_clean(x) for x in o]
            return o
        with open(stats_path, "w") as f:
            json.dump(_clean(final_stats), f, indent=2, allow_nan=True)

        return meta_path, stats_path

    def close(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def load_calibration_data():
    """載入校準資料"""
    print("="*100)
    print("2. Loading calibration data...")
    calib_file = 'QATcode/calibration_diffae.pth'
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration data not found: {calib_file}")
        
    dataset = DiffusionInputDataset(calib_file)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    print(f"✅ Calibration dataset loaded: {len(dataset)} samples")
    
    return dataset, data_loader

def setup_quantization_params():
    """設置量化參數"""
    print("="*100)
    print("3. Setting up quantization parameters...")
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    print(f"Weight quantization: {wq_params}")
    print(f"Activation quantization: {aq_params}")
    
    return wq_params, aq_params

def create_quantized_model(model, wq_params, aq_params, quantize_skip_connections=False):
    """創建選擇性量化模型"""
    print("="*100)
    print("4. Creating selective quantized model...")
    print(f"Skip connection 量化設定: {'啟用' if quantize_skip_connections else '停用'}")
    
    # 分析 UNet 結構
    analyze_unet_structure(model)
    
    # 創建量化模型
    qnn = SelectiveQuantModel(
        model=model, 
        weight_quant_params=wq_params, 
        act_quant_params=aq_params, 
        need_init=True,
        quantize_skip_connections=quantize_skip_connections
    )
    qnn.cuda()
    qnn.eval()
    
    # 統計量化模組
    total_quant, total_original = count_quantized_modules(qnn.model)
    if total_quant == 0:
        raise RuntimeError("No quantizable modules found!")
    
    print(f"✅ Successfully quantized {total_quant} modules")
    return qnn, total_quant

def setup_first_last_layers(qnn):
    """設置首尾層為 8-bit"""
    print("="*100)
    print("5. Setting first and last layers to 8-bit...")
    
    quant_modules = []
    target_modules = ['time_embed', 'input_blocks', 'middle_block', 'output_blocks', 'out']
    
    for module_name in target_modules:
        if hasattr(qnn.model, module_name):
            module = getattr(qnn.model, module_name)
            for name, child in module.named_modules():
                if isinstance(child, QuantModule):
                    quant_modules.append((f"{module_name}.{name}", child))
    
    # 設置前 3 層為 8-bit
    for i in range(min(3, len(quant_modules))):
        name, module = quant_modules[i]
        module.weight_quantizer.bitwidth_refactor(8)
        module.act_quantizer.bitwidth_refactor(8)
        module.ignore_reconstruction = True
        print(f"  Set layer {i} ({name}) to 8-bit")
    
    # 設置最後一層為 8-bit
    if len(quant_modules) > 3:
        name, module = quant_modules[-1]
        module.weight_quantizer.bitwidth_refactor(8)
        module.act_quantizer.bitwidth_refactor(8)
        module.ignore_reconstruction = True
        print(f"  Set last layer ({name}) to 8-bit")

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    #print(f"✅ Loaded {len(y_data)} samples")
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

def calibrate_quantized_model(qnn, model : LitModel, cali_images, cali_t, cali_y):
    """校準量化模型"""
    print("="*100)
    print("7. Calibrating quantized model...")
    
    # 設置量化狀態
    qnn.set_quant_state(True, True)

    start = time.time()
    # 準備資料和採樣器
    device = next(qnn.parameters()).device
    collector = ActivationCollector(model=qnn, out_dir="./analysis_step4", run_tag="step4_diffae")  # change run_tag to "step4_efficientdm" when running EfficientDM script
    collector.set_mode("pre")
    qnn.set_quant_state(True, False)   # disable act fake-quant but keep weight quant (if that API toggles that way)
    qnn.eval()
    
    with torch.no_grad():
        print('First run to init quantization parameters (latent cond if available)...')
        _ = qnn(x=cali_images[:32].to(device),t=cali_t[:32].to(device),cond=cali_y[:32].to(device))
        calibration_success = True
    print('Finished PRE pass')
    collector.set_mode("post")
    qnn.set_quant_state(True, True)
    qnn.eval()
    with torch.no_grad():
        _ = qnn(x=cali_images[:32].to(device),t=cali_t[:32].to(device),cond=cali_y[:32].to(device))
    print('Finished POST pass')
    meta_path, stats_path = collector.dump_results(out_dir="QATcode/analysis_step4_asym_signed")
    collector.close()
    print("[STEP4 HOOK] Wrote:", meta_path, stats_path)
    print('Finished dumping results')
    print(f"Time taken: {time.time() - start} seconds")
    
    return calibration_success



def save_quantized_model(qnn, total_quant):
    """保存量化模型和配置"""
    print("="*100)
    print("9. Saving quantized model...")
    
    output_dir = "QATcode"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model_path = f'{output_dir}/diffae_unet_quantw{n_bits_w}a{n_bits_a}_selective.pth'
    torch.save(qnn.state_dict(), model_path)
    print(f"✅ Quantized model saved: {model_path}")
    
    # 保存配置
    config_path = f'{output_dir}/diffae_unet_quant_config.pth'
    config = {
        'n_bits_w': n_bits_w,
        'n_bits_a': n_bits_a,
        'quantized_modules': total_quant,
        'quantization_scope': 'UNet only (input_blocks, middle_block, output_blocks, out)',
        'excluded_modules': 'time_embed, encoder, latent_net',
        'quantize_skip_connections': quantize_skip_connections
    }
    torch.save(config, config_path)
    print(f"✅ Quantization config saved: {config_path}")
    
    return model_path, config_path

def main():
    """主函數 - Diff-AE Step 4: 選擇性量化 UNet 架構"""
    print("=== Diff-AE Step 4: 選擇性量化 UNet 架構 ===")
    
    try:
        # 1. 載入模型
        conf = ffhq128_autoenc_latent()
        model = LitModel(conf)
        model.load_state_dict(torch.load(f'{conf.logdir}/last.ckpt', map_location='cpu', weights_only=False)['state_dict'])
        model = model.ema_model
        assert model is not None, "ema_model is None"
        model.cuda()
        model.eval()
        print("✅ Model loaded successfully!")
        # 2. 載入校準資料
        dataset, data_loader = load_calibration_data()
        
        # 3. 設置量化參數
        wq_params, aq_params = setup_quantization_params()
        
        # 4. 創建量化模型
        qnn, total_quant = create_quantized_model(model, wq_params, aq_params, quantize_skip_connections=False)
        #print(qnn)
        
        # 5. 設置首尾層
        setup_first_last_layers(qnn)
        
        # 6. 擷取校準樣本
        cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
        
        # 7. 校準量化模型
        calibration_success = calibrate_quantized_model(qnn, model, cali_images, cali_t, cali_y)
        
        if not calibration_success:
            print("❌ Step 4 failed due to calibration error!")
            return
        
        
        
        # 8. 保存量化模型
        model_path, config_path = save_quantized_model(qnn, total_quant)
        
        # 10. 總結
        print(f"\n🎉 Step 4 completed successfully!")
        print(f"📊 Summary:")
        print(f"   - 量化範圍: UNet 核心架構")
        print(f"   - 量化模組數: {total_quant}")
        print(f"   - 權重位數: {n_bits_w} bits")
        print(f"   - 激活位數: {n_bits_a} bits")
        print(f"   - Skip connection 量化: {'啟用' if quantize_skip_connections else '停用'}")
        print(f"   - 模型路徑: {model_path}")
        print(f"   - 配置路徑: {config_path}")
            
    except Exception as e:
        print(f"❌ Error in Step 4: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 