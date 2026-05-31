#!/usr/bin/env python3
"""Q-DiffAE data-movement and CIM array analysis.

Run with:
  /home/jimmy/anaconda3/envs/diffae_bw/bin/python data_movement_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_KEY = "qdiffae"
REPO_ROOT = Path("/home/jimmy/diffae")
QAT_ROOT = REPO_ROOT / "QATcode"
DEFAULT_OUTPUT_DIR = QAT_ROOT / "analysis" / "output" / "data_movement"

BASE_CKPT = REPO_ROOT / "checkpoints/ffhq128_autoenc_latent/last.ckpt"
QUANT_CKPT = QAT_ROOT / "quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
SCHEDULER_PATH = (
    QAT_ROOT
    / "cache_method/Stage2/stage2_output/fullExperimentsK25sw2/"
    / "baseline_908030/stage2_refined_scheduler_config.json"
)

T_EXPECTED = 100
BYTES_PER_ELEMENT = 1
WEIGHT_BYTES_PER_ELEMENT = 1
ACTIVATION_BYTES_PER_ELEMENT = 4
CACHE_OUTPUT_BYTES_PER_ELEMENT = 4
INPUT_RESOLUTION = 128
BATCH_SIZE = 1
CIM_WEIGHT_BITS = 8
CIM_ARRAY_256_CELLS = 256 * 256
CIM_ARRAY_32_CELLS = 32 * 32


@dataclass(frozen=True)
class Shape:
    c: int
    h: int
    w: int


@dataclass
class LayerRecord:
    model: str
    block_id: int
    block_name: str
    layer_name: str
    layer_type: str
    weight_shape: str
    weight_bytes: int
    act_bytes: int
    dm_per_exec: int
    cim_rows: int
    cim_cols: int
    cim_max_dim: int
    cim_array_size: int
    num_params: int
    cim_weight_bits: int
    cim_cell_count_w8: int
    cim_arrays_256x256: int
    cim_arrays_32x32: int
    is_quantized: bool
    seq_len: int


@dataclass
class BlockRecord:
    model: str
    block_id: int
    block_name: str
    canonical_name: str
    spatial_h: int
    spatial_w: int
    input_channels: int
    output_channels: int
    weight_bytes_per_exec: int
    act_bytes_per_exec: int
    output_write_bytes: int
    total_dm_bytes: int
    dm_per_exec: int
    exec_count_baseline: int
    exec_count_cached: int
    dm_baseline: int
    dm_cached: int
    cim_block_max_dim: int
    cim_block_max_array_size: int
    s3_cache_output_bytes: int


def format_bytes(n: int | float) -> str:
    n = int(n)
    if n >= 1 << 30:
        return f"{n / (1 << 30):.4f} GB ({n:,} B)"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.4f} MB ({n:,} B)"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.4f} KB ({n:,} B)"
    return f"{n:,} B"


def byte_summary_fields(prefix: str, n: int | float) -> dict[str, Any]:
    n = float(n)
    return {
        f"{prefix}_readable": format_bytes(n),
        f"{prefix}_KB": n / (1 << 10),
        f"{prefix}_MB": n / (1 << 20),
        f"{prefix}_GB": n / (1 << 30),
    }


def count_summary_fields(prefix: str, n: int | float, unit: str) -> dict[str, Any]:
    n = float(n)
    return {
        f"{prefix}_readable": f"{n:,.0f} {unit}",
        f"{prefix}_K": n / 1_000,
        f"{prefix}_M": n / 1_000_000,
    }


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def cim_storage_fields(num_params: int) -> dict[str, int]:
    cells = num_params * CIM_WEIGHT_BITS
    return {
        "num_params": num_params,
        "cim_weight_bits": CIM_WEIGHT_BITS,
        "cim_cell_count_w8": cells,
        "cim_arrays_256x256": ceil_div(cells, CIM_ARRAY_256_CELLS),
        "cim_arrays_32x32": ceil_div(cells, CIM_ARRAY_32_CELLS),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    rows = [[str(x) for x in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def load_scheduler(path: Path) -> tuple[int, list[dict[str, Any]]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    t = int(cfg["T"])
    if t != T_EXPECTED:
        raise ValueError(f"Scheduler T={t}, expected {T_EXPECTED}")
    blocks = sorted(cfg["blocks"], key=lambda b: int(b["canonical_runtime_block_id"]))
    seen = [int(b["canonical_runtime_block_id"]) for b in blocks]
    if seen != list(range(len(blocks))):
        raise ValueError(f"canonical_runtime_block_id must be contiguous, got {seen}")
    for b in blocks:
        mask = b["expanded_mask"]
        if len(mask) != t:
            raise ValueError(f"{b.get('runtime_name')}: mask length {len(mask)} != {t}")
        if str(b["runtime_name"]) != runtime_name_from_id(int(b["canonical_runtime_block_id"])):
            raise ValueError(f"runtime name/id mismatch in scheduler block: {b}")
    return t, blocks


def runtime_name_from_id(block_id: int) -> str:
    if block_id < 15:
        return f"encoder_layer_{block_id}"
    if block_id == 15:
        return "middle_layer"
    return f"decoder_layer_{block_id - 16}"


def module_for_runtime(model: Any, runtime_name: str) -> Any:
    if runtime_name.startswith("encoder_layer_"):
        return model.input_blocks[int(runtime_name.rsplit("_", 1)[1])]
    if runtime_name == "middle_layer":
        return model.middle_block
    if runtime_name.startswith("decoder_layer_"):
        return model.output_blocks[int(runtime_name.rsplit("_", 1)[1])]
    raise ValueError(f"Unknown runtime name: {runtime_name}")


def _weight(module: Any) -> Any | None:
    w = getattr(module, "org_weight", None)
    if w is not None:
        return w
    w = getattr(module, "weight", None)
    return w if w is not None else None


def _is_quantized(module: Any) -> bool:
    return hasattr(module, "org_weight")


def _conv2d_record(
    *,
    block_id: int,
    block_name: str,
    layer_name: str,
    module: Any,
    shape: Shape,
) -> tuple[LayerRecord, Shape]:
    w = _weight(module)
    if w is None or int(w.ndim) != 4:
        raise ValueError(f"{layer_name}: expected 4D weight")
    c_out, c_in, kh, kw = [int(x) for x in w.shape]
    weight_bytes = c_out * c_in * kh * kw * WEIGHT_BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * c_in * shape.h * shape.w * ACTIVATION_BYTES_PER_ELEMENT
    rows = c_in * kh * kw
    cols = c_out
    num_params = rows * cols
    rec = LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="conv2d",
        weight_shape=str((c_out, c_in, kh, kw)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=rows,
        cim_cols=cols,
        cim_max_dim=max(rows, cols),
        cim_array_size=num_params,
        **cim_storage_fields(num_params),
        is_quantized=_is_quantized(module),
        seq_len=1,
    )
    return rec, Shape(c_out, shape.h, shape.w)


def _conv1d_record(
    *,
    block_id: int,
    block_name: str,
    layer_name: str,
    module: Any,
    shape: Shape,
) -> LayerRecord:
    w = _weight(module)
    if w is None or int(w.ndim) != 3:
        raise ValueError(f"{layer_name}: expected 3D weight")
    c_out, c_in, k = [int(x) for x in w.shape]
    length = shape.h * shape.w
    weight_bytes = c_out * c_in * k * WEIGHT_BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * c_in * length * ACTIVATION_BYTES_PER_ELEMENT
    rows = c_in * k
    cols = c_out
    num_params = rows * cols
    return LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="conv1d",
        weight_shape=str((c_out, c_in, k)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=rows,
        cim_cols=cols,
        cim_max_dim=max(rows, cols),
        cim_array_size=num_params,
        **cim_storage_fields(num_params),
        is_quantized=_is_quantized(module),
        seq_len=1,
    )


def _linear_record(
    *,
    block_id: int,
    block_name: str,
    layer_name: str,
    module: Any,
    seq_len: int = 1,
) -> LayerRecord:
    w = _weight(module)
    if w is None or int(w.ndim) != 2:
        raise ValueError(f"{layer_name}: expected 2D weight")
    out_f, in_f = [int(x) for x in w.shape]
    num_params = out_f * in_f
    weight_bytes = out_f * in_f * WEIGHT_BYTES_PER_ELEMENT
    act_bytes = BATCH_SIZE * seq_len * in_f * ACTIVATION_BYTES_PER_ELEMENT
    return LayerRecord(
        model=MODEL_KEY,
        block_id=block_id,
        block_name=block_name,
        layer_name=layer_name,
        layer_type="linear",
        weight_shape=str((out_f, in_f)),
        weight_bytes=weight_bytes,
        act_bytes=act_bytes,
        dm_per_exec=weight_bytes + act_bytes,
        cim_rows=in_f,
        cim_cols=out_f,
        cim_max_dim=max(in_f, out_f),
        cim_array_size=num_params,
        **cim_storage_fields(num_params),
        is_quantized=_is_quantized(module),
        seq_len=seq_len,
    )


def _module_out_channels(module: Any, fallback: int) -> int:
    w = _weight(module)
    if w is not None and int(w.ndim) in (3, 4):
        return int(w.shape[0])
    return int(getattr(module, "out_channels", fallback))


def _enumerate_resblock(
    rb: Any,
    *,
    block_id: int,
    block_name: str,
    prefix: str,
    shape: Shape,
) -> tuple[list[LayerRecord], Shape]:
    records: list[LayerRecord] = []
    conf = getattr(rb, "conf", rb)
    up = bool(getattr(conf, "up", False))
    down = bool(getattr(conf, "down", False))
    out_channels = int(getattr(conf, "out_channels"))

    conv_shape = shape
    if up:
        conv_shape = Shape(shape.c, shape.h * 2, shape.w * 2)
    elif down:
        conv_shape = Shape(shape.c, shape.h // 2, shape.w // 2)

    in_conv = rb.in_layers[-1]
    rec, _ = _conv2d_record(
        block_id=block_id,
        block_name=block_name,
        layer_name=f"{prefix}.in_layers.conv",
        module=in_conv,
        shape=conv_shape,
    )
    records.append(rec)

    if hasattr(rb, "emb_layers"):
        records.append(
            _linear_record(
                block_id=block_id,
                block_name=block_name,
                layer_name=f"{prefix}.emb_layers.linear",
                module=rb.emb_layers[-1],
            )
        )
    if hasattr(rb, "cond_emb_layers"):
        records.append(
            _linear_record(
                block_id=block_id,
                block_name=block_name,
                layer_name=f"{prefix}.cond_emb_layers.linear",
                module=rb.cond_emb_layers[-1],
            )
        )

    out_conv = rb.out_layers[-1]
    rec, _ = _conv2d_record(
        block_id=block_id,
        block_name=block_name,
        layer_name=f"{prefix}.out_layers.conv",
        module=out_conv,
        shape=Shape(out_channels, conv_shape.h, conv_shape.w),
    )
    records.append(rec)

    skip = rb.skip_connection
    if _weight(skip) is not None:
        rec, _ = _conv2d_record(
            block_id=block_id,
            block_name=block_name,
            layer_name=f"{prefix}.skip_connection",
            module=skip,
            shape=conv_shape,
        )
        records.append(rec)

    return records, Shape(out_channels, conv_shape.h, conv_shape.w)


def _enumerate_attention(
    attn: Any,
    *,
    block_id: int,
    block_name: str,
    prefix: str,
    shape: Shape,
) -> list[LayerRecord]:
    records: list[LayerRecord] = []
    for name, module in [("qkv", attn.qkv), ("proj_out", attn.proj_out)]:
        records.append(
            _conv1d_record(
                block_id=block_id,
                block_name=block_name,
                layer_name=f"{prefix}.{name}",
                module=module,
                shape=shape,
            )
        )
    return records


def enumerate_block(
    block: Any,
    *,
    block_id: int,
    block_name: str,
    in_shape: Shape,
    classes: dict[str, Any],
) -> tuple[list[LayerRecord], Shape]:
    records: list[LayerRecord] = []
    shape = in_shape
    for idx, layer in enumerate(block):
        prefix = f"{block_name}[{idx}]"
        if isinstance(layer, classes["ResBlock"]):
            recs, shape = _enumerate_resblock(
                layer, block_id=block_id, block_name=block_name, prefix=prefix, shape=shape
            )
            records.extend(recs)
        elif isinstance(layer, classes["AttentionBlock"]):
            records.extend(
                _enumerate_attention(
                    layer, block_id=block_id, block_name=block_name, prefix=prefix, shape=shape
                )
            )
        elif isinstance(layer, classes["Downsample"]):
            target = getattr(layer, "op", layer)
            if _weight(target) is not None:
                rec, _ = _conv2d_record(
                    block_id=block_id,
                    block_name=block_name,
                    layer_name=f"{prefix}.downsample",
                    module=target,
                    shape=shape,
                )
                records.append(rec)
            shape = Shape(_module_out_channels(target, shape.c), shape.h // 2, shape.w // 2)
        elif isinstance(layer, classes["Upsample"]):
            target = getattr(layer, "conv", layer)
            up_shape = Shape(shape.c, shape.h * 2, shape.w * 2)
            if _weight(target) is not None:
                rec, _ = _conv2d_record(
                    block_id=block_id,
                    block_name=block_name,
                    layer_name=f"{prefix}.upsample",
                    module=target,
                    shape=up_shape,
                )
                records.append(rec)
            shape = Shape(_module_out_channels(target, shape.c), up_shape.h, up_shape.w)
        elif _weight(layer) is not None:
            w = _weight(layer)
            if int(w.ndim) == 4:
                rec, shape = _conv2d_record(
                    block_id=block_id,
                    block_name=block_name,
                    layer_name=f"{prefix}.conv",
                    module=layer,
                    shape=shape,
                )
                records.append(rec)
            elif int(w.ndim) == 2:
                records.append(
                    _linear_record(
                        block_id=block_id,
                        block_name=block_name,
                        layer_name=f"{prefix}.linear",
                        module=layer,
                    )
                )
    return records, shape


def load_quantized_model() -> Any:
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    import torch
    from QATcode.quantize_ver2.common_utils import load_diffae_model
    from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModel_DiffAE_LoRA

    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]
        torch.nn.Module.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]

    base = load_diffae_model(str(BASE_CKPT), logging.getLogger("qdiffae_dm"))
    wq = {"n_bits": 8, "channel_wise": True, "scale_method": "absmax"}
    aq = {"n_bits": 8, "channel_wise": False, "scale_method": "absmax", "leaf_param": True}
    qm = QuantModel_DiffAE_LoRA(
        model=base.model,
        weight_quant_params=wq,
        act_quant_params=aq,
        num_steps=T_EXPECTED,
        lora_rank=32,
        mode="train",
    )
    ckpt = torch.load(str(QUANT_CKPT), map_location="cpu", weights_only=False)
    qm.load_state_dict(ckpt["ema_model_state_dict"], strict=False)
    qm.eval()
    return qm.model


def analyze() -> tuple[list[BlockRecord], list[LayerRecord], dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT))
    from model.blocks import AttentionBlock, Downsample, ResBlock, Upsample

    t, sched_blocks = load_scheduler(SCHEDULER_PATH)
    model = load_quantized_model()
    classes = {
        "ResBlock": ResBlock,
        "AttentionBlock": AttentionBlock,
        "Downsample": Downsample,
        "Upsample": Upsample,
    }

    shapes_in: dict[str, Shape] = {}
    shapes_out: dict[str, Shape] = {}
    layer_records: list[LayerRecord] = []
    block_records: list[BlockRecord] = []

    current = Shape(3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    encoder_outputs: list[Shape] = []

    for b in sched_blocks:
        block_id = int(b["canonical_runtime_block_id"])
        runtime = str(b["runtime_name"])
        block = module_for_runtime(model, runtime)
        if runtime.startswith("decoder_layer_"):
            skip = encoder_outputs.pop()
            current = Shape(current.c + skip.c, current.h, current.w)

        shapes_in[runtime] = current
        recs, out_shape = enumerate_block(
            block, block_id=block_id, block_name=runtime, in_shape=current, classes=classes
        )
        if not recs:
            raise ValueError(f"No compute layers found for {runtime}")
        layer_records.extend(recs)
        shapes_out[runtime] = out_shape

        if runtime.startswith("encoder_layer_"):
            encoder_outputs.append(out_shape)
        current = out_shape

        weight_sum = sum(r.weight_bytes for r in recs)
        act_sum = sum(r.act_bytes for r in recs)
        exec_cached = sum(bool(x) for x in b["expanded_mask"])
        output_write_bytes = BATCH_SIZE * out_shape.c * out_shape.h * out_shape.w * ACTIVATION_BYTES_PER_ELEMENT
        dm_exec = weight_sum + act_sum
        total_dm = dm_exec + output_write_bytes
        block_records.append(
            BlockRecord(
                model=MODEL_KEY,
                block_id=block_id,
                block_name=runtime,
                canonical_name=str(b["name"]),
                spatial_h=shapes_in[runtime].h,
                spatial_w=shapes_in[runtime].w,
                input_channels=shapes_in[runtime].c,
                output_channels=out_shape.c,
                weight_bytes_per_exec=weight_sum,
                act_bytes_per_exec=act_sum,
                output_write_bytes=output_write_bytes,
                total_dm_bytes=total_dm,
                dm_per_exec=dm_exec,
                exec_count_baseline=t,
                exec_count_cached=exec_cached,
                dm_baseline=total_dm * t,
                dm_cached=total_dm * exec_cached,
                cim_block_max_dim=max(r.cim_max_dim for r in recs),
                cim_block_max_array_size=max(r.cim_array_size for r in recs),
                s3_cache_output_bytes=(
                    BATCH_SIZE
                    * out_shape.c
                    * out_shape.h
                    * out_shape.w
                    * CACHE_OUTPUT_BYTES_PER_ELEMENT
                ),
            )
        )

    if encoder_outputs:
        raise ValueError(f"Encoder skip stack not fully consumed: {len(encoder_outputs)}")
    validate_records(t, block_records, layer_records)
    summary = make_summary(t, block_records, layer_records)
    return block_records, layer_records, summary


def validate_records(t: int, blocks: list[BlockRecord], layers: list[LayerRecord]) -> None:
    if len(blocks) != 31:
        raise ValueError(f"Expected 31 blocks, got {len(blocks)}")
    expected = [runtime_name_from_id(i) for i in range(31)]
    got = [b.block_name for b in blocks]
    if got != expected:
        raise ValueError(f"Runtime order mismatch: {got}")
    layer_blocks = {r.block_name for r in layers}
    if layer_blocks != set(expected):
        raise ValueError("Layer/block coverage mismatch")
    for r in layers:
        if not r.layer_name.startswith(r.block_name):
            raise ValueError(f"Layer name not under block: {r.block_name} vs {r.layer_name}")
        if r.weight_bytes <= 0 or r.act_bytes <= 0 or r.dm_per_exec != r.weight_bytes + r.act_bytes:
            raise ValueError(f"Invalid bytes in {r}")
    for b in blocks:
        child = [r for r in layers if r.block_id == b.block_id]
        if b.weight_bytes_per_exec != sum(r.weight_bytes for r in child):
            raise ValueError(f"Weight sum mismatch for {b.block_name}")
        if b.act_bytes_per_exec != sum(r.act_bytes for r in child):
            raise ValueError(f"Activation sum mismatch for {b.block_name}")
        if b.output_write_bytes <= 0:
            raise ValueError(f"Invalid output write bytes for {b.block_name}")
        if b.dm_per_exec != b.weight_bytes_per_exec + b.act_bytes_per_exec:
            raise ValueError(f"Read-only DM mismatch for {b.block_name}")
        if b.total_dm_bytes != b.weight_bytes_per_exec + b.act_bytes_per_exec + b.output_write_bytes:
            raise ValueError(f"Total DM mismatch for {b.block_name}")
        if b.dm_baseline != b.total_dm_bytes * t:
            raise ValueError(f"Baseline mismatch for {b.block_name}")
        if b.dm_cached != b.total_dm_bytes * b.exec_count_cached:
            raise ValueError(f"Cached mismatch for {b.block_name}")


def make_summary(t: int, blocks: list[BlockRecord], layers: list[LayerRecord]) -> dict[str, Any]:
    baseline = sum(b.dm_baseline for b in blocks)
    cached = sum(b.dm_cached for b in blocks)
    baseline_per_step = baseline / t
    cached_per_step = cached / t
    summary = {
        "model": MODEL_KEY,
        "T": t,
        "num_blocks": len(blocks),
        "num_layers": len(layers),
        "num_quantized_layers": sum(1 for r in layers if r.is_quantized),
        "bytes_per_element": BYTES_PER_ELEMENT,
        "weight_bytes_per_element": WEIGHT_BYTES_PER_ELEMENT,
        "activation_bytes_per_element": ACTIVATION_BYTES_PER_ELEMENT,
        "cache_output_bytes_per_element": CACHE_OUTPUT_BYTES_PER_ELEMENT,
        "baseline_bytes": baseline,
        "cached_bytes": cached,
        "reduction_ratio": (baseline - cached) / baseline,
        "baseline_bytes_per_step": baseline_per_step,
        "cached_bytes_per_step": cached_per_step,
        "global_cim_max_dim": max(r.cim_max_dim for r in layers),
        "global_cim_max_array_size": max(r.cim_array_size for r in layers),
        "s3_cache_storage_bytes": sum(b.s3_cache_output_bytes for b in blocks),
    }
    summary.update(byte_summary_fields("baseline", baseline))
    summary.update(byte_summary_fields("cached", cached))
    summary.update(byte_summary_fields("baseline_per_step", baseline_per_step))
    summary.update(byte_summary_fields("cached_per_step", cached_per_step))
    summary.update(
        count_summary_fields(
            "global_cim_max_array_size",
            summary["global_cim_max_array_size"],
            "cells",
        )
    )
    summary.update(byte_summary_fields("s3_cache_storage", summary["s3_cache_storage_bytes"]))
    return summary


def save_outputs(output_dir: Path, blocks: list[BlockRecord], layers: list[LayerRecord], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "exp1_block_detail.csv", [asdict(b) for b in blocks])
    write_csv(output_dir / "exp1_layer_detail.csv", [asdict(r) for r in layers])
    write_csv(output_dir / "exp2_cim_layer_detail.csv", [asdict(r) for r in layers])
    block_cim_rows = [
        {
            "model": b.model,
            "block_id": b.block_id,
            "block_name": b.block_name,
            "num_layers": sum(1 for r in layers if r.block_id == b.block_id),
            "block_max_dim": b.cim_block_max_dim,
            "block_max_array_size": b.cim_block_max_array_size,
            "spatial_h": b.spatial_h,
            "spatial_w": b.spatial_w,
            "input_channels": b.input_channels,
            "output_channels": b.output_channels,
            "s3_cache_output_bytes": b.s3_cache_output_bytes,
        }
        for b in blocks
    ]
    write_csv(output_dir / "exp2_cim_block_summary.csv", block_cim_rows)
    write_csv(output_dir / "exp1_summary.csv", [summary])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_report(blocks: list[BlockRecord], layers: list[LayerRecord], summary: dict[str, Any]) -> None:
    print(f"\nModel: {MODEL_KEY}")
    print(f"Baseline: {format_bytes(summary['baseline_bytes'])}")
    print(f"Cached:   {format_bytes(summary['cached_bytes'])}")
    print(f"Reduction: {summary['reduction_ratio']:.2%}")
    print(f"CIM max dim: {summary['global_cim_max_dim']}")
    print(f"CIM max array size: {summary['global_cim_max_array_size']:,}")
    print_table(
        "Block-Level Data Movement",
        ["id", "block", "shape", "weight", "act", "output_write", "total/exec", "exec", "cached DM", "CIM"],
        [
            [
                b.block_id,
                b.block_name,
                f"{b.input_channels}x{b.spatial_h}x{b.spatial_w}",
                format_bytes(b.weight_bytes_per_exec),
                format_bytes(b.act_bytes_per_exec),
                format_bytes(b.output_write_bytes),
                format_bytes(b.total_dm_bytes),
                b.exec_count_cached,
                format_bytes(b.dm_cached),
                b.cim_block_max_dim,
            ]
            for b in blocks
        ],
    )
    print_table(
        "Layer-Level Data Movement",
        ["block", "layer", "type", "weight", "act", "total", "CIM(r,c)", "Q"],
        [
            [
                r.block_name,
                r.layer_name,
                r.layer_type,
                format_bytes(r.weight_bytes),
                format_bytes(r.act_bytes),
                format_bytes(r.dm_per_exec),
                f"({r.cim_rows},{r.cim_cols})",
                int(r.is_quantized),
            ]
            for r in layers
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    blocks, layers, summary = analyze()
    save_outputs(Path(args.output_dir), blocks, layers, summary)
    print_report(blocks, layers, summary)
    print(f"\n[Done] Results written to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
