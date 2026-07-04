#!/usr/bin/env python3
"""Resumable Q-DiffAE S3-Cache resweep orchestrator."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/home/jimmy/anaconda3/envs/diffae_bw/bin/python"
DEFAULT_OUTPUT_DIR = CACHE_ROOT / "resweep_output"

STAGE0_DIR = REPO_ROOT / "QATcode/cache_method/Stage0/stage0e_output"
STAGE1_SCHEDULER = REPO_ROOT / "QATcode/cache_method/Stage1/stage1_scheduler.py"
STAGE2_REFINE_SCRIPT = REPO_ROOT / "QATcode/cache_method/Stage2/stage2_runtime_refine.py"
STAGE2_THRESH_SCRIPT = REPO_ROOT / "QATcode/cache_method/Stage2/build_blockwise_thresholds.py"
FID_SCRIPT = REPO_ROOT / "QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py"
QAT_CKPT = REPO_ROOT / "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
BASE_CKPT = REPO_ROOT / "checkpoints/ffhq128_autoenc_latent/last.ckpt"
CALIBRATION = REPO_ROOT / "QATcode/quantize_ver2/calibration_diffae.pth"

# PRE-REGISTERED, DO NOT MODIFY: Q-DiffAE mainline baseline_908030 Stage2 threshold variant.
STAGE2_Q_ZONE = 0.90
STAGE2_Q_PEAK = 0.80
STAGE2_PEAK_OVER_ZONE_RATIO_MIN = 1.3
STAGE2_EVAL_NUM_IMAGES = 8
STAGE2_EVAL_CHUNK_SIZE = 1

PRESET_60 = {
    "K": [8, 12, 16, 20, 25],
    "sw": [2, 3, 5],
    "lambda": [0.25, 0.5, 1.0, 2.0],
    "k_max": [4],
}

EST_STAGE_A_MIN_PER_CONFIG = 1.0
EST_STAGE_A2_MIN_PER_CONFIG = 4.5
EST_FID5K_MIN_PER_CONFIG = 30.0
EST_FID50K_MIN_PER_CONFIG = 240.0
REFERENCE_EVAL_SAMPLES = 5000


def lam_str(value: float) -> str:
    return str(value)


def config_id(cfg: Dict[str, Any]) -> str:
    return f"K{cfg['K']}_sw{cfg['sw']}_lam{lam_str(cfg['lambda'])}_kmax{cfg['k_max']}"


def build_grid(name: str) -> List[Dict[str, Any]]:
    if name != "preset_60":
        raise ValueError(f"unknown config grid: {name}")
    out: List[Dict[str, Any]] = []
    for k in PRESET_60["K"]:
        for sw in PRESET_60["sw"]:
            for lam in PRESET_60["lambda"]:
                for k_max in PRESET_60["k_max"]:
                    cfg = {"K": k, "sw": sw, "lambda": lam, "k_max": k_max}
                    cfg["id"] = config_id(cfg)
                    out.append(cfg)
    return out


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_state(path: Path, grid: List[Dict[str, Any]]) -> Dict[str, Any]:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {
            "created_at": utc_now(),
            "updated_at": None,
            "grid_name": "preset_60",
            "configs": {cfg["id"]: cfg for cfg in grid},
            "stage_a": {},
            "stage_a2": {},
            "stage_b": {},
            "stage_c": {},
            "stage_d": {},
            "stage_e": {},
        }
    for key in ("stage_a", "stage_a2", "stage_b", "stage_c", "stage_d", "stage_e"):
        state.setdefault(key, {})
    state.setdefault("configs", {cfg["id"]: cfg for cfg in grid})
    return state


def reference_cache_path(eval_samples: int = REFERENCE_EVAL_SAMPLES) -> Path:
    # Mirrors TrainConfig.fid_cache plus metrics.evaluate_fid cache_dir suffix.
    return REPO_ROOT / (
        f"mycache/eval_images/ffhqlmdb256_size128_{eval_samples}_{eval_samples}"
    )


def protocol_metadata(python: str) -> Dict[str, Any]:
    gpu = check_gpu(python)
    return {
        "timestamp_committed": utc_now(),
        "grid": PRESET_60,
        "iteration_order": "K (outer) -> sw -> lambda -> k_max (inner), deterministic",
        "seed_policy": {"stage_b_fid5k": 0, "stage_d_fid50k": 0},
        "sampling_config": {"mode": "float", "num_steps": 100, "quant_state": "tt"},
        "stage2": {
            "pipeline": [
                "stage2_runtime_refine.py (global refine)",
                "build_blockwise_thresholds.py",
                "stage2_runtime_refine.py (2nd refine with threshold-config)",
            ],
            "threshold_variant": "Q-DiffAE mainline baseline_908030",
            "q_zone": STAGE2_Q_ZONE,
            "q_peak": STAGE2_Q_PEAK,
            "peak_over_zone_ratio_min": STAGE2_PEAK_OVER_ZONE_RATIO_MIN,
            "eval_num_images": STAGE2_EVAL_NUM_IMAGES,
            "eval_chunk_size": STAGE2_EVAL_CHUNK_SIZE,
            "fid_input": "stage2_refined_scheduler_config.json from Stage2 Step 3",
            "rationale": (
                "Fixed threshold matches published Q-DiffAE mainline setting to ensure "
                "resweep comparability with existing thesis Q-DiffAE table (FID@50K = 10.311)."
            ),
        },
        "selection_rule": (
            "Rank successful configs by FID@5K ascending; select top-3 for FID@50K evaluation"
        ),
        "checkpoints": {
            "qat_ckpt": str(QAT_CKPT),
            "base_diffae_ckpt": str(BASE_CKPT),
            "calibration": str(CALIBRATION),
        },
        "environment": {
            "python": python,
            "cuda_available": bool(gpu["ok"]),
            "cuda_detail": gpu["detail"],
        },
        "reference_stats": str(reference_cache_path()),
    }


def ensure_protocol_metadata(state: Dict[str, Any], python: str) -> None:
    protocol = state.get("protocol")
    if not isinstance(protocol, dict):
        state["protocol"] = protocol_metadata(python)
        return

    defaults = protocol_metadata(python)
    for key, value in defaults.items():
        if key == "stage2" and isinstance(protocol.get("stage2"), dict):
            for stage2_key, stage2_value in value.items():
                protocol["stage2"].setdefault(stage2_key, stage2_value)
        else:
            protocol.setdefault(key, value)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def ensure_dirs(output_dir: Path, dry_run: bool) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "stage_a": output_dir / "stage_a_stage1_sweep",
        "stage_a2": output_dir / "stage_a2_stage2",
        "stage_b": output_dir / "stage_b_fid5k",
        "stage_d": output_dir / "stage_d_fid50k",
    }
    if not dry_run:
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        (dirs["stage_b"] / "runs").mkdir(parents=True, exist_ok=True)
        (dirs["stage_d"] / "runs").mkdir(parents=True, exist_ok=True)
    return dirs


def first_existing_parent(path: Path) -> Path:
    cur = path
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def check_gpu(python: str) -> Dict[str, Any]:
    cmd = [
        python,
        "-c",
        "import torch; print('cuda_available=' + str(torch.cuda.is_available())); print('device_count=' + str(torch.cuda.device_count()))",
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
        text = out.stdout.strip()
        return {"ok": "cuda_available=True" in text, "detail": text}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def preflight(output_dir: Path, python: str, dry_run: bool) -> bool:
    checks = [
        ("python", Path(python).is_file(), python),
        ("stage0_dir", STAGE0_DIR.is_dir(), str(STAGE0_DIR)),
        ("stage1_scheduler", STAGE1_SCHEDULER.is_file(), str(STAGE1_SCHEDULER)),
        ("stage2_refine_script", STAGE2_REFINE_SCRIPT.is_file(), str(STAGE2_REFINE_SCRIPT)),
        ("stage2_threshold_script", STAGE2_THRESH_SCRIPT.is_file(), str(STAGE2_THRESH_SCRIPT)),
        ("fid_script", FID_SCRIPT.is_file(), str(FID_SCRIPT)),
        ("qat_ckpt", QAT_CKPT.is_file(), str(QAT_CKPT)),
        ("base_ckpt", BASE_CKPT.is_file(), str(BASE_CKPT)),
        ("calibration", CALIBRATION.is_file(), str(CALIBRATION)),
    ]
    parent = first_existing_parent(output_dir)
    checks.append(("output_parent_writable", os.access(parent, os.W_OK), str(parent)))
    gpu = check_gpu(python)
    checks.append(("gpu_available", bool(gpu["ok"]), str(gpu["detail"])))

    print("Pre-flight checks:")
    ok = True
    for name, passed, detail in checks:
        mark = "OK" if passed else "FAIL"
        print(f"  [{mark}] {name}: {detail}")
        ok = ok and bool(passed)

    if not dry_run and not ok:
        print("Pre-flight failed; aborting before launching stages.", file=sys.stderr)
    return ok


def write_command_header(log_path: Path, cmd: List[str], cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Started: {utc_now()}\n")
        f.write(f"# CWD: {cwd}\n")
        f.write("# Command:\n")
        f.write(shlex.join(cmd) + "\n\n")


def run_command(cmd: List[str], log_path: Path, dry_run: bool) -> Dict[str, Any]:
    cmd_str = shlex.join(cmd)
    if dry_run:
        print(f"[dry-run] {cmd_str}")
        print(f"[dry-run] log: {log_path}")
        return {"status": "dry_run", "command": cmd_str, "returncode": None}

    write_command_header(log_path, cmd, REPO_ROOT)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
        return {"status": "success", "command": cmd_str, "returncode": 0}
    except subprocess.CalledProcessError as exc:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n# FAILED returncode={exc.returncode} at {utc_now()}\n")
        return {"status": "failed", "command": cmd_str, "returncode": exc.returncode}


def stage_a(
    grid: List[Dict[str, Any]],
    state: Dict[str, Any],
    state_path: Path,
    dirs: Dict[str, Path],
    python: str,
    dry_run: bool,
) -> None:
    for cfg in grid:
        cid = cfg["id"]
        out_dir = dirs["stage_a"] / cid
        log_path = dirs["stage_a"] / f"{cid}.log"
        sched_path = out_dir / "scheduler_config.json"
        prev = state["stage_a"].get(cid, {})
        if sched_path.is_file() or (
            prev.get("status") == "success" and Path(prev.get("scheduler_config", "")).is_file()
        ):
            print(f"[stage_a] skip completed {cid}")
            if not dry_run and prev.get("status") != "success":
                state["stage_a"][cid] = {
                    "status": "success",
                    "scheduler_config": str(sched_path),
                    "log": str(log_path),
                    "resume_source": "existing_output",
                }
                save_state(state_path, state)
            continue
        cmd = [
            python,
            str(STAGE1_SCHEDULER.relative_to(REPO_ROOT)),
            "--stage0_dir",
            str(STAGE0_DIR.relative_to(REPO_ROOT)),
            "--output_dir",
            str(out_dir),
            "--K",
            str(cfg["K"]),
            "--smooth_window",
            str(cfg["sw"]),
            "--lambda",
            str(cfg["lambda"]),
            "--k_max",
            str(cfg["k_max"]),
        ]
        result = run_command(cmd, log_path, dry_run)
        if dry_run:
            continue
        if result["status"] == "success" and not sched_path.is_file():
            result["status"] = "failed"
            result["error"] = f"missing scheduler_config.json: {sched_path}"
        result.update({"scheduler_config": str(sched_path), "log": str(log_path)})
        state["stage_a"][cid] = result
        save_state(state_path, state)


def stage1_scheduler_path(dirs: Dict[str, Path], cid: str) -> Path:
    return dirs["stage_a"] / cid / "scheduler_config.json"


def stage2_config_dir(dirs: Dict[str, Path], cid: str) -> Path:
    return dirs["stage_a2"] / cid


def stage2_refined_scheduler_path(dirs: Dict[str, Path], cid: str) -> Path:
    return stage2_config_dir(dirs, cid) / "baseline" / "stage2_refined_scheduler_config.json"


def stage2_threshold_path(dirs: Dict[str, Path], cid: str) -> Path:
    return stage2_config_dir(dirs, cid) / "01_blockwise_threshold" / "stage2_thresholds_blockwise.json"


def stage2_step1_command(python: str, scheduler_config: Path, output_dir: Path) -> List[str]:
    return [
        python,
        str(STAGE2_REFINE_SCRIPT.relative_to(REPO_ROOT)),
        "--scheduler_config",
        str(scheduler_config),
        "--output_dir",
        str(output_dir),
        "--seed",
        "0",
        "--best_ckpt",
        str(QAT_CKPT),
        "--model_path",
        str(BASE_CKPT),
        "--calib",
        str(CALIBRATION),
        "--eval-num-images",
        str(STAGE2_EVAL_NUM_IMAGES),
        "--eval-chunk-size",
        str(STAGE2_EVAL_CHUNK_SIZE),
    ]


def stage2_step2_command(diagnostics: Path, output_json: Path, python: str) -> List[str]:
    return [
        python,
        str(STAGE2_THRESH_SCRIPT.relative_to(REPO_ROOT)),
        "--diagnostics",
        str(diagnostics),
        "--output",
        str(output_json),
        "--q_zone",
        f"{STAGE2_Q_ZONE:.2f}",
        "--q_peak",
        f"{STAGE2_Q_PEAK:.2f}",
        "--peak_over_zone_ratio_min",
        f"{STAGE2_PEAK_OVER_ZONE_RATIO_MIN:.1f}",
    ]


def stage2_step3_command(
    python: str,
    scheduler_config: Path,
    output_dir: Path,
    threshold_config: Path,
) -> List[str]:
    cmd = stage2_step1_command(python, scheduler_config, output_dir)
    cmd.extend(["--threshold-config", str(threshold_config)])
    return cmd


def stage_a2(
    grid: List[Dict[str, Any]],
    state: Dict[str, Any],
    state_path: Path,
    dirs: Dict[str, Path],
    python: str,
    dry_run: bool,
    simulate_failures: int = 0,
) -> None:
    for idx, cfg in enumerate(grid):
        cid = cfg["id"]
        cfg_dir = stage2_config_dir(dirs, cid)
        step1_dir = cfg_dir / "00_global_refine"
        step2_dir = cfg_dir / "01_blockwise_threshold"
        step3_dir = cfg_dir / "baseline"
        threshold_json = stage2_threshold_path(dirs, cid)
        refined_json = stage2_refined_scheduler_path(dirs, cid)
        stage1_json = stage1_scheduler_path(dirs, cid)
        prev = state["stage_a2"].get(cid, {})
        if refined_json.is_file() or (
            prev.get("status") == "success" and Path(prev.get("step3_refined_scheduler_json", "")).is_file()
        ):
            print(f"[stage_a2] skip completed {cid}")
            if not dry_run and prev.get("status") != "success":
                state["stage_a2"][cid] = {
                    "status": "success",
                    "step1_output_dir": str(step1_dir),
                    "step2_threshold_json": str(threshold_json),
                    "step3_refined_scheduler_json": str(refined_json),
                    "resume_source": "existing_output",
                    "finished_at": utc_now(),
                    "wall_time_sec": 0.0,
                }
                save_state(state_path, state)
            continue

        commands = [
            ("step1", stage2_step1_command(python, stage1_json, step1_dir), cfg_dir / "step1.log"),
            (
                "step2",
                stage2_step2_command(
                    step1_dir / "stage2_runtime_diagnostics.json",
                    threshold_json,
                    python,
                ),
                cfg_dir / "step2.log",
            ),
            (
                "step3",
                stage2_step3_command(python, stage1_json, step3_dir, threshold_json),
                cfg_dir / "step3.log",
            ),
        ]

        if dry_run:
            if idx == 0:
                print(f"[dry-run] stage_a2 first config: {cid}")
                for step_name, cmd, log_path in commands:
                    print(f"[dry-run] {step_name}: {shlex.join(cmd)}")
                    print(f"[dry-run] log: {log_path}")
            elif idx == 1:
                print(f"[dry-run] ... stage_a2 has {len(grid)} configs total (3 commands each)")
            if simulate_failures > 0 and idx >= len(grid) - int(simulate_failures):
                print(f"[dry-run] simulate stage_a2 failure for {cid}")
            continue

        started = utc_now()
        start_ts = dt.datetime.now(dt.timezone.utc)
        record: Dict[str, Any] = {
            "status": "running",
            "started_at": started,
            "step1_output_dir": str(step1_dir),
            "step2_threshold_json": str(threshold_json),
            "step3_refined_scheduler_json": str(refined_json),
        }
        state["stage_a2"][cid] = record
        save_state(state_path, state)

        for out_dir in (step1_dir, step2_dir, step3_dir):
            out_dir.mkdir(parents=True, exist_ok=True)

        if not stage1_json.is_file():
            record.update(
                {
                    "status": "failed",
                    "failed_step": "precheck",
                    "error": f"missing Stage1 scheduler: {stage1_json}",
                    "finished_at": utc_now(),
                    "wall_time_sec": 0.0,
                }
            )
            save_state(state_path, state)
            print(f"[stage_a2] missing Stage1 scheduler for {cid}; continue")
            continue

        failed_step: Optional[str] = None
        error_msg: Optional[str] = None
        for step_name, cmd, log_path in commands:
            result = run_command(cmd, log_path, dry_run=False)
            if result["status"] != "success":
                failed_step = step_name
                error_msg = f"subprocess failed returncode={result.get('returncode')}"
                break
            if step_name == "step1" and not (step1_dir / "stage2_runtime_diagnostics.json").is_file():
                failed_step = step_name
                error_msg = f"missing diagnostics: {step1_dir / 'stage2_runtime_diagnostics.json'}"
                break
            if step_name == "step2" and not threshold_json.is_file():
                failed_step = step_name
                error_msg = f"missing threshold json: {threshold_json}"
                break
            if step_name == "step3" and not refined_json.is_file():
                failed_step = step_name
                error_msg = f"missing refined scheduler: {refined_json}"
                break

        end_ts = dt.datetime.now(dt.timezone.utc)
        record.update(
            {
                "status": "failed" if failed_step else "success",
                "failed_step": failed_step,
                "error": error_msg,
                "finished_at": end_ts.isoformat(timespec="seconds"),
                "wall_time_sec": round((end_ts - start_ts).total_seconds(), 2),
            }
        )
        state["stage_a2"][cid] = record
        save_state(state_path, state)


def fid_command(
    *,
    python: str,
    scheduler_config: Path,
    scheduler_name: str,
    eval_samples: int,
    run_output_dir: Path,
    runs_index_path: Path,
    log_path: Path,
) -> List[str]:
    return [
        python,
        str(FID_SCRIPT.relative_to(REPO_ROOT)),
        "--mode",
        "float",
        "--num_steps",
        "100",
        "--eval_samples",
        str(eval_samples),
        "--seed",
        "0",
        "--quant-state",
        "tt",
        "--use_cache_scheduler",
        "--cache_scheduler_json",
        str(scheduler_config),
        "--scheduler-name",
        scheduler_name,
        "--run-output-dir",
        str(run_output_dir),
        "--runs-index-path",
        str(runs_index_path),
        "--log_file",
        str(log_path),
    ]


def stage_fid(
    *,
    stage_key: str,
    stage_dir: Path,
    scheduler_stage_dir: Path,
    configs: Iterable[Dict[str, Any]],
    eval_samples: int,
    state: Dict[str, Any],
    state_path: Path,
    python: str,
    dry_run: bool,
) -> None:
    runs_index = stage_dir / "runs_index.jsonl"
    for cfg in configs:
        cid = cfg["id"]
        prev = state[stage_key].get(cid, {})
        if prev.get("status") == "success" and Path(prev.get("summary", "")).is_file():
            print(f"[{stage_key}] skip completed {cid}")
            continue
        sched = Path(
            state["stage_a2"].get(cid, {}).get(
                "step3_refined_scheduler_json",
                str(scheduler_stage_dir / cid / "baseline" / "stage2_refined_scheduler_config.json"),
            )
        )
        if not dry_run and not sched.is_file():
            state[stage_key][cid] = {
                "status": "failed",
                "error": f"missing Stage2 refined scheduler for {cid}: {sched}",
            }
            save_state(state_path, state)
            print(f"[{stage_key}] missing Stage2 refined scheduler for {cid}; continue")
            continue
        run_dir = stage_dir / "runs" / cid
        log_path = stage_dir / f"{cid}.log"
        cmd = fid_command(
            python=python,
            scheduler_config=sched,
            scheduler_name=cid,
            eval_samples=eval_samples,
            run_output_dir=run_dir,
            runs_index_path=runs_index,
            log_path=log_path,
        )
        result = run_command(cmd, log_path, dry_run)
        if dry_run:
            continue
        summary = run_dir / "summary.json"
        if result["status"] == "success" and not summary.is_file():
            result["status"] = "failed"
            result["error"] = f"missing summary.json: {summary}"
        result.update({"summary": str(summary), "run_output_dir": str(run_dir), "log": str(log_path)})
        state[stage_key][cid] = result
        save_state(state_path, state)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_fid(summary_path: Path, expected_key: Optional[str] = None) -> Optional[float]:
    if not summary_path.is_file():
        return None
    data = read_json(summary_path)
    if "fid" in data and data["fid"] is not None:
        if expected_key is None or data.get("fid_key") == expected_key:
            return float(data["fid"])
    if expected_key and data.get(expected_key) is not None:
        return float(data[expected_key])
    if data.get("fid_5k") is not None:
        return float(data["fid_5k"])
    return None


def read_rho(summary_path: Path) -> Optional[float]:
    if not summary_path.is_file():
        return None
    data = read_json(summary_path)
    value = data.get("full_compute_ratio")
    return float(value) if value is not None else None


def failed_or_missing_stage_b(grid: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for cfg in grid:
        cid = cfg["id"]
        rec = state["stage_b"].get(cid)
        if not rec:
            failures.append({"config_id": cid, "stage": "stage_b", "reason": "missing"})
            continue
        status = rec.get("status", "missing")
        if status != "success":
            failures.append(
                {
                    "config_id": cid,
                    "stage": "stage_b",
                    "reason": str(rec.get("error") or status),
                }
            )
            continue
        summary = Path(rec.get("summary", ""))
        if read_fid(summary, "fid_5k") is None:
            failures.append(
                {
                    "config_id": cid,
                    "stage": "stage_b",
                    "reason": "missing FID@5K in summary.json",
                }
            )
    return failures


def stage_a_success_count(grid: List[Dict[str, Any]], state: Dict[str, Any], dirs: Dict[str, Path]) -> int:
    count = 0
    for cfg in grid:
        cid = cfg["id"]
        rec = state["stage_a"].get(cid, {})
        sched = Path(rec.get("scheduler_config", str(stage1_scheduler_path(dirs, cid))))
        if sched.is_file():
            count += 1
    return count


def stage_a2_success_count(grid: List[Dict[str, Any]], state: Dict[str, Any], dirs: Dict[str, Path]) -> int:
    count = 0
    for cfg in grid:
        cid = cfg["id"]
        rec = state["stage_a2"].get(cid, {})
        refined = Path(
            rec.get("step3_refined_scheduler_json", str(stage2_refined_scheduler_path(dirs, cid)))
        )
        if refined.is_file():
            count += 1
    return count


def missing_stage_a2_outputs(
    grid: List[Dict[str, Any]], state: Dict[str, Any], dirs: Dict[str, Path]
) -> List[Dict[str, str]]:
    missing: List[Dict[str, str]] = []
    for cfg in grid:
        cid = cfg["id"]
        rec = state["stage_a2"].get(cid, {})
        refined = Path(
            rec.get("step3_refined_scheduler_json", str(stage2_refined_scheduler_path(dirs, cid)))
        )
        if refined.is_file():
            continue
        missing.append(
            {
                "config_id": cid,
                "stage": "A2",
                "failed_step": str(rec.get("failed_step") or ""),
                "reason": str(rec.get("error") or rec.get("status") or "missing Stage2 refined scheduler"),
            }
        )
    return missing


def ensure_stage_a2_complete_for_fid(
    grid: List[Dict[str, Any]], state: Dict[str, Any], dirs: Dict[str, Path], stage_name: str
) -> bool:
    missing = missing_stage_a2_outputs(grid, state, dirs)
    if not missing:
        return True
    print(
        f"[{stage_name}] refused to start: Stage A2 refined schedulers are incomplete "
        f"({len(grid) - len(missing)}/{len(grid)} ready)."
    )
    print("Run `--stage stage_a2` first, then resume this stage.")
    print("Missing/failed Stage A2 configs:")
    for item in missing[:20]:
        step = f" {item['failed_step']}" if item.get("failed_step") else ""
        print(f"  - {item['config_id']}: A2{step} | {item['reason']}")
    if len(missing) > 20:
        print(f"  ... {len(missing) - 20} more")
    return False


def stage_b_success_count(grid: List[Dict[str, Any]], state: Dict[str, Any]) -> int:
    count = 0
    for cfg in grid:
        rec = state["stage_b"].get(cfg["id"], {})
        if rec.get("status") == "success" and read_fid(Path(rec.get("summary", "")), "fid_5k") is not None:
            count += 1
    return count


def stage_d_success_count(grid: List[Dict[str, Any]], state: Dict[str, Any]) -> int:
    count = 0
    for cfg in grid:
        rec = state["stage_d"].get(cfg["id"], {})
        if rec.get("status") == "success" and read_fid(Path(rec.get("summary", "")), "fid_50k") is not None:
            count += 1
    return count


def collect_failed_or_missing_configs(
    grid: List[Dict[str, Any]], state: Dict[str, Any], dirs: Dict[str, Path]
) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for cfg in grid:
        cid = cfg["id"]
        a = state["stage_a"].get(cid, {})
        a_sched = Path(a.get("scheduler_config", str(stage1_scheduler_path(dirs, cid))))
        if not a_sched.is_file():
            failures.append(
                {
                    "config_id": cid,
                    "stage": "A",
                    "failed_step": "",
                    "reason": str(a.get("error") or a.get("status") or "missing Stage1 scheduler"),
                }
            )
            continue

        a2 = state["stage_a2"].get(cid, {})
        refined = Path(a2.get("step3_refined_scheduler_json", str(stage2_refined_scheduler_path(dirs, cid))))
        if not refined.is_file():
            failures.append(
                {
                    "config_id": cid,
                    "stage": "A2",
                    "failed_step": str(a2.get("failed_step") or ""),
                    "reason": str(a2.get("error") or a2.get("status") or "missing Stage2 refined scheduler"),
                }
            )
            continue

        b = state["stage_b"].get(cid, {})
        if b.get("status") != "success":
            failures.append(
                {
                    "config_id": cid,
                    "stage": "B",
                    "failed_step": "",
                    "reason": str(b.get("error") or b.get("status") or "missing FID@5K run"),
                }
            )
            continue
        if read_fid(Path(b.get("summary", "")), "fid_5k") is None:
            failures.append(
                {
                    "config_id": cid,
                    "stage": "B",
                    "failed_step": "",
                    "reason": "missing FID@5K in summary.json",
                }
            )
    return failures


def print_incomplete_warning(success_count: int, total: int, failures: List[Dict[str, str]]) -> None:
    failed_ids = ", ".join(item["config_id"] for item in failures) or "<none>"
    print(f"WARNING: Sweep incomplete: {success_count}/{total} configs succeeded.")
    print(f"    Failed configs: {failed_ids}")
    print("    Selection rule was pre-registered for 60 configs.")
    print("")
    print("    Options:")
    print("    (a) Investigate failures and re-run failed configs (recommended)")
    print("    (b) Proceed with incomplete selection using --allow-incomplete-selection")


def stage_c(
    grid: List[Dict[str, Any]],
    state: Dict[str, Any],
    top_k: int,
    dry_run: bool,
    allow_incomplete_selection: bool,
    simulate_failures: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cfg in grid:
        cid = cfg["id"]
        rec = state["stage_b"].get(cid, {})
        summary = Path(rec.get("summary", ""))
        fid = read_fid(summary, "fid_5k")
        rows.append(
            {
                "config_id": cid,
                "fid_5k": fid,
                "status": rec.get("status", "missing"),
                "summary": str(summary) if str(summary) != "." else "",
            }
        )
    total = len(grid)
    if dry_run and simulate_failures <= 0 and not state.get("stage_b"):
        print(
            f"[dry-run] stage_c would rank successful FID@5K runs after Stage B completes "
            f"and pick top {top_k}."
        )
        return []
    if dry_run and simulate_failures > 0:
        success_count = max(0, total - int(simulate_failures))
        failures = [
            {"config_id": cfg["id"], "stage": "stage_b", "reason": "simulated failure"}
            for cfg in grid[-int(simulate_failures) :]
        ]
    else:
        success_count = stage_b_success_count(grid, state)
        failures = failed_or_missing_stage_b(grid, state)

    ranked = sorted(
        (r for r in rows if r["fid_5k"] is not None and r["status"] == "success"),
        key=lambda r: float(r["fid_5k"]),
    )
    if success_count < total:
        print_incomplete_warning(success_count, total, failures)
        state["stage_c"] = {
            "status": "incomplete_allowed" if allow_incomplete_selection else "skipped_incomplete",
            "success_count": success_count,
            "total_count": total,
            "failed_configs": failures,
            "allow_incomplete_selection": bool(allow_incomplete_selection),
            "warning": (
                f"top-3 selected from {success_count}/{total} successful configs (incomplete sweep)"
                if allow_incomplete_selection
                else "top-3 selection skipped due to incomplete sweep"
            ),
            "updated_at": utc_now(),
        }
        if not allow_incomplete_selection:
            return []

    top = ranked[: int(top_k)]
    if dry_run:
        print(f"[dry-run] stage_c would rank {len(ranked)} successful FID@5K runs and pick top {top_k}.")
        return top
    state["stage_c"] = {
        "status": "success" if top else "failed",
        "ranked": ranked,
        "top_k": top,
        "success_count": success_count,
        "total_count": total,
        "failed_configs": failures,
        "allow_incomplete_selection": bool(allow_incomplete_selection),
        "updated_at": utc_now(),
    }
    return top


def configs_by_ids(grid: List[Dict[str, Any]], ids: Iterable[str]) -> List[Dict[str, Any]]:
    by_id = {cfg["id"]: cfg for cfg in grid}
    return [by_id[cid] for cid in ids if cid in by_id]


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_summary(
    grid: List[Dict[str, Any]],
    state: Dict[str, Any],
    output_dir: Path,
    final: bool = False,
) -> None:
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    rows: List[Dict[str, Any]] = []
    for cfg in grid:
        cid = cfg["id"]
        b = state["stage_b"].get(cid, {})
        d = state["stage_d"].get(cid, {})
        b_summary = Path(b.get("summary", ""))
        d_summary = Path(d.get("summary", ""))
        rows.append(
            {
                "config": cid,
                "K": cfg["K"],
                "sw": cfg["sw"],
                "lambda": cfg["lambda"],
                "k_max": cfg["k_max"],
                "fid_5k": read_fid(b_summary, "fid_5k"),
                "fid_50k": read_fid(d_summary, "fid_50k"),
                "rho": read_rho(d_summary) or read_rho(b_summary),
                "status_5k": b.get("status", "missing"),
                "status_50k": d.get("status", "missing"),
                "summary_5k": str(b_summary) if str(b_summary) != "." else "",
                "summary_50k": str(d_summary) if str(d_summary) != "." else "",
            }
        )
    ranked = sorted((r for r in rows if r["fid_5k"] is not None), key=lambda r: float(r["fid_5k"]))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    top_ids = [r.get("config_id") for r in state.get("stage_c", {}).get("top_k", [])]
    top_rows = [r for r in rows if r["config"] in top_ids]
    protocol = state.get("protocol", {})
    stage_c_state = state.get("stage_c", {})
    dirs = {
        "stage_a": output_dir / "stage_a_stage1_sweep",
        "stage_a2": output_dir / "stage_a2_stage2",
        "stage_b": output_dir / "stage_b_fid5k",
        "stage_d": output_dir / "stage_d_fid50k",
    }
    failures = collect_failed_or_missing_configs(grid, state, dirs)
    stage_a_success = stage_a_success_count(grid, state, dirs)
    stage_a2_success = stage_a2_success_count(grid, state, dirs)
    stage_b_success = stage_b_success_count(grid, state)
    stage_d_success = stage_d_success_count(grid, state)
    total_count = len(grid)
    sweep_complete = stage_b_success == total_count
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q-DiffAE Resweep Summary\n\n")
        f.write("## Pre-Registration Protocol\n\n")
        grid_meta = protocol.get("grid", PRESET_60)
        f.write(f"Timestamp committed: {protocol.get('timestamp_committed', state.get('created_at', ''))}\n\n")
        f.write("### Stage 1 (zone segmentation)\n\n")
        f.write(
            "- Grid: "
            f"K={grid_meta.get('K', PRESET_60['K'])}, "
            f"sw={grid_meta.get('sw', PRESET_60['sw'])}, "
            f"lambda={grid_meta.get('lambda', PRESET_60['lambda'])}, "
            f"k_max={grid_meta.get('k_max', PRESET_60['k_max'])} = 60 configs\n"
        )
        f.write(f"- Iteration order: {protocol.get('iteration_order', '')}\n\n")

        f.write("### Stage 2 (refinement)\n\n")
        stage2 = protocol.get("stage2", {})
        f.write("- Three-step pipeline (fixed, not swept):\n")
        f.write("  - Step 1: stage2_runtime_refine.py (global refine)\n")
        f.write("  - Step 2: build_blockwise_thresholds.py\n")
        f.write("  - Step 3: stage2_runtime_refine.py (2nd refine with threshold-config)\n")
        f.write("- Threshold parameters (Q-DiffAE mainline `baseline_908030` variant):\n")
        f.write(f"  - q_zone = {float(stage2.get('q_zone', STAGE2_Q_ZONE)):.2f}\n")
        f.write(f"  - q_peak = {float(stage2.get('q_peak', STAGE2_Q_PEAK)):.2f}\n")
        f.write(
            "  - peak_over_zone_ratio_min = "
            f"{float(stage2.get('peak_over_zone_ratio_min', STAGE2_PEAK_OVER_ZONE_RATIO_MIN)):.1f}\n"
        )
        f.write(f"- Rationale: {stage2.get('rationale', '')}\n\n")

        f.write("### FID evaluation\n\n")
        seed_policy = protocol.get("seed_policy", {})
        sampling = protocol.get("sampling_config", {})
        f.write(
            "- Stage B: FID@5K on 60 configs, "
            f"seed={seed_policy.get('stage_b_fid5k', 0)}, "
            f"mode={sampling.get('mode', 'float')}, "
            f"num_steps={sampling.get('num_steps', 100)}, "
            f"quant-state={sampling.get('quant_state', 'tt')}\n"
        )
        f.write("- Stage D: FID@50K on top-3 configs (same sampling config)\n")
        f.write("- FID input: Step 3 refined scheduler (`stage2_refined_scheduler_config.json`)\n\n")

        f.write("### Selection rule\n\n")
        f.write("- Rank 60 configs by FID@5K ascending\n")
        f.write("- Select top-3 for FID@50K evaluation\n")
        f.write("- If sweep incomplete (< 60 success), require --allow-incomplete-selection\n\n")

        checkpoints = protocol.get("checkpoints", {})
        f.write("### Environment / fixed inputs\n\n")
        f.write("- Checkpoints:\n")
        f.write(f"  - QAT ckpt: {checkpoints.get('qat_ckpt', QAT_CKPT)}\n")
        f.write(f"  - Base Diff-AE ckpt: {checkpoints.get('base_diffae_ckpt', BASE_CKPT)}\n")
        f.write(f"  - Calibration: {checkpoints.get('calibration', CALIBRATION)}\n")
        environment = protocol.get("environment", {})
        f.write(
            f"- Environment: {environment.get('python', DEFAULT_PYTHON)}, "
            f"CUDA: {environment.get('cuda_detail', '')}\n"
        )
        f.write(f"- Reference stats: {protocol.get('reference_stats', reference_cache_path())}\n\n")

        f.write("## Sweep Status\n\n")
        f.write(f"- Stage A (Stage 1 scheduler synthesis): {stage_a_success}/{total_count}\n")
        f.write(f"- Stage A2 (Stage 2 refinement): {stage_a2_success}/{total_count}\n")
        f.write(f"- Stage B (FID@5K): {stage_b_success}/{total_count}\n")
        f.write(f"- Stage C (top-3 selection): {stage_c_state.get('status', 'missing')}\n")
        f.write(f"- Stage D (FID@50K on top-3): {stage_d_success}/{len(top_ids) or 3}\n")
        f.write(f"- Sweep complete through Stage B: {'Yes' if sweep_complete else 'No'}\n")
        if stage_c_state.get("warning"):
            f.write(f"- **selection warning**: {stage_c_state['warning']}\n")
        f.write("\n")

        f.write("## Stage 1 Sweep Results (60 configs, ranked by FID@5K)\n\n")
        f.write("| Rank | Config | FID@5K | Status |\n")
        f.write("|------|--------|--------|--------|\n")
        for i, row in enumerate(ranked, 1):
            f.write(f"| {i} | {row['config']} | {format_float(row['fid_5k'])} | {row['status_5k']} |\n")
        f.write("\n## Top-3 FID@50K\n\n")
        if stage_c_state.get("status") == "skipped_incomplete":
            f.write("Top-3 selection skipped due to incomplete sweep.\n")
        elif not top_rows:
            f.write("Top-3 not available yet.\n")
        else:
            f.write("| Config | FID@5K | FID@50K | rho | Speedup |\n")
            f.write("|--------|--------|---------|-----|---------|\n")
            for row in top_rows:
                rho = row["rho"]
                speedup = (1.0 / float(rho)) if rho and float(rho) > 0 else None
                f.write(
                    f"| {row['config']} | {format_float(row['fid_5k'])} | "
                    f"{format_float(row['fid_50k'])} | {format_float(rho)} | {format_float(speedup, 2)} |\n"
                )

        f.write("\n## Failed / Missing Configs\n\n")
        if not failures:
            f.write("None -- sweep complete.\n")
        else:
            f.write("| Config ID | Failed at Stage | Failed Step | Failure Reason |\n")
            f.write("|-----------|-----------------|-------------|----------------|\n")
            for item in failures:
                f.write(
                    f"| {item['config_id']} | {item['stage']} | "
                    f"{item.get('failed_step', '')} | {item['reason']} |\n"
                )

    state["summary"] = {
        "updated_at": utc_now(),
        "summary_csv": str(csv_path),
        "summary_md": str(md_path),
    }
    if final:
        state["stage_e"] = {"status": "success", "summary_csv": str(csv_path), "summary_md": str(md_path)}


def print_plan(grid: List[Dict[str, Any]], output_dir: Path, top_k: int) -> None:
    n = len(grid)
    total_min = (
        n * EST_STAGE_A_MIN_PER_CONFIG
        + n * EST_STAGE_A2_MIN_PER_CONFIG
        + n * EST_FID5K_MIN_PER_CONFIG
        + top_k * EST_FID50K_MIN_PER_CONFIG
    )
    print("Dry-run plan:")
    print(f"  grid: preset_60 ({n} configs)")
    print(f"  first config: {grid[0]['id']}")
    print(f"  last config: {grid[-1]['id']}")
    print(f"  stage A: build {n} Stage1 scheduler_config.json files")
    print("  stage A estimate: 1.0 hour (existing outputs may already satisfy this)")
    print(f"  stage A2: run Stage2 refinement for {n} configs (3 commands each)")
    print(f"  stage A2 estimate: {n} configs x 3 steps x ~1.5 min = {n * EST_STAGE_A2_MIN_PER_CONFIG / 60.0:.1f} hours")
    print(f"  stage B: run FID@5K for {n} configs using Stage2 refined schedulers")
    print(f"  stage B estimate: {n} configs x ~{EST_FID5K_MIN_PER_CONFIG:g} min = {n * EST_FID5K_MIN_PER_CONFIG / 60.0:.1f} hours")
    print(f"  stage C: rank by FID@5K and select top {top_k}")
    print("  stage C estimate: 0.0 hours")
    print(f"  stage D: run FID@50K for top {top_k} using Stage2 refined schedulers")
    print(f"  stage D estimate: {top_k} configs x ~{EST_FID50K_MIN_PER_CONFIG:g} min = {top_k * EST_FID50K_MIN_PER_CONFIG / 60.0:.1f} hours")
    print("  stage E: write summary.csv and summary.md")
    print("  stage E estimate: 0.0 hours")
    print(f"  output: {output_dir}")
    print(f"  rough estimated wall time: {total_min / 60.0:.1f} hours")


def print_protocol_preview(state: Dict[str, Any]) -> None:
    protocol = state.get("protocol", {})
    grid_meta = protocol.get("grid", PRESET_60)
    print("Protocol preview:")
    print(f"  timestamp_committed: {protocol.get('timestamp_committed', state.get('created_at', ''))}")
    print(
        "  grid: "
        f"K={grid_meta.get('K', PRESET_60['K'])}, "
        f"sw={grid_meta.get('sw', PRESET_60['sw'])}, "
        f"lambda={grid_meta.get('lambda', PRESET_60['lambda'])}, "
        f"k_max={grid_meta.get('k_max', PRESET_60['k_max'])}"
    )
    print(f"  iteration_order: {protocol.get('iteration_order', '')}")
    stage2 = protocol.get("stage2", {})
    print("  stage2_pipeline: global refine -> blockwise thresholds -> threshold refine")
    print(
        "  stage2_thresholds: "
        f"variant={stage2.get('threshold_variant', 'Q-DiffAE mainline baseline_908030')}, "
        f"q_zone={float(stage2.get('q_zone', STAGE2_Q_ZONE)):.2f}, "
        f"q_peak={float(stage2.get('q_peak', STAGE2_Q_PEAK)):.2f}, "
        "peak_over_zone_ratio_min="
        f"{float(stage2.get('peak_over_zone_ratio_min', STAGE2_PEAK_OVER_ZONE_RATIO_MIN)):.1f}"
    )
    print("  fid_input: Stage A2 Step 3 stage2_refined_scheduler_config.json")
    print("  seed_policy: Stage B=0, Stage D=0")
    print("  sampling_config: mode=float, num_steps=100, quant-state=tt")
    checkpoints = protocol.get("checkpoints", {})
    print(f"  qat_ckpt: {checkpoints.get('qat_ckpt', QAT_CKPT)}")
    print(f"  base_diffae_ckpt: {checkpoints.get('base_diffae_ckpt', BASE_CKPT)}")
    print(f"  calibration: {checkpoints.get('calibration', CALIBRATION)}")
    environment = protocol.get("environment", {})
    print(f"  environment: {environment.get('python', DEFAULT_PYTHON)} | {environment.get('cuda_detail', '')}")
    print(f"  reference_stats: {protocol.get('reference_stats', reference_cache_path())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q-DiffAE pre-registered resweep orchestrator")
    parser.add_argument(
        "--stage",
        choices=["all", "stage_a", "stage_a2", "stage_b", "stage_c", "stage_d", "stage_e"],
        default="all",
    )
    parser.add_argument("--config-grid", choices=["preset_60"], default="preset_60")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--python", type=str, default=os.environ.get("PYTHON", DEFAULT_PYTHON))
    parser.add_argument(
        "--allow-incomplete-selection",
        action="store_true",
        help="Explicitly allow Stage C to select top-k from an incomplete Stage B sweep.",
    )
    parser.add_argument(
        "--simulate-failures",
        type=int,
        default=0,
        help="Dry-run only: simulate N Stage A2/Stage B failures for guard previews.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid = build_grid(args.config_grid)
    output_dir = Path(args.output_dir).resolve()
    dirs = ensure_dirs(output_dir, args.dry_run)
    state_path = output_dir / "resweep_state.json"
    state = load_state(state_path, grid)
    ensure_protocol_metadata(state, args.python)

    if args.dry_run:
        print_plan(grid, output_dir, int(args.top_k))
        print_protocol_preview(state)

    ok = preflight(output_dir, args.python, args.dry_run)
    if not ok and not args.dry_run:
        return 2
    if not args.dry_run:
        save_state(state_path, state)

    stages = (
        ["stage_a", "stage_a2", "stage_b", "stage_c", "stage_d", "stage_e"]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages:
        if stage == "stage_a":
            stage_a(grid, state, state_path, dirs, args.python, args.dry_run)
        elif stage == "stage_a2":
            stage_a2(
                grid,
                state,
                state_path,
                dirs,
                args.python,
                args.dry_run,
                int(args.simulate_failures),
            )
        elif stage == "stage_b":
            if not args.dry_run and not ensure_stage_a2_complete_for_fid(
                grid, state, dirs, "stage_b"
            ):
                write_summary(grid, state, output_dir, final=False)
                save_state(state_path, state)
                return 2
            stage_fid(
                stage_key="stage_b",
                stage_dir=dirs["stage_b"],
                scheduler_stage_dir=dirs["stage_a2"],
                configs=grid,
                eval_samples=5000,
                state=state,
                state_path=state_path,
                python=args.python,
                dry_run=args.dry_run,
            )
        elif stage == "stage_c":
            stage_c(
                grid,
                state,
                int(args.top_k),
                args.dry_run,
                bool(args.allow_incomplete_selection),
                int(args.simulate_failures),
            )
        elif stage == "stage_d":
            top = state.get("stage_c", {}).get("top_k", [])
            if not top and args.dry_run:
                print("[dry-run] stage_d top-3 configs are unknown until stage_c has real FID@5K results.")
                top_configs: List[Dict[str, Any]] = []
            elif state.get("stage_c", {}).get("status") == "skipped_incomplete":
                print("[stage_d] skipped: top-3 selection skipped due to incomplete sweep.")
                top_configs = []
            else:
                top = top or stage_c(
                    grid,
                    state,
                    int(args.top_k),
                    args.dry_run,
                    bool(args.allow_incomplete_selection),
                    int(args.simulate_failures),
                )
                top_ids = [row["config_id"] for row in top]
                top_configs = configs_by_ids(grid, top_ids)
            if not args.dry_run and top_configs and not ensure_stage_a2_complete_for_fid(
                top_configs, state, dirs, "stage_d"
            ):
                write_summary(grid, state, output_dir, final=False)
                save_state(state_path, state)
                return 2
            stage_fid(
                stage_key="stage_d",
                stage_dir=dirs["stage_d"],
                scheduler_stage_dir=dirs["stage_a2"],
                configs=top_configs,
                eval_samples=50000,
                state=state,
                state_path=state_path,
                python=args.python,
                dry_run=args.dry_run,
            )
        elif stage == "stage_e":
            if args.dry_run:
                print(f"[dry-run] would write {output_dir / 'summary.csv'} and {output_dir / 'summary.md'}")
            else:
                write_summary(grid, state, output_dir, final=True)

        if not args.dry_run:
            if stage != "stage_e":
                write_summary(grid, state, output_dir, final=False)
            save_state(state_path, state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
