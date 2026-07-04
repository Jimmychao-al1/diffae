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
FID_SCRIPT = REPO_ROOT / "QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py"
QAT_CKPT = REPO_ROOT / "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
BASE_CKPT = REPO_ROOT / "checkpoints/ffhq128_autoenc_latent/last.ckpt"
CALIBRATION = REPO_ROOT / "QATcode/quantize_ver2/calibration_diffae.pth"

PRESET_60 = {
    "K": [8, 12, 16, 20, 25],
    "sw": [2, 3, 5],
    "lambda": [0.25, 0.5, 1.0, 2.0],
    "k_max": [4],
}

EST_STAGE_A_MIN_PER_CONFIG = 1.0
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
            "stage_b": {},
            "stage_c": {},
            "stage_d": {},
            "stage_e": {},
        }
    for key in ("stage_a", "stage_b", "stage_c", "stage_d", "stage_e"):
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
    if isinstance(protocol, dict):
        return
    state["protocol"] = protocol_metadata(python)


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
        if prev.get("status") == "success" and Path(prev.get("scheduler_config", "")).is_file():
            print(f"[stage_a] skip completed {cid}")
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
            state["stage_a"].get(cid, {}).get(
                "scheduler_config",
                str(scheduler_stage_dir / cid / "scheduler_config.json"),
            )
        )
        if not dry_run and not sched.is_file():
            state[stage_key][cid] = {
                "status": "failed",
                "error": f"missing scheduler_config for {cid}: {sched}",
            }
            save_state(state_path, state)
            print(f"[{stage_key}] missing scheduler for {cid}; continue")
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


def stage_b_success_count(grid: List[Dict[str, Any]], state: Dict[str, Any]) -> int:
    count = 0
    for cfg in grid:
        rec = state["stage_b"].get(cfg["id"], {})
        if rec.get("status") == "success" and read_fid(Path(rec.get("summary", "")), "fid_5k") is not None:
            count += 1
    return count


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
    failures = failed_or_missing_stage_b(grid, state)
    success_count = stage_b_success_count(grid, state)
    total_count = len(grid)
    sweep_complete = success_count == total_count
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q-DiffAE Resweep Summary\n\n")
        f.write("## Pre-Registration Protocol\n\n")
        f.write(f"- **Timestamp (committed)**: {protocol.get('timestamp_committed', state.get('created_at', ''))}\n")
        f.write("- **Grid**: 60 configs (5 x 3 x 4 x 1)\n")
        grid_meta = protocol.get("grid", PRESET_60)
        f.write(f"  - K = {grid_meta.get('K', PRESET_60['K'])}\n")
        f.write(f"  - sw = {grid_meta.get('sw', PRESET_60['sw'])}\n")
        f.write(f"  - lambda = {grid_meta.get('lambda', PRESET_60['lambda'])}\n")
        f.write(f"  - k_max = {grid_meta.get('k_max', PRESET_60['k_max'])}\n")
        f.write(f"- **Iteration order**: {protocol.get('iteration_order', '')}\n")
        seed_policy = protocol.get("seed_policy", {})
        f.write("- **Seed policy**:\n")
        f.write(f"  - Stage B (FID@5K): seed = {seed_policy.get('stage_b_fid5k', 0)}\n")
        f.write(f"  - Stage D (FID@50K): seed = {seed_policy.get('stage_d_fid50k', 0)}\n")
        sampling = protocol.get("sampling_config", {})
        f.write(
            "- **Sampling config**: "
            f"mode={sampling.get('mode', 'float')}, "
            f"num_steps={sampling.get('num_steps', 100)}, "
            f"quant-state={sampling.get('quant_state', 'tt')}\n"
        )
        f.write(f"- **Selection rule**: {protocol.get('selection_rule', '')}\n")
        checkpoints = protocol.get("checkpoints", {})
        f.write("- **Checkpoints**:\n")
        f.write(f"  - QAT ckpt: {checkpoints.get('qat_ckpt', QAT_CKPT)}\n")
        f.write(f"  - Base Diff-AE ckpt: {checkpoints.get('base_diffae_ckpt', BASE_CKPT)}\n")
        f.write(f"  - Calibration: {checkpoints.get('calibration', CALIBRATION)}\n")
        environment = protocol.get("environment", {})
        f.write(
            f"- **Environment**: {environment.get('python', DEFAULT_PYTHON)}, "
            f"CUDA: {environment.get('cuda_detail', '')}\n"
        )
        f.write(f"- **Reference stats**: {protocol.get('reference_stats', reference_cache_path())}\n\n")

        f.write("## Sweep Status\n\n")
        f.write(f"- **total**: {total_count} configs\n")
        f.write(f"- **success**: {success_count} configs\n")
        f.write(f"- **failed**: {total_count - success_count} configs\n")
        f.write(f"- **sweep complete**: {'Yes' if sweep_complete else 'No'}\n")
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
            f.write("| Config ID | Stage | Failure reason |\n")
            f.write("|-----------|-------|----------------|\n")
            for item in failures:
                f.write(f"| {item['config_id']} | {item['stage']} | {item['reason']} |\n")

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
        + n * EST_FID5K_MIN_PER_CONFIG
        + top_k * EST_FID50K_MIN_PER_CONFIG
    )
    print("Dry-run plan:")
    print(f"  grid: preset_60 ({n} configs)")
    print(f"  first config: {grid[0]['id']}")
    print(f"  last config: {grid[-1]['id']}")
    print(f"  stage A: build {n} Stage1 scheduler_config.json files")
    print(f"  stage B: run FID@5K for {n} configs")
    print(f"  stage C: rank by FID@5K and select top {top_k}")
    print(f"  stage D: run FID@50K for top {top_k}")
    print("  stage E: write summary.csv and summary.md")
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
        choices=["all", "stage_a", "stage_b", "stage_c", "stage_d", "stage_e"],
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
        help="Dry-run only: simulate N Stage B failures to preview incomplete-sweep guard.",
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

    stages = ["stage_a", "stage_b", "stage_c", "stage_d", "stage_e"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "stage_a":
            stage_a(grid, state, state_path, dirs, args.python, args.dry_run)
        elif stage == "stage_b":
            stage_fid(
                stage_key="stage_b",
                stage_dir=dirs["stage_b"],
                scheduler_stage_dir=dirs["stage_a"],
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
            stage_fid(
                stage_key="stage_d",
                stage_dir=dirs["stage_d"],
                scheduler_stage_dir=dirs["stage_a"],
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
