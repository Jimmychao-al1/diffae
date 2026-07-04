#!/usr/bin/env python3
"""Run transfer sanity checks for fixed FP Diff-AE and Q-LDM S3-Cache configs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Optional


DIFFAE_ROOT = Path("/home/jimmy/diffae")
LDM_ROOT = Path("/home/jimmy/latent-diffusion")
TFMQ_ROOT = Path("/home/jimmy/TFMQ-DM")
CACHE_ROOT = DIFFAE_ROOT / "QATcode/cache_method"
DEFAULT_OUTPUT_DIR = CACHE_ROOT / "resweep_output"

DIFFAE_PYTHON = "/home/jimmy/anaconda3/envs/diffae_bw/bin/python"
LDM_PYTHON = "/home/jimmy/anaconda3/envs/ldm/bin/python"

FP_EXPECTED = 9.441
QLDM_EXPECTED = 5.788
DEFAULT_TOLERANCE = 0.05

FP_SCHEDULER = (
    DIFFAE_ROOT
    / "QATcode/cache_method/Stage2/stage2_output_fp/K25_sw3_lam0.5/baseline/stage2_refined_scheduler_config.json"
)
FP_SCRIPT = DIFFAE_ROOT / "QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py"

QLDM_SCRIPT = LDM_ROOT / "ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py"
QLDM_SCHEDULER = (
    LDM_ROOT
    / "ldm_S3cache/cache_method/Stage2/stage2_output_qldm/sweep_K8_sw5_lam0.5/02_refined_blockwise/stage2_refined_scheduler_config.json"
)
QLDM_CALI_CKPT = TFMQ_ROOT / "cali_ckpt/ffhq256_w8a8.pth"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_fid(summary_path: Path) -> Optional[float]:
    if not summary_path.is_file():
        return None
    data = read_json(summary_path)
    if data.get("fid") is not None:
        return float(data["fid"])
    if data.get("fid_50k") is not None:
        return float(data["fid_50k"])
    if data.get("fid_5k") is not None and int(data.get("num_images", 0)) == 50000:
        return float(data["fid_5k"])
    return None


def write_command_header(log_path: Path, cmd: list[str], cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Started: {now()}\n")
        f.write(f"# CWD: {cwd}\n")
        f.write("# Command:\n")
        f.write(shlex.join(cmd) + "\n\n")


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> Dict[str, Any]:
    cmd_str = shlex.join(cmd)
    if dry_run:
        print(f"[dry-run] cwd={cwd}")
        print(f"[dry-run] {cmd_str}")
        print(f"[dry-run] log: {log_path}")
        return {"status": "dry_run", "command": cmd_str, "returncode": None}
    write_command_header(log_path, cmd, cwd)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
        return {"status": "success", "command": cmd_str, "returncode": 0}
    except subprocess.CalledProcessError as exc:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n# FAILED returncode={exc.returncode} at {now()}\n")
        return {"status": "failed", "command": cmd_str, "returncode": exc.returncode}


def build_fp_command(output_dir: Path, log_path: Path, python: str) -> tuple[list[str], Path]:
    run_dir = output_dir / "transfer_fp_diffae_k25_sw3_lam0.5"
    runs_index = output_dir / "transfer_fp_runs_index.jsonl"
    cmd = [
        python,
        "QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py",
        "--fp",
        "--num_steps",
        "100",
        "--eval_samples",
        "50000",
        "--seed",
        "0",
        "--use_cache_scheduler",
        "--cache_scheduler_json",
        str(FP_SCHEDULER.relative_to(DIFFAE_ROOT)),
        "--scheduler-name",
        "fp_K25_sw3_lam0.50_baseline",
        "--run-output-dir",
        str(run_dir),
        "--runs-index-path",
        str(runs_index),
        "--log_file",
        str(log_path),
    ]
    return cmd, run_dir / "summary.json"


def build_qldm_command(output_dir: Path, log_path: Path, python: str) -> tuple[list[str], Path]:
    run_dir = output_dir / "transfer_qldm_k8_sw5_lam0.5"
    runs_index = output_dir / "transfer_qldm_runs_index.jsonl"
    cmd = [
        python,
        "ldm_S3cache/cache_method/start_run/sample_stage2_cache_scheduler_ldm.py",
        "--mode",
        "cache",
        "--ckpt",
        "models/ldm/ffhq256/model.ckpt",
        "--config",
        "models/ldm/ffhq256/config.yaml",
        "--real_image_dir",
        "ffhq-dataset/images1024x1024",
        "--scheduler_json",
        str(QLDM_SCHEDULER.relative_to(LDM_ROOT)),
        "--scheduler_name",
        "K8_sw5_lam0.5",
        "--out_root",
        "outputs",
        "--results_json",
        "results/fid_results_qldm.json",
        "--n_samples",
        "50000",
        "--cali_ckpt",
        str(QLDM_CALI_CKPT),
        "--run-output-dir",
        str(run_dir),
        "--runs-index-path",
        str(runs_index),
        "--log_file",
        str(log_path),
    ]
    return cmd, run_dir / "summary.json"


def check_required_paths(diffae_python: str, ldm_python: str) -> bool:
    checks = [
        ("diffae_python", Path(diffae_python).is_file(), diffae_python),
        ("ldm_python", Path(ldm_python).is_file(), ldm_python),
        ("fp_script", FP_SCRIPT.is_file(), str(FP_SCRIPT)),
        ("fp_scheduler", FP_SCHEDULER.is_file(), str(FP_SCHEDULER)),
        ("qldm_script", QLDM_SCRIPT.is_file(), str(QLDM_SCRIPT)),
        ("qldm_scheduler", QLDM_SCHEDULER.is_file(), str(QLDM_SCHEDULER)),
        ("qldm_cali_ckpt", QLDM_CALI_CKPT.is_file(), str(QLDM_CALI_CKPT)),
    ]
    ok = True
    print("Pre-flight checks:")
    for name, passed, detail in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    return ok


def result_block(title: str, expected: float, measured: Optional[float], tolerance: float) -> str:
    if measured is None:
        delta = None
        passed = False
    else:
        delta = measured - expected
        passed = abs(delta) < tolerance
    lines = [
        f"## {title}",
        f"- Expected FID@50K: {expected:.3f}",
        f"- Measured FID@50K: {'' if measured is None else f'{measured:.3f}'}",
        f"- Delta: {'' if delta is None else f'{delta:.3f}'}",
        f"- Pass (|Delta| < {tolerance:.2f}): {'yes' if passed else 'no'}",
        "",
    ]
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    fp_measured: Optional[float],
    qldm_measured: Optional[float],
    tolerance: float,
) -> Path:
    fp_pass = fp_measured is not None and abs(fp_measured - FP_EXPECTED) < tolerance
    qldm_pass = qldm_measured is not None and abs(qldm_measured - QLDM_EXPECTED) < tolerance
    verdict = "Environment stable" if fp_pass and qldm_pass else "Environment drift detected"
    report = output_dir / "transfer_check.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Transfer Sanity Check\n\n")
        f.write(result_block("FP Diff-AE + S3-Cache (K25/sw3/lambda=0.5)", FP_EXPECTED, fp_measured, tolerance))
        f.write(result_block("Q-LDM + S3-Cache (K8/sw5/lambda=0.5)", QLDM_EXPECTED, qldm_measured, tolerance))
        f.write("## Verdict\n")
        f.write(verdict + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer sanity check runner")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--diffae-python", type=str, default=os.environ.get("DIFFAE_PYTHON", DIFFAE_PYTHON))
    parser.add_argument("--ldm-python", type=str, default=os.environ.get("LDM_PYTHON", LDM_PYTHON))
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    ok = check_required_paths(args.diffae_python, args.ldm_python)
    if not ok and not args.dry_run:
        return 2

    fp_log = output_dir / "transfer_fp_diffae_k25_sw3_lam0.5.log"
    qldm_log = output_dir / "transfer_qldm_k8_sw5_lam0.5.log"
    fp_cmd, fp_summary = build_fp_command(output_dir, fp_log, args.diffae_python)
    qldm_cmd, qldm_summary = build_qldm_command(output_dir, qldm_log, args.ldm_python)

    run_command(fp_cmd, DIFFAE_ROOT, fp_log, args.dry_run)
    run_command(qldm_cmd, LDM_ROOT, qldm_log, args.dry_run)

    if args.dry_run:
        print(f"[dry-run] would write {output_dir / 'transfer_check.md'}")
        return 0

    fp_measured = read_fid(fp_summary)
    qldm_measured = read_fid(qldm_summary)
    report = write_report(output_dir, fp_measured, qldm_measured, float(args.tolerance))
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
