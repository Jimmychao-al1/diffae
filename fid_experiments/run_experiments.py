#!/usr/bin/env python3
"""
FID 實驗批次腳本

功能：
1) 自動建立 experiment_results/（存放所有 FID 結果）
2) 執行 run_ffhq128.py，擷取最終 FID
3) 執行 QATcode/sample_lora_intmodel.py，擷取最終 FID
4) 可用不同 T、不同 k（k 表示千張，例如 5 => 5000）控制實驗

預設會跑：
- T: 20, 100
- k: 5, 50  （=> eval_samples: 5000, 50000）

範例：
    python run_experiments.py
    python run_experiments.py --steps 20 100 --k 5 50
    python run_experiments.py --steps 100 --k 5 --qat-mode int --enable-cache --cache-method Res --cache-threshold 0.03
    python run_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# 從 fid_experiments/ 執行，所以 ROOT 是上一層（DiffAE root）
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiment_results"
LOGS_DIR = RESULTS_DIR / "logs"
CSV_PATH = RESULTS_DIR / "fid_results.csv"
JSONL_PATH = RESULTS_DIR / "fid_results.jsonl"


FID_PATTERNS = [
    # 最常見格式：FID@50000 100 steps score: 11.09
    re.compile(
        r"FID@\s*(?P<eval>\d+)\s+(?P<steps>\d+)\s+steps\s+score:\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    # 格式：fid (0): 21.126529693603516 或 fid (rank): value
    re.compile(
        r"\bfid\s*\(\s*\d+\s*\)\s*:\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    # 備援格式：FID: 11.09 / fid = 11.09
    re.compile(
        r"\bFID\b\s*[:=]\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def parse_fid_from_text(text: str) -> Optional[float]:
    """從完整 log 文字抓最後一個 FID。"""
    last_fid: Optional[float] = None
    for line in text.splitlines():
        for p in FID_PATTERNS:
            m = p.search(line)
            if m is None:
                continue
            try:
                last_fid = float(m.group("fid"))
            except Exception:
                continue
    return last_fid


def run_and_capture(cmd: List[str], log_path: Path, cwd: Path) -> Tuple[int, str]:
    """
    執行命令，stdout/stderr 同步印出並寫入 log。
    回傳 (return_code, full_text)
    """
    print(f"\n[RUN] {' '.join(cmd)}")
    print(f"[LOG] {log_path}")
    print("-" * 80)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: List[str] = []
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
            chunks.append(line)
        proc.wait()

    print("-" * 80)
    print(f"[DONE] return_code={proc.returncode}")
    return proc.returncode, "".join(chunks)


def append_result(row: Dict[str, str]) -> None:
    """把結果 append 到 CSV + JSONL（保留歷史）。"""
    fieldnames = [
        "timestamp",
        "script",
        "mode",
        "steps",
        "k",
        "eval_samples",
        "fid",
        "status",
        "return_code",
        "log_path",
        "command",
    ]

    csv_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_result_key(
    script: str,
    mode: str,
    steps: int,
    k: int,
    eval_samples: int,
) -> Tuple[str, str, str, str, str]:
    """標準化實驗鍵，用來比對是否已完成。"""
    return (
        str(script),
        str(mode),
        str(int(steps)),
        str(int(k)),
        str(int(eval_samples)),
    )


def load_completed_keys_from_jsonl(path: Path) -> Set[Tuple[str, str, str, str, str]]:
    """
    讀取歷史 JSONL，回傳所有 status=ok 的實驗鍵。
    只把成功結果視為完成；failed/dry_run 不算完成，會重新執行。
    """
    done: Set[Tuple[str, str, str, str, str]] = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if str(row.get("status", "")).lower() != "ok":
                continue
            try:
                key = make_result_key(
                    script=str(row.get("script", "")),
                    mode=str(row.get("mode", "")),
                    steps=int(row.get("steps")),
                    k=int(row.get("k")),
                    eval_samples=int(row.get("eval_samples")),
                )
            except Exception:
                continue
            done.add(key)
    return done


def build_baseline_cmd(steps: int, eval_samples: int) -> List[str]:
    return [
        sys.executable,
        "run_ffhq128.py",
        "--steps",
        str(steps),
        "--eval_samples",
        str(eval_samples),
    ]


def build_qat_cmd(
    steps: int,
    eval_samples: int,
    mode: str,
    qat_internal_log: Path,
    enable_cache: bool,
    cache_method: str,
    cache_threshold: float,
) -> List[str]:
    cmd = [
        sys.executable,
        "QATcode/sample_lora_intmodel.py",
        "--mode",
        mode,
        "--num_steps",
        str(steps),
        "--eval_samples",
        str(eval_samples),
        "--log_file",
        str(qat_internal_log),
    ]
    if enable_cache:
        cmd.extend(
            [
                "--enable_cache",
                "--cache_method",
                cache_method,
                "--cache_threshold",
                str(cache_threshold),
            ]
        )
    return cmd


def run_one_experiment(
    script_name: str,
    mode: str,
    steps: int,
    eval_samples: int,
    k: int,
    cmd: List[str],
    run_log_path: Path,
    fallback_log_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, str]:
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return {
            "timestamp": timestamp,
            "script": script_name,
            "mode": mode,
            "steps": str(steps),
            "k": str(k),
            "eval_samples": str(eval_samples),
            "fid": "",
            "status": "dry_run",
            "return_code": "0",
            "log_path": str(run_log_path.relative_to(ROOT)),
            "command": " ".join(cmd),
        }

    return_code, text = run_and_capture(cmd, run_log_path, cwd=ROOT)
    fid = parse_fid_from_text(text)

    # sample_lora_intmodel.py 另外也會寫自己的 log，必要時補抓
    if fid is None and fallback_log_path is not None and fallback_log_path.exists():
        fallback_text = fallback_log_path.read_text(encoding="utf-8", errors="ignore")
        fid = parse_fid_from_text(fallback_text)

    status = "ok" if (return_code == 0 and fid is not None) else "failed"

    row = {
        "timestamp": timestamp,
        "script": script_name,
        "mode": mode,
        "steps": str(steps),
        "k": str(k),
        "eval_samples": str(eval_samples),
        "fid": "" if fid is None else f"{fid:.10f}",
        "status": status,
        "return_code": str(return_code),
        "log_path": str(run_log_path.relative_to(ROOT)),
        "command": " ".join(cmd),
    }
    append_result(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FID experiments for run_ffhq128.py and QATcode/sample_lora_intmodel.py"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[20, 100],
        help="T values, e.g. --steps 20 100",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 50],
        help="k in thousand images, e.g. --k 5 50 => eval_samples 5000,50000",
    )
    parser.add_argument(
        "--qat-mode",
        type=str,
        default="float",
        choices=["float", "int"],
        help="mode for QATcode/sample_lora_intmodel.py",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="skip run_ffhq128.py",
    )
    parser.add_argument(
        "--skip-qat",
        action="store_true",
        help="skip QATcode/sample_lora_intmodel.py",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="enable cache flags for sample_lora_intmodel.py",
    )
    parser.add_argument(
        "--cache-method",
        type=str,
        default="Res",
        choices=["Res", "Att"],
        help="cache method for sample_lora_intmodel.py",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.1,
        help="cache threshold for sample_lora_intmodel.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print commands; do not execute",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="skip experiments that already have status=ok in experiment_results/fid_results.jsonl (default: on)",
    )
    parser.add_argument(
        "--no-skip-completed",
        dest="skip_completed",
        action="store_false",
        help="disable skip-completed and run everything again",
    )
    args = parser.parse_args()

    ensure_dirs()
    completed_keys = load_completed_keys_from_jsonl(JSONL_PATH) if args.skip_completed else set()

    now_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 88)
    print("FID experiment runner")
    print(f"results_dir: {RESULTS_DIR}")
    print(f"steps: {args.steps}")
    print(f"k (thousand): {args.k}")
    print(f"skip_baseline={args.skip_baseline}, skip_qat={args.skip_qat}, qat_mode={args.qat_mode}")
    print(f"skip_completed={args.skip_completed}, completed_records={len(completed_keys)}")
    print("=" * 88)

    # 固定執行順序（你指定）：
    # 20/5k -> 20/50k -> 100/5k -> 100/50k
    # 同時每組只執行一次（去除重複參數）。
    desired_steps_order = [20, 100]
    desired_k_order = [5, 50]

    unique_steps = list(dict.fromkeys(int(s) for s in args.steps))
    unique_k = list(dict.fromkeys(int(x) for x in args.k))

    ordered_steps = [s for s in desired_steps_order if s in unique_steps] + [
        s for s in unique_steps if s not in desired_steps_order
    ]
    ordered_k = [k for k in desired_k_order if k in unique_k] + [
        k for k in unique_k if k not in desired_k_order
    ]

    experiment_plan: List[Tuple[int, int]] = []
    seen_pairs = set()
    for steps in ordered_steps:
        for k in ordered_k:
            pair = (steps, k)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            experiment_plan.append(pair)

    print("execution order (T, k):", experiment_plan)

    summary_rows: List[Dict[str, str]] = []
    for steps, k in experiment_plan:
            eval_samples = int(k) * 1000

            if not args.skip_baseline:
                baseline_key = make_result_key(
                    script="run_ffhq128.py",
                    mode="baseline",
                    steps=steps,
                    k=k,
                    eval_samples=eval_samples,
                )
                if baseline_key in completed_keys:
                    print(f"[SKIP] already completed: script=run_ffhq128.py, mode=baseline, T={steps}, k={k}k")
                    summary_rows.append(
                        {
                            "status": "skipped_completed",
                            "script": "run_ffhq128.py",
                            "mode": "baseline",
                            "steps": str(steps),
                            "k": str(k),
                            "eval_samples": str(eval_samples),
                            "fid": "",
                        }
                    )
                else:
                    log_path = LOGS_DIR / f"{now_tag}_baseline_T{steps}_k{k}.log"
                    cmd = build_baseline_cmd(steps=steps, eval_samples=eval_samples)
                    row = run_one_experiment(
                        script_name="run_ffhq128.py",
                        mode="baseline",
                        steps=steps,
                        eval_samples=eval_samples,
                        k=k,
                        cmd=cmd,
                        run_log_path=log_path,
                        fallback_log_path=None,
                        dry_run=args.dry_run,
                    )
                    summary_rows.append(row)
                    if row.get("status") == "ok":
                        completed_keys.add(baseline_key)

            if not args.skip_qat:
                qat_key = make_result_key(
                    script="QATcode/sample_lora_intmodel.py",
                    mode=args.qat_mode,
                    steps=steps,
                    k=k,
                    eval_samples=eval_samples,
                )
                if qat_key in completed_keys:
                    print(
                        f"[SKIP] already completed: script=QATcode/sample_lora_intmodel.py, "
                        f"mode={args.qat_mode}, T={steps}, k={k}k"
                    )
                    summary_rows.append(
                        {
                            "status": "skipped_completed",
                            "script": "QATcode/sample_lora_intmodel.py",
                            "mode": args.qat_mode,
                            "steps": str(steps),
                            "k": str(k),
                            "eval_samples": str(eval_samples),
                            "fid": "",
                        }
                    )
                else:
                    run_log_path = LOGS_DIR / f"{now_tag}_qat_{args.qat_mode}_T{steps}_k{k}.log"
                    qat_internal_log = LOGS_DIR / f"{now_tag}_qat_internal_{args.qat_mode}_T{steps}_k{k}.log"
                    cmd = build_qat_cmd(
                        steps=steps,
                        eval_samples=eval_samples,
                        mode=args.qat_mode,
                        qat_internal_log=qat_internal_log,
                        enable_cache=args.enable_cache,
                        cache_method=args.cache_method,
                        cache_threshold=args.cache_threshold,
                    )
                    row = run_one_experiment(
                        script_name="QATcode/sample_lora_intmodel.py",
                        mode=args.qat_mode,
                        steps=steps,
                        eval_samples=eval_samples,
                        k=k,
                        cmd=cmd,
                        run_log_path=run_log_path,
                        fallback_log_path=qat_internal_log,
                        dry_run=args.dry_run,
                    )
                    summary_rows.append(row)
                    if row.get("status") == "ok":
                        completed_keys.add(qat_key)

    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    for r in summary_rows:
        print(
            f"[{r['status']}] script={r['script']}, mode={r['mode']}, "
            f"T={r['steps']}, k={r['k']}k, eval={r['eval_samples']}, fid={r['fid'] or 'N/A'}"
        )

    if not args.dry_run:
        print("\nSaved:")
        print(f"- {CSV_PATH}")
        print(f"- {JSONL_PATH}")
        print(f"- {LOGS_DIR}")


if __name__ == "__main__":
    main()
