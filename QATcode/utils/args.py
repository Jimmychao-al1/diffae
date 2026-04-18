"""Shared argparse helpers for QAT and cache CLI scripts."""

from argparse import ArgumentParser


def add_common_generation_args(parser: ArgumentParser) -> ArgumentParser:
    """Add shared generation/cache arguments used across CLI entry scripts."""
    parser.add_argument("--num_steps", "--n", type=int, default=100)
    parser.add_argument("--samples", "--s", type=int, default=5)
    parser.add_argument("--eval_samples", "--es", type=int, default=50000)
    parser.add_argument("--mode", "--m", type=str, default="float")
    parser.add_argument("--enable_cache", action="store_true", help="啟用 cache scheduler")
    parser.add_argument(
        "--cache_method",
        type=str,
        default="Res",
        choices=["Res", "Att"],
        help="Cache 方法：Res (TimestepEmbedSequential) 或 Att (AttentionBlock)",
    )
    parser.add_argument(
        "--cache_threshold", type=float, default=0.1, help="Cache scheduler 的 L1rel 閾值"
    )
    parser.add_argument("--enable_quantitative_analysis", action="store_true", help="啟用定量分析")
    parser.add_argument("--analysis_num_samples", type=int, default=10, help="生成時間測試的樣本數")
    parser.add_argument(
        "--log_file",
        "--lf",
        type=str,
        default=None,
        help="指定 log 檔案路徑",
    )
    return parser
