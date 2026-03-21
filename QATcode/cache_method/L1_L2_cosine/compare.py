"""
詳細比較各個 block 的 L1rel 和 L1rel Rate (L1 change) 的差異
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from typing import Dict, List, Tuple
import json

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class L1Comparison:
    def __init__(self, npz_dir: str, output_dir: str = None):
        """
        Args:
            npz_dir: .npz 檔案所在目錄
            output_dir: 輸出目錄（比較報告和圖表）
        """
        self.npz_dir = Path(npz_dir)
        if output_dir is None:
            self.output_dir = self.npz_dir.parent / "comparison_results"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}

    def _safe_corrcoef(self, a: np.ndarray, b: np.ndarray) -> float:
        """安全計算皮爾森相關係數，避免常數序列/NaN。"""
        a = a.flatten()
        b = b.flatten()
        if a.size == 0 or b.size == 0:
            return float('nan')
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return float('nan')
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        """計算 cosine similarity，回傳 [0,1] 之間的近似度。"""
        a = a.flatten().astype(np.float64)
        b = b.flatten().astype(np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
        return float(np.dot(a, b) / denom)
        
    def load_all_blocks(self) -> Dict[str, np.ndarray]:
        """載入所有 block 的資料"""
        npz_files = sorted(list(self.npz_dir.glob("*.npz")))
        print(f"找到 {len(npz_files)} 個 .npz 檔案")
        
        data_dict = {}
        for npz_file in npz_files:
            block_name = npz_file.stem
            data = np.load(npz_file)
            data_dict[block_name] = {
                'l1rel': data['l1rel'],
                'l1rel_rate': data['l1rel_rate'],
                'l1_step_mean': data['l1_step_mean'],
                'l1_step_std': data['l1_step_std'],
                'l1_rate_step_mean': data['l1_rate_step_mean'],
                'l1_rate_step_std': data['l1_rate_step_std'],
            }
        return data_dict
    
    def compare_block(self, block_name: str, data: Dict) -> Dict:
        """比較單個 block 的 L1rel 和 L1rel Rate"""
        l1rel = data['l1rel']
        l1rel_rate = data['l1rel_rate']
        l1_step_mean = data['l1_step_mean']
        l1_rate_step_mean = data['l1_rate_step_mean']
        
        # 矩陣層面的比較
        matrix_diff = np.abs(l1rel - l1rel_rate)
        # 避免除以零，使用 mask
        mask = l1rel > 1e-8
        matrix_rel_diff = np.zeros_like(matrix_diff)
        matrix_rel_diff[mask] = (matrix_diff[mask] / l1rel[mask]) * 100
        
        # Step-wise 層面的比較
        step_diff = np.abs(l1_step_mean - l1_rate_step_mean)
        step_rel_diff = (step_diff / (l1_step_mean + 1e-8)) * 100
        
        # 統計信息
        stats = {
            'block_name': block_name,
            # 矩陣統計
            'matrix_abs_diff_mean': float(matrix_diff.mean()),
            'matrix_abs_diff_max': float(matrix_diff.max()),
            'matrix_abs_diff_min': float(matrix_diff.min()),
            'matrix_abs_diff_std': float(matrix_diff.std()),
            'matrix_rel_diff_mean': float(matrix_rel_diff.mean()),
            'matrix_rel_diff_max': float(matrix_rel_diff.max()),
            'matrix_rel_diff_std': float(matrix_rel_diff.std()),
            # Step-wise 統計
            'step_abs_diff_mean': float(step_diff.mean()),
            'step_abs_diff_max': float(step_diff.max()),
            'step_abs_diff_min': float(step_diff.min()),
            'step_abs_diff_std': float(step_diff.std()),
            'step_rel_diff_mean': float(step_rel_diff.mean()),
            'step_rel_diff_max': float(step_rel_diff.max()),
            'step_rel_diff_std': float(step_rel_diff.std()),
            # 原始數值統計
            'l1rel_mean': float(l1rel.mean()),
            'l1rel_rate_mean': float(l1rel_rate.mean()),
            'l1_step_mean_avg': float(l1_step_mean.mean()),
            'l1_rate_step_mean_avg': float(l1_rate_step_mean.mean()),
        }
        
        return stats
    
    def plot_block_comparison(self, block_name: str, data: Dict, stats: Dict):
        """為單個 block 生成比較圖"""
        l1_step_mean = data['l1_step_mean']
        l1_step_std = data['l1_step_std']
        l1_rate_step_mean = data['l1_rate_step_mean']
        l1_rate_step_std = data['l1_rate_step_std']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        x = np.arange(len(l1_step_mean))
        
        # 上排左：L1rel
        axes[0, 0].plot(x, l1_step_mean, 'b-', linewidth=2, label='L1rel')
        axes[0, 0].fill_between(x, l1_step_mean - l1_step_std, 
                               l1_step_mean + l1_step_std, alpha=0.2, color='blue')
        axes[0, 0].set_title(f'{block_name} - L1rel (Symmetric)', fontsize=12)
        axes[0, 0].set_xlabel('Denoising steps')
        axes[0, 0].set_ylabel('Relative metric')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 上排右：L1rel Rate
        axes[0, 1].plot(x, l1_rate_step_mean, 'r-', linewidth=2, label='L1rel Rate')
        axes[0, 1].fill_between(x, l1_rate_step_mean - l1_rate_step_std,
                               l1_rate_step_mean + l1_rate_step_std, alpha=0.2, color='red')
        axes[0, 1].set_title(f'{block_name} - L1rel Rate (Asymmetric)', fontsize=12)
        axes[0, 1].set_xlabel('Denoising steps')
        axes[0, 1].set_ylabel('Relative metric')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 下排左：疊加比較
        axes[1, 0].plot(x, l1_step_mean, 'b-', linewidth=2.5, alpha=0.8, label='L1rel')
        axes[1, 0].plot(x, l1_rate_step_mean, 'r--', linewidth=2.5, alpha=0.8, label='L1rel Rate')
        axes[1, 0].set_title(f'{block_name} - Overlay Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Denoising steps')
        axes[1, 0].set_ylabel('Relative metric')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 下排右：差異圖
        step_diff = np.abs(l1_step_mean - l1_rate_step_mean)
        axes[1, 1].plot(x, step_diff, 'g-', linewidth=2, label='Absolute Difference')
        axes[1, 1].axhline(y=step_diff.mean(), color='orange', linestyle='--', 
                           linewidth=1.5, label=f'Mean: {step_diff.mean():.6f}')
        axes[1, 1].set_title(f'{block_name} - Absolute Difference', fontsize=12)
        axes[1, 1].set_xlabel('Denoising steps')
        axes[1, 1].set_ylabel('Absolute difference')
        axes[1, 1].legend(fontsize=10, loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加統計信息（放在左下角，避免與圖例重疊）
        stats_text = (f'Mean Diff: {stats["step_abs_diff_mean"]:.6f}\n'
                     f'Max Diff: {stats["step_abs_diff_max"]:.6f}\n'
                     f'Mean Rel Diff: {stats["step_rel_diff_mean"]:.2f}%\n'
                     f'Max Rel Diff: {stats["step_rel_diff_max"]:.2f}%')
        axes[1, 1].text(0.02, 0.02, stats_text, transform=axes[1, 1].transAxes,
                       verticalalignment='bottom', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{block_name}_comparison.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_summary_comparison(self, all_stats: List[Dict]):
        """生成所有 block 的摘要比較圖"""
        df = pd.DataFrame(all_stats)
        
        # 圖1: 各 block 的平均差異比較
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 上排左：Step-wise 平均絕對差異
        block_names = df['block_name'].values
        step_diff_means = df['step_abs_diff_mean'].values
        step_diff_maxs = df['step_abs_diff_max'].values
        
        x_pos = np.arange(len(block_names))
        axes[0, 0].bar(x_pos, step_diff_means, alpha=0.7, color='skyblue', label='Mean Diff')
        axes[0, 0].plot(x_pos, step_diff_maxs, 'ro-', markersize=4, label='Max Diff')
        axes[0, 0].set_title('Step-wise Absolute Difference by Block', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Block')
        axes[0, 0].set_ylabel('Absolute difference')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(block_names, rotation=45, ha='right', fontsize=8)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 上排右：Step-wise 平均相對差異
        step_rel_diff_means = df['step_rel_diff_mean'].values
        step_rel_diff_maxs = df['step_rel_diff_max'].values
        
        axes[0, 1].bar(x_pos, step_rel_diff_means, alpha=0.7, color='lightcoral', label='Mean Rel Diff')
        axes[0, 1].plot(x_pos, step_rel_diff_maxs, 'ro-', markersize=4, label='Max Rel Diff')
        axes[0, 1].set_title('Step-wise Relative Difference (%) by Block', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Block')
        axes[0, 1].set_ylabel('Relative difference (%)')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(block_names, rotation=45, ha='right', fontsize=8)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 下排左：矩陣層面的平均差異
        matrix_diff_means = df['matrix_abs_diff_mean'].values
        matrix_diff_maxs = df['matrix_abs_diff_max'].values
        
        axes[1, 0].bar(x_pos, matrix_diff_means, alpha=0.7, color='lightgreen', label='Mean Diff')
        axes[1, 0].plot(x_pos, matrix_diff_maxs, 'ro-', markersize=4, label='Max Diff')
        axes[1, 0].set_title('Matrix-level Absolute Difference by Block', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Block')
        axes[1, 0].set_ylabel('Absolute difference')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(block_names, rotation=45, ha='right', fontsize=8)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 下排右：L1rel vs L1rel Rate 的平均值比較
        l1rel_means = df['l1rel_mean'].values
        l1rel_rate_means = df['l1rel_rate_mean'].values
        
        width = 0.35
        x1 = x_pos - width/2
        x2 = x_pos + width/2
        axes[1, 1].bar(x1, l1rel_means, width, alpha=0.7, label='L1rel', color='blue')
        axes[1, 1].bar(x2, l1rel_rate_means, width, alpha=0.7, label='L1rel Rate', color='red')
        axes[1, 1].set_title('L1rel vs L1rel Rate Mean by Block', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Block')
        axes[1, 1].set_ylabel('Mean value')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(block_names, rotation=45, ha='right', fontsize=8)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "summary_comparison.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_report(self, all_stats: List[Dict]):
        """生成詳細的比較報告"""
        df = pd.DataFrame(all_stats)
        
        # 保存 CSV
        csv_path = self.output_dir / "comparison_stats.csv"
        df.to_csv(csv_path, index=False)
        
        # 生成文字報告
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("L1rel vs L1rel Rate 詳細比較報告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("整體統計摘要:\n")
            f.write("-" * 80 + "\n")
            f.write(f"總共比較的 Block 數量: {len(df)}\n")
            f.write(f"\nStep-wise 差異統計:\n")
            f.write(f"  平均絕對差異: {df['step_abs_diff_mean'].mean():.8f} ± {df['step_abs_diff_mean'].std():.8f}\n")
            f.write(f"  最大絕對差異: {df['step_abs_diff_max'].max():.8f} (Block: {df.loc[df['step_abs_diff_max'].idxmax(), 'block_name']})\n")
            f.write(f"  平均相對差異: {df['step_rel_diff_mean'].mean():.4f}% ± {df['step_rel_diff_mean'].std():.4f}%\n")
            f.write(f"  最大相對差異: {df['step_rel_diff_max'].max():.4f}% (Block: {df.loc[df['step_rel_diff_max'].idxmax(), 'block_name']})\n")
            
            f.write(f"\n矩陣層面差異統計:\n")
            f.write(f"  平均絕對差異: {df['matrix_abs_diff_mean'].mean():.8f} ± {df['matrix_abs_diff_mean'].std():.8f}\n")
            f.write(f"  最大絕對差異: {df['matrix_abs_diff_max'].max():.8f} (Block: {df.loc[df['matrix_abs_diff_max'].idxmax(), 'block_name']})\n")
            f.write(f"  平均相對差異: {df['matrix_rel_diff_mean'].mean():.4f}% ± {df['matrix_rel_diff_mean'].std():.4f}%\n")
            f.write(f"  最大相對差異: {df['matrix_rel_diff_max'].max():.4f}% (Block: {df.loc[df['matrix_rel_diff_max'].idxmax(), 'block_name']})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("各 Block 詳細統計:\n")
            f.write("=" * 80 + "\n\n")
            
            # 按差異大小排序
            df_sorted = df.sort_values('step_abs_diff_mean', ascending=False)
            
            for idx, row in df_sorted.iterrows():
                f.write(f"Block: {row['block_name']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Step-wise 差異:\n")
                f.write(f"    平均絕對差異: {row['step_abs_diff_mean']:.8f}\n")
                f.write(f"    最大絕對差異: {row['step_abs_diff_max']:.8f}\n")
                f.write(f"    平均相對差異: {row['step_rel_diff_mean']:.4f}%\n")
                f.write(f"    最大相對差異: {row['step_rel_diff_max']:.4f}%\n")
                f.write(f"  矩陣層面差異:\n")
                f.write(f"    平均絕對差異: {row['matrix_abs_diff_mean']:.8f}\n")
                f.write(f"    最大絕對差異: {row['matrix_abs_diff_max']:.8f}\n")
                f.write(f"    平均相對差異: {row['matrix_rel_diff_mean']:.4f}%\n")
                f.write(f"    最大相對差異: {row['matrix_rel_diff_max']:.4f}%\n")
                f.write(f"  原始數值:\n")
                f.write(f"    L1rel 平均值: {row['l1rel_mean']:.8f}\n")
                f.write(f"    L1rel Rate 平均值: {row['l1rel_rate_mean']:.8f}\n")
                f.write(f"    L1rel Step 平均: {row['l1_step_mean_avg']:.8f}\n")
                f.write(f"    L1rel Rate Step 平均: {row['l1_rate_step_mean_avg']:.8f}\n")
                f.write("\n")
        
        # 保存 JSON 格式（方便程式讀取）
        json_path = self.output_dir / "comparison_stats.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
        return csv_path, report_path, json_path

    def check_redundancy(self, data_dict: Dict[str, Dict], all_stats: List[Dict]):
        """
        檢查 L1rel 與 L1rel Rate 是否冗餘：
        - Step-wise 相關係數接近 1
        - 矩陣 cosine similarity 接近 1
        - 平均相對差異足夠小
        """
        stats_map = {s["block_name"]: s for s in all_stats}
        redundancy_rows = []

        # 你可以調整這些判斷門檻
        corr_threshold = 0.999
        cos_threshold = 0.999
        rel_diff_threshold = 0.5  # %

        for block_name, data in data_dict.items():
            l1_step_mean = data["l1_step_mean"]
            l1_rate_step_mean = data["l1_rate_step_mean"]
            l1rel = data["l1rel"]
            l1rel_rate = data["l1rel_rate"]

            step_corr = self._safe_corrcoef(l1_step_mean, l1_rate_step_mean)
            matrix_cos = self._cosine_similarity(l1rel, l1rel_rate)
            step_rel_diff_mean = stats_map[block_name]["step_rel_diff_mean"]

            is_redundant = (
                (not np.isnan(step_corr)) and
                step_corr >= corr_threshold and
                matrix_cos >= cos_threshold and
                step_rel_diff_mean <= rel_diff_threshold
            )

            redundancy_rows.append({
                "block_name": block_name,
                "step_corr": step_corr,
                "matrix_cosine": matrix_cos,
                "step_rel_diff_mean": step_rel_diff_mean,
                "is_redundant": bool(is_redundant),
            })

        df = pd.DataFrame(redundancy_rows).sort_values(
            ["is_redundant", "step_rel_diff_mean"],
            ascending=[False, True]
        )

        txt_path = self.output_dir / "redundancy_check.txt"
        json_path = self.output_dir / "redundancy_check.json"
        csv_path = self.output_dir / "redundancy_check.csv"

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(redundancy_rows, f, indent=2, ensure_ascii=False)

        redundant_blocks = df[df["is_redundant"]]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("L1rel vs L1rel Rate 冗餘檢查\n")
            f.write("=" * 80 + "\n")
            f.write(f"判斷門檻: step_corr >= {corr_threshold}, matrix_cosine >= {cos_threshold}, "
                    f"step_rel_diff_mean <= {rel_diff_threshold}%\n\n")
            f.write(f"總共 block: {len(df)}\n")
            f.write(f"冗餘 block 數量: {len(redundant_blocks)}\n\n")
            f.write("冗餘 block 列表:\n")
            f.write("-" * 80 + "\n")
            if len(redundant_blocks) == 0:
                f.write("無\n")
            else:
                for _, row in redundant_blocks.iterrows():
                    f.write(
                        f"{row['block_name']}: step_corr={row['step_corr']:.6f}, "
                        f"matrix_cosine={row['matrix_cosine']:.6f}, "
                        f"step_rel_diff_mean={row['step_rel_diff_mean']:.4f}%\n"
                    )
            f.write("\n全部 block 詳細值已輸出到 CSV/JSON。\n")

        print("\n冗餘檢查完成！")
        print(f"   - 冗餘檢查報告: {txt_path}")
        print(f"   - 冗餘檢查 CSV: {csv_path}")
        print(f"   - 冗餘檢查 JSON: {json_path}")
    
    def run(self):
        """執行完整的比較流程"""
        print("開始載入資料...")
        data_dict = self.load_all_blocks()
        
        print(f"\n開始比較 {len(data_dict)} 個 block...")
        all_stats = []
        
        for block_name, data in data_dict.items():
            print(f"  處理 {block_name}...")
            stats = self.compare_block(block_name, data)
            all_stats.append(stats)
            
            # 生成單個 block 的比較圖
            self.plot_block_comparison(block_name, data, stats)
        
        print("\n生成摘要比較圖...")
        self.plot_summary_comparison(all_stats)
        
        print("生成詳細報告...")
        csv_path, report_path, json_path = self.generate_report(all_stats)

        # 冗餘檢查
        self.check_redundancy(data_dict, all_stats)
        
        print(f"\n✅ 比較完成！")
        print(f"   - 詳細報告: {report_path}")
        print(f"   - CSV 統計: {csv_path}")
        print(f"   - JSON 統計: {json_path}")
        print(f"   - 摘要圖表: {self.output_dir / 'summary_comparison.png'}")
        print(f"   - 各 Block 比較圖: {self.output_dir}/*_comparison.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="比較 L1rel 和 L1rel Rate 的差異")
    parser.add_argument("--npz_dir", type=str, 
                       default="QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz",
                       help=".npz 檔案所在目錄")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="輸出目錄（預設為 npz_dir 的父目錄下的 comparison_results）")
    
    args = parser.parse_args()
    
    comparator = L1Comparison(args.npz_dir, args.output_dir)
    comparator.run()
