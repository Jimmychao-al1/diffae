"""
驗證 scheduler_config.json 的正確性。

索引語意：zones 與 recompute mask 使用 **analysis axis** 0..T-1（與 Stage 0 / Stage 1 圖一致）；
DDIM timestep **t_ddim = 99 - axis_idx**（T=100）。新 JSON 使用 axis_start/axis_end；舊檔可僅有 t_start/t_end。

檢查：
1. Zones 是否完整覆蓋 analysis axis 0..T-1
2. K 是否在 [k_min, k_max]
3. Recompute mask 的合理性
4. 統計資訊
"""

import json
import numpy as np
from pathlib import Path


def zone_axis_range(z: dict):
    """讀取 zone 的 analysis axis 範圍；支援新鍵 axis_* 或舊鍵 t_*。"""
    a0 = z.get("axis_start", z.get("t_start"))
    a1 = z.get("axis_end", z.get("t_end"))
    if a0 is None or a1 is None:
        raise KeyError(f"Zone 缺少 axis_start/axis_end（或舊版 t_start/t_end）: {z}")
    return int(a0), int(a1)


def load_config(config_path: str):
    """載入 config"""
    with open(config_path) as f:
        return json.load(f)


def check_zone_coverage(config):
    """檢查 zones 是否完整覆蓋 analysis axis 0..T-1（與 Stage 0 圖橫軸一致；DDIM: t_ddim=99-axis_idx）"""
    T = config['T']
    zones = config['zones']
    
    covered = set()
    for z in zones:
        a0, a1 = zone_axis_range(z)
        covered.update(range(a0, a1 + 1))
    
    expected = set(range(T))
    
    if covered == expected:
        print(f"✅ Zones 完整覆蓋 analysis axis 0..{T-1}")
        return True
    else:
        missing = expected - covered
        overlap = covered - expected
        if missing:
            print(f"❌ 缺少 axis indices: {sorted(missing)}")
        if overlap:
            print(f"❌ 多餘 axis indices: {sorted(overlap)}")
        return False


def check_k_range(config):
    """檢查 k 是否在合法範圍"""
    k_min = config['params']['k_min']
    k_max = config['params']['k_max']
    
    all_k = []
    for block in config['blocks']:
        all_k.extend(block['k_per_zone'])
    all_k = np.array(all_k)
    
    valid = (all_k >= k_min).all() and (all_k <= k_max).all()
    
    if valid:
        print(f"✅ 所有 k 在 [{k_min}, {k_max}] 範圍內")
        print(f"   實際範圍: [{all_k.min()}, {all_k.max()}]")
        return True
    else:
        out_of_range = all_k[(all_k < k_min) | (all_k > k_max)]
        print(f"❌ {len(out_of_range)} 個 k 超出範圍: {out_of_range}")
        return False


def build_recompute_mask(T, zones, k_per_zone):
    """從 zones + k 建立 recompute mask（索引為 analysis axis；DDIM 對照 t_ddim=99-idx）"""
    mask = np.zeros(T, dtype=bool)
    
    for zone, k in zip(zones, k_per_zone):
        a0, a1 = zone_axis_range(zone)
        t = a0
        while t <= a1:
            if t < T:
                mask[t] = True
            t += k
    
    return mask


def analyze_recompute_patterns(config):
    """分析 recompute patterns"""
    T = config['T']
    zones = config['zones']
    blocks = config['blocks']
    
    print(f"\n📊 Recompute 統計 (T={T}):")
    print(f"   Blocks: {len(blocks)}")
    print(f"   Zones: {len(zones)}")
    print()
    
    # Per-block recompute count
    recompute_counts = []
    for block in blocks:
        mask = build_recompute_mask(T, zones, block['k_per_zone'])
        count = mask.sum()
        recompute_counts.append(count)
    
    recompute_counts = np.array(recompute_counts)
    
    print(f"   Recompute count per block:")
    print(f"     Min: {recompute_counts.min()}/{T} ({recompute_counts.min()/T*100:.1f}%)")
    print(f"     Max: {recompute_counts.max()}/{T} ({recompute_counts.max()/T*100:.1f}%)")
    print(f"     Mean: {recompute_counts.mean():.1f}/{T} ({recompute_counts.mean()/T*100:.1f}%)")
    print(f"     Median: {np.median(recompute_counts):.0f}/{T} ({np.median(recompute_counts)/T*100:.1f}%)")
    print()
    
    # 平均 cache 節省
    avg_save = 1 - recompute_counts.mean() / T
    print(f"   平均 cache 節省: {avg_save*100:.1f}%（只需重算 {(1-avg_save)*100:.1f}%）")
    print()
    
    # Sample blocks
    print(f"   Sample recompute patterns (前 3 個 blocks):")
    for i in range(min(3, len(blocks))):
        block = blocks[i]
        mask = build_recompute_mask(T, zones, block['k_per_zone'])
        recompute_t = np.where(mask)[0]
        print(f"     Block {i} ({block['name']}):")
        print(f"       k_per_zone: {block['k_per_zone']}")
        print(f"       Recompute at: {recompute_t.tolist()[:20]}..." if len(recompute_t) > 20 else f"       Recompute at: {recompute_t.tolist()}")
        print(f"       Total: {mask.sum()}/{T} ({mask.sum()/T*100:.1f}%)")
    
    return True


def check_zone_start_recompute(config):
    """檢查每個 zone 的起點是否 recompute"""
    T = config['T']
    zones = config['zones']
    blocks = config['blocks']
    
    all_ok = True
    for block in blocks:
        mask = build_recompute_mask(T, zones, block['k_per_zone'])
        for z in zones:
            a0, _ = zone_axis_range(z)
            if not mask[a0]:
                print(f"❌ Block {block['id']} Zone {z['id']} 起點 axis={a0} 未 recompute")
                all_ok = False
    
    if all_ok:
        print(f"✅ 所有 zone 起點都會 recompute")
    
    return all_ok


def check_regularization(config):
    """檢查 regularization 是否生效"""
    regularize = config['params']['regularize']
    blocks = config['blocks']
    
    violations = []
    
    for block in blocks:
        k_list = block['k_per_zone']
        if len(k_list) < 2:
            continue
        
        for i in range(1, len(k_list)):
            diff = k_list[i] - k_list[i - 1]
            
            if regularize == "delta1":
                if abs(diff) > 1:
                    violations.append((block['id'], i, diff))
            elif regularize == "nondecreasing":
                if diff < 0:
                    violations.append((block['id'], i, diff))
    
    if violations:
        print(f"❌ Regularization ({regularize}) 違規 {len(violations)} 次:")
        for bid, zi, diff in violations[:5]:
            print(f"   Block {bid}, Zone {zi}: diff={diff}")
        return False
    else:
        print(f"✅ Regularization ({regularize}) 全部通過")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify scheduler config")
    parser.add_argument(
        "--config",
        type=str,
        default="QATcode/cache_method/Stage1/stage1_output/scheduler_config.json",
        help="Path to scheduler_config.json"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Scheduler Config 驗證")
    print("=" * 80)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config 檔案不存在: {config_path}")
        return
    
    print(f"\n載入: {config_path}")
    config = load_config(config_path)
    
    print(f"\nVersion: {config['version']}")
    print(f"T: {config['T']}")
    print(f"Params: {config['params']}")
    print()
    
    # 執行檢查
    print("=" * 80)
    print("開始驗證...")
    print("=" * 80)
    
    checks = [
        ("Zone coverage", check_zone_coverage),
        ("K range", check_k_range),
        ("Zone start recompute", check_zone_start_recompute),
        ("Regularization", check_regularization),
    ]
    
    results = []
    for name, func in checks:
        print(f"\n[{name}]")
        try:
            ok = func(config)
            results.append((name, ok))
        except Exception as e:
            print(f"❌ 檢查失敗: {e}")
            results.append((name, False))
    
    # Recompute analysis（不算作 pass/fail，只是資訊）
    print("\n[Recompute patterns]")
    analyze_recompute_patterns(config)
    
    # 總結
    print("\n" + "=" * 80)
    print("驗證總結")
    print("=" * 80)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    if passed == total:
        print(f"🎉 全部通過！({passed}/{total})")
    else:
        print(f"⚠️  部分失敗：{passed}/{total} 通過")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
