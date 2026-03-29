"""
驗證新版 Stage-1：shared_zones + k_per_zone + expanded_mask（DDIM 步序 99→0）。

步序 i：i=0 為 t=99（第一步），i=T-1 為 t=0。
expanded_mask[b,i]==True 為 full compute (F)，False 為 reuse (R)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_zone_mask_ddim(
    t_start: int, t_end: int, k: int, T: int
) -> np.ndarray:
    """與 stage1_scheduler 一致：回傳 (T,) bool，索引為步序 i（i=0 -> t=99）。"""
    L = t_start - t_end + 1
    mask_step = np.zeros(T, dtype=bool)
    for p in range(L):
        t = t_start - p
        is_f = p % k == 0
        i = (T - 1) - t
        mask_step[i] = is_f
    return mask_step


def rebuild_mask(
    shared_zones: List[Dict[str, Any]], k_per_zone: List[int], T: int
) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for z, k in zip(shared_zones, k_per_zone):
        m |= expand_zone_mask_ddim(int(z["t_start"]), int(z["t_end"]), int(k), T)
    m[0] = True
    return m


def check_shared_zones_cover_ddim(shared_zones: List[Dict[str, Any]], T: int) -> bool:
    """每個 DDIM timestep t ∈ [0,T-1] 恰落在一個 zone [t_start,t_end]（t_start >= t_end）。"""
    covered = np.zeros(T, dtype=bool)
    for z in shared_zones:
        ts, te = int(z["t_start"]), int(z["t_end"])
        if ts < te:
            print(f"❌ zone id={z.get('id')} 需 t_start >= t_end，收到 t_start={ts}, t_end={te}")
            return False
        for t in range(te, ts + 1):
            if covered[t]:
                print(f"❌ DDIM t={t} 被多個 zone 覆蓋")
                return False
            covered[t] = True
    if not covered.all():
        print(f"❌ 未覆蓋的 DDIM t: {np.where(~covered)[0].tolist()}")
        return False
    print("✅ shared_zones 完整、不重疊覆蓋所有 DDIM timestep")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify Stage-1 baseline scheduler JSON")
    parser.add_argument(
        "--config",
        type=str,
        default="QATcode/cache_method/Stage1/stage1_output/scheduler_config.json",
    )
    args = parser.parse_args()

    path = Path(args.config)
    if not path.exists():
        print(f"❌ 找不到 {path}")
        return

    cfg = load_config(str(path))
    T = int(cfg["T"])
    shared_zones = cfg["shared_zones"]
    params = cfg.get("stage1_baseline_params", {})
    k_min = int(params.get("k_min", 1))
    k_max = int(params.get("k_max", 4))

    print("=" * 72)
    print("Stage-1 baseline 驗證")
    print("=" * 72)
    print(f"time_order: {cfg.get('time_order')}")
    print(f"T={T}, |shared_zones|={len(shared_zones)}, blocks={len(cfg['blocks'])}")

    all_ok = True
    all_ok &= check_shared_zones_cover_ddim(shared_zones, T)

    exp_all = np.array([b["expanded_mask"] for b in cfg["blocks"]], dtype=bool)
    if exp_all.shape[1] != T:
        print(f"❌ expanded_mask 寬度 {exp_all.shape[1]} != T={T}")
        all_ok = False

    if not exp_all[:, 0].all():
        print("❌ 所有 block 在步序 i=0（DDIM t=99）必須為 F（True）")
        all_ok = False
    else:
        print("✅ 所有 block：步序 i=0（t=99）為 F")

    zone_start_ok = True
    for z in shared_zones:
        ts = int(z["t_start"])
        i0 = (T - 1) - ts
        if not exp_all[:, i0].all():
            print(f"❌ zone id={z['id']} 起點 t={ts}（步序 i={i0}）須全為 F")
            zone_start_ok = False
            all_ok = False
    if zone_start_ok:
        print("✅ 每個 zone 在該區第一步（DDIM t=t_start）為 F")

    for block in cfg["blocks"]:
        bid = block["id"]
        kz = block["k_per_zone"]
        if len(kz) != len(shared_zones):
            print(f"❌ block {bid} k_per_zone 長度 {len(kz)} != {len(shared_zones)}")
            all_ok = False
            continue
        arr = np.array(kz, dtype=int)
        if (arr < k_min).any() or (arr > k_max).any():
            print(f"❌ block {bid} k 超出 [{k_min}, {k_max}]: {kz}")
            all_ok = False

    if all_ok:
        print(f"✅ 所有 k 在 [{k_min}, {k_max}]")

    for b, block in enumerate(cfg["blocks"]):
        recon = rebuild_mask(shared_zones, block["k_per_zone"], T)
        if not np.array_equal(exp_all[b], recon):
            diff = np.where(exp_all[b] != recon)[0]
            print(f"❌ block {b} expanded_mask 與重建不符，相異步序（前 30 個）: {diff[:30].tolist()}")
            all_ok = False
    if all_ok:
        print("✅ 每個 block 的 expanded_mask 與 shared_zones + k_per_zone 重建一致")

    nF = exp_all.sum(axis=1)
    nR = T - nF
    print("\n📊 每 block #F / #R:")
    for b in range(len(cfg["blocks"])):
        print(f"   block {b} ({cfg['blocks'][b].get('name', '')}): F={int(nF[b])}, R={int(nR[b])}")

    print("\n" + ("🎉 全部通過" if all_ok else "⚠️ 有項目失敗"))
    print("=" * 72)


if __name__ == "__main__":
    main()
