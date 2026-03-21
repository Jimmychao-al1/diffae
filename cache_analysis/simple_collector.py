"""
簡化版快取分析收集器
直接整合到現有的 sample_lora_intmodel.py 中使用
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SimpleBlockCollector:
    """簡化版的 Block 輸出收集器"""
    
    def __init__(self,
                 save_dir: str = "cache_analysis/simple_outputs",
                 max_batch_collect: int = 5,
                 cache_method: str = "Res",
                 max_timesteps: Optional[int] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_batch_collect = max_batch_collect
        self.cache_method = cache_method  # "Res" / "Att" / "Both"
        self.max_timesteps = max_timesteps  # e.g., 20 or 100
        self.collected_data = {}
        self.hooks = []
        self.current_timestep = None
        self._step_counter = -1
        self.enabled = False
        
        logger.info(f"SimpleBlockCollector 初始化，保存目錄: {self.save_dir}")
    
    def enable_collection(self, timestep: int):
        """啟用收集並設置當前時間步"""
        self.current_timestep = timestep
        self.enabled = True
    
    def disable_collection(self):
        """停用收集"""
        self.enabled = False
        self.current_timestep = None
    
    '''def register_hooks(self, model: nn.Module):
        """註冊收集鉤子"""
        from model.blocks import TimestepEmbedSequential
        
        hook_count = 0
        for name, module in model.named_modules():
            if isinstance(module, TimestepEmbedSequential):
                # 跳過 encoder 相關的 TimestepEmbedSequential
                if 'encoder' in name:
                    logger.info(f"跳過 encoder 鉤子: {name}")
                    continue
                
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                hook_count += 1
                logger.info(f"註冊鉤子: {name}")
        
        logger.info(f"總共註冊了 {hook_count} 個 TimestepEmbedSequential 鉤子 (已排除 encoder)")
        
        # 同時註冊模型的主 forward 鉤子來追蹤時間步
        main_hook = model.register_forward_hook(self._create_timestep_tracker())
        self.hooks.append(main_hook)
        logger.info("註冊主模型時間步追蹤鉤子")
    
    def _create_hook(self, block_name: str):
        """創建收集鉤子"""
        def hook_fn(module, input, output):
            # 總是嘗試收集，不依賴外部的 enable/disable
            # 從 forward 的 kwargs 中尋找時間步
            timestep = None
            
            # 方法1: 檢查輸入參數
            if hasattr(module, '_current_timestep'):
                timestep = module._current_timestep
            
            # 方法2: 從輸入中推斷（如果是 TimestepEmbedSequential 的直接調用）
            if timestep is None and len(input) >= 2:
                potential_t = input[1]
                if torch.is_tensor(potential_t) and potential_t.numel() <= 4:
                    timestep = potential_t[0].item() if potential_t.numel() > 1 else potential_t.item()
            
            # 方法3: 使用全局計數器（備用方案）
            if timestep is None:
                if not hasattr(self, '_step_counter'):
                    self._step_counter = 0
                timestep = self._step_counter
                self._step_counter = (self._step_counter + 1) % 100  # 假設最多100步
            
            # 收集數據
            if block_name not in self.collected_data:
                self.collected_data[block_name] = {}
            if timestep not in self.collected_data[block_name]:
                self.collected_data[block_name][timestep] = []
            
            output_cpu = output[:self.max_batch_collect].detach().cpu().clone()
            self.collected_data[block_name][timestep].append(output_cpu)
            
            logger.debug(f"收集: {block_name}, t={timestep}, samples={len(self.collected_data[block_name][timestep])}")
        
        return hook_fn'''
    def register_hooks(self, model: nn.Module):
        self._step_counter = -1
        self.current_step = None
        """註冊鉤子到模型"""
        from model.blocks import TimestepEmbedSequential, AttentionBlock

        method = (self.cache_method or "Res").lower()
        want_res = method in ("res", "both")
        want_att = method in ("att", "both")

        hook_count = 0
        for name, module in model.named_modules():
            if want_res and isinstance(module, TimestepEmbedSequential):
                # 跳過 encoder 相關的 TimestepEmbedSequential
                if 'encoder' in name:
                    logger.info(f"跳過 encoder 鉤子: {name}")
                    continue
                
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                hook_count += 1
                logger.info(f"註冊鉤子: {name}")
            
            if want_att and isinstance(module, AttentionBlock):
                # 跳過 encoder 相關的 AttentionBlock
                if 'encoder' in name:
                    logger.info(f"跳過 encoder 鉤子: {name}")
                    continue
                
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                hook_count += 1
                logger.info(f"註冊鉤子: {name}")
        
        logger.info(f"總共註冊了 {hook_count} 個 TimestepEmbedSequential (已排除 encoder)")
        
        # 同時註冊模型的主 forward 鉤子來追蹤時間步
        self.hooks.append(model.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True))
        logger.info("註冊主模型時間步追蹤鉤子")
        
    def _create_step_pre_hook(self):
        def pre_hook(module, args, kwargs):
            # 每次模型 forward 都自增一次（無論是否有 t）
            if self.max_timesteps is None:
                self.current_timestep = None
                return
                
            self._step_counter = (self._step_counter + 1) % int(self.max_timesteps)
            self.current_timestep = self._step_counter
            
            # 可選：如果有真實的 t，可以用來驗證或記錄
            t = None
            if kwargs and 't' in kwargs and torch.is_tensor(kwargs['t']):
                t = kwargs['t']
            elif len(args) >= 2 and torch.is_tensor(args[1]):
                t = args[1]
                
            if t is not None:
                real_t = t[0].item() if t.numel() > 1 else t.item()
                # 可以在這裡加 debug 輸出比較 real_t 和 self.current_timestep
        return pre_hook

    def _create_hook(self, block_name: str):
        def hook_fn(module, input, output):
            # 使用統一的 current_timestep
            if self.current_timestep is None:
                return
            t = int(self.current_timestep)
            if self.max_timesteps is not None and not (0 <= t < int(self.max_timesteps)):
                return

            d = self.collected_data.setdefault(block_name, {})
            d.setdefault(t, [])
            d[t].append(output[: self.max_batch_collect].detach().cpu().clone())
        return hook_fn

    def _create_timestep_tracker(self):
        """創建時間步追蹤鉤子"""
        def timestep_hook(module, input, output):
            # 嘗試從輸入中提取時間步
            if len(input) >= 2:
                t = input[1]  # 通常 t 是第二個參數
                if torch.is_tensor(t):
                    if t.numel() > 1:
                        timestep = t[0].item()
                    else:
                        timestep = t.item()
                    
                    # 啟用收集
                    self.enable_collection(timestep)
        
        return timestep_hook
    
    def remove_hooks(self):
        """移除所有鉤子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def calculate_and_save_l1rel_matrices(self):
        """計算並保存 L1rel 矩陣"""
        logger.info("開始計算 L1rel 矩陣...")
        
        for block_name, block_data in self.collected_data.items():
            logger.info(f"處理 block: {block_name}")
            
            # 獲取時間步
            timesteps = sorted(block_data.keys())
            T = len(timesteps)
            
            if T == 0:
                logger.warning(f"Block {block_name} 沒有收集到數據")
                continue
            
            # 計算平均 L1rel 矩陣
            l1rel_matrix = np.zeros((T, T))
            
            for i, t1 in enumerate(timesteps):
                for j, t2 in enumerate(timesteps):
                    if t1 in block_data and t2 in block_data:
                        # 計算所有樣本的平均 L1rel
                        l1rel_values = []
                        
                        min_samples = min(len(block_data[t1]), len(block_data[t2]))
                        for k in range(min_samples):
                            try:
                                tensor1 = block_data[t1][k]
                                tensor2 = block_data[t2][k]
                                l1rel = self._calculate_l1rel(tensor1, tensor2)
                                l1rel_values.append(l1rel)
                            except Exception as e:
                                logger.warning(f"計算 L1rel 失敗 ({t1}, {t2}, sample {k}): {e}")
                        
                        if l1rel_values:
                            l1rel_matrix[i, j] = np.mean(l1rel_values)
            
            # 保存為 CSV
            self._save_matrix_csv(l1rel_matrix, timesteps, block_name)
    
    def _calculate_l1rel(self, tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-6) -> float:
        """計算 L1 相對差異（對稱版本）"""
        diff = torch.abs(tensor1 - tensor2)
        ref_norm1 = torch.abs(tensor1)
        ref_norm2 = torch.abs(tensor2)
        
        l1_diff = torch.mean(diff)
        l1_ref = (torch.mean(ref_norm1) + torch.mean(ref_norm2)) / 2 + eps  # 對稱化
        
        return (l1_diff / l1_ref).item()
    
    def _save_matrix_csv(self, matrix: np.ndarray, timesteps: List[int], block_name: str):
        """保存矩陣為 CSV"""
        # 創建 DataFrame
        df = pd.DataFrame(
            matrix,
            index=[f"t_{t}" for t in timesteps],
            columns=[f"t_{t}" for t in timesteps]
        )
        
        # 保存文件
        csv_file = self.save_dir / f"{block_name.replace('.', '_')}_l1rel.csv"
        df.to_csv(csv_file)
        logger.info(f"L1rel 矩陣已保存: {csv_file}")
    
    def get_stats(self) -> Dict:
        """獲取收集統計"""
        stats = {
            'total_blocks': len(self.collected_data),
            'blocks': {}
        }
        
        for block_name, block_data in self.collected_data.items():
            timesteps = list(block_data.keys())
            sample_counts = [len(samples) for samples in block_data.values()]
            
            stats['blocks'][block_name] = {
                'timesteps': len(timesteps),
                'timestep_range': (min(timesteps), max(timesteps)) if timesteps else (0, 0),
                'avg_samples_per_timestep': np.mean(sample_counts) if sample_counts else 0
            }
        
        return stats


