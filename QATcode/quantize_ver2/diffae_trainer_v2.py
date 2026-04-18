"""
Diff-AE + EfficientDM 整合訓練器
結合 Diff-AE 的 SpacedDiffusionBeatGans 和 EfficientDM 的知識蒸餾邏輯
實作完整的 Diff-AE inference 流程進行訓練
"""

from fvcore.nn import flop_count_table
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
import random
from copy import deepcopy
import copy
from contextlib import nullcontext
from itertools import combinations
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModel_DiffAE_LoRA
from diffusion.diffusion import *
from model.unet_autoenc import BeatGANsAutoencModel
from model.nn import timestep_embedding
from config import *

LOGGER = logging.getLogger("DiffAE_Trainer")

USE_AMP = True
AMP_DTYPE = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
LOGGER.info("使用自動混合精度: %s", amp_dtype)


class EMAHelper:
    """指數移動平均輔助類"""

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}
        self.backup = {}

    def register(self, module: nn.Module) -> None:
        """註冊模型參數用於 EMA"""
        if module is None:
            return
        self.shadow = {}
        self.backup = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        """更新 EMA 權重"""
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (1 - self.mu) * param.data + self.mu * self.shadow[name]

    def ema(self, module: nn.Module) -> None:
        """將 EMA 權重應用到模型，並備份原始權重"""
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()  # 備份原始權重
                param.data.copy_(self.shadow[name])

    def restore(self, module: nn.Module) -> None:
        """恢復原始權重"""
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])

    def state_dict(self) -> Dict[str, Any]:
        """供 checkpoint 儲存 EMA shadow。"""
        return {
            "mu": float(self.mu),
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """從 checkpoint 恢復 EMA shadow。"""
        if state is None:
            return
        self.mu = float(state.get("mu", self.mu))
        shadow = state.get("shadow", {})
        self.shadow = {}
        for k, v in shadow.items():
            self.shadow[k] = v.detach().clone() if torch.is_tensor(v) else v
        self.backup = {}

    def init_from_state_dict(
        self, model_state_dict: Dict[str, torch.Tensor], module: nn.Module
    ) -> None:
        """
        由完整 model state_dict 初始化 EMA shadow（只取 requires_grad 參數）。
        """
        self.shadow = {}
        self.backup = {}
        msd = model_state_dict or {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if name in msd and torch.is_tensor(msd[name]):
                self.shadow[name] = msd[name].detach().to(param.device, dtype=param.dtype).clone()
            else:
                self.shadow[name] = param.detach().clone()


def set_quant_step(model: nn.Module, k: int) -> None:
    """
    設定模型中所有激活量化器的 current_step 為指定值

    Args:
        model: 要設定的模型
        k: 要設定的時間步索引 (0-99 for DDIM 100 steps)
    """
    for name, module in model.named_modules():
        if hasattr(module, "act_quantizer") and hasattr(module.act_quantizer, "current_step"):
            module.act_quantizer.current_step = k


# 修復 DDIM 時間步生成邏輯
def make_ddim_timesteps(T_train: int = 1000, S: int = 100) -> list:
    """
    生成 DDIM 採樣的時間步序列，輸出如 [990, 980, ..., 10, 0]
    """
    step_size = T_train // S
    ddim_timesteps = [i for i in range(T_train - step_size, -1, -step_size)]
    return ddim_timesteps


class SpacedDiffusionBeatGans_Trainer(SpacedDiffusionBeatGans):
    """
    繼承 Diff-AE 的 SpacedDiffusionBeatGans，加入 EfficientDM 的知識蒸餾邏輯

    此類別結合了兩個框架的優勢：
    - Diff-AE: 自編碼器擴散模型架構、條件輸入 x_start
    - EfficientDM: 知識蒸餾訓練邏輯、浮點-量化模型對比
    - 在完整的 inference 流程中進行訓練
    """

    def __init__(
        self,
        base_sampler: SpacedDiffusionBeatGans,
        fp_model: BeatGANsAutoencModel,
        quant_model: QuantModel_DiffAE_LoRA,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        conds_mean: Optional[torch.Tensor] = None,
        conds_std: Optional[torch.Tensor] = None,
        conf: TrainConfig = None,
    ):
        """
        初始化整合訓練器

        Args:
            base_sampler: 基礎的 SpacedDiffusionBeatGans 採樣器
            fp_model: 浮點教師模型
            quant_model: 量化學生模型
            optimizer: 最佳化器
            lr_scheduler: 學習率調度器
            conds_mean: 潛在條件標準化均值
            conds_std: 潛在條件標準化標準差
        """
        # 複製基礎 sampler 的所有屬性
        self.__dict__.update(base_sampler.__dict__)
        self.fp_model = fp_model
        self.quant_model = quant_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.conds_mean = conds_mean
        self.conds_std = conds_std
        self.conf_data = conf
        # 訓練統計
        self.step_count = 0
        # 舊版（mean MSE）統計保留作為參考：
        self.total_distill_loss = 0.0
        # 新版（EfficientDM 對齊，sum MSE）統計：
        # self.total_distill_loss_eff = 0.0
        self.total_noise_loss = 0.0
        self.global_optimizer_step = 0
        self.loss_chunk_size = 20
        self._chunk_total_losses = []
        self._chunk_distill_losses = []
        self._completed_loss_chunks = []
        self._rollout_step_records = []
        # Optional diagnostics / A-B flags (default off = zero-diff behavior)
        self.teacher_autocast_match = False
        self.debug_scale_list_update = False
        self.debug_scale_list_interval = 0
        self._debug_scale_list_param_names = []
        self.debug_timestep_grad_conflict = False
        self.debug_timestep_grad_steps = [0, 80, 99]
        self.debug_timestep_grad_interval = 0
        # Step-2：rollout 結尾 tail-repair（預設關閉，不改主 loss 定義）
        self.tail_repair_enable = False
        self.tail_repair_steps = 1  # 外層重複次數（1 或 2）
        self.tail_repair_t_range: List[int] = [0]
        self.tail_repair_lr_scale = 0.25
        self._last_tail_repair_stats: Optional[Dict[str, Any]] = None
        self._tail_repair_xt_cache: Dict[int, torch.Tensor] = {}

        # 單步訓練相關參數
        self.lambda_distill = 0.8  # 知識蒸餾損失權重
        self.lambda_noise = 0.2  # 噪聲預測損失權重

        # DDIM 時間步序列
        self.ddim_timesteps = make_ddim_timesteps(T_train=1000, S=100)
        self.T_sampler = conf.make_T_sampler()

        # 初始化 EMA
        self.ema = EMAHelper(mu=0.999)
        self.ema.register(self.quant_model)

        LOGGER.info(f"DDIM 時間步序列長度: {len(self.ddim_timesteps)}")
        LOGGER.info("EMA 初始化完成")
        LOGGER.info("SpacedDiffusionBeatGans_Trainer 初始化完成")

    def set_loss_chunk_size(self, chunk_size: int) -> None:
        """設定 training-step chunk 大小（以 optimizer/global step 為單位）。"""
        if chunk_size <= 0:
            raise ValueError(f"loss_chunk_size must be > 0, got {chunk_size}")
        self.loss_chunk_size = int(chunk_size)
        self._chunk_total_losses = []
        self._chunk_distill_losses = []
        self._completed_loss_chunks = []

    def set_teacher_autocast_match(self, enabled: bool) -> None:
        """是否讓 teacher 分支也在與 student 相同 autocast 條件下前向。"""
        self.teacher_autocast_match = bool(enabled)

    def set_scale_list_update_debug(self, enabled: bool, interval: int = 1) -> None:
        """
        啟用/停用 scale_list 更新診斷（最小可移除 debug）。
        interval: 每幾個 optimizer step 記錄一次（<=0 視為關閉）。
        """
        self.debug_scale_list_update = bool(enabled)
        self.debug_scale_list_interval = int(interval)
        self._debug_scale_list_param_names = []

    def set_timestep_grad_conflict_debug(
        self,
        enabled: bool,
        steps: Optional[list] = None,
        interval: int = 0,
    ) -> None:
        """
        啟用/停用 timestep gradient conflict 診斷。
        - steps: 要診斷的 timestep 清單（例如 [0,80,99]）
        - interval: 每隔多少個 epoch 觸發一次（<=0 視為關閉）
        """
        self.debug_timestep_grad_conflict = bool(enabled)
        if steps is None:
            self.debug_timestep_grad_steps = [0, 80, 99]
        else:
            parsed = sorted({int(s) for s in steps})
            self.debug_timestep_grad_steps = parsed
        self.debug_timestep_grad_interval = int(interval)

    def set_tail_repair(
        self,
        enabled: bool,
        steps: int = 1,
        t_range: Optional[List[int]] = None,
        lr_scale: float = 0.25,
    ) -> None:
        """
        rollout 結尾額外小步更新（實驗用，預設關閉）。
        - steps: 外層重複次數，僅允許 1 或 2
        - t_range: 要補強的 timestep（如 [0] 或 [0,1,2,3,4]）
        - lr_scale: 在 tail-repair block 內暫時將各 param_group 的 lr 乘上此係數，結束後還原
        """
        self.tail_repair_enable = bool(enabled)
        if int(steps) not in (1, 2):
            raise ValueError(f"tail_repair_steps must be 1 or 2, got {steps}")
        self.tail_repair_steps = int(steps)
        if t_range is None or len(t_range) == 0:
            self.tail_repair_t_range = [0]
        else:
            self.tail_repair_t_range = sorted({int(x) for x in t_range})
        self.tail_repair_lr_scale = float(lr_scale)

    def _tail_repair_clip_grad_with_stats(self, max_norm: float = 0.5) -> Tuple[float, bool, float]:
        """clip_grad_norm_ 回傳值為 clip 前 total norm；另計算 clip 後 total norm。"""
        pre_norm = torch.nn.utils.clip_grad_norm_(self.quant_model.parameters(), max_norm=max_norm)
        pre_norm = float(pre_norm)
        hit_clip = bool(pre_norm > float(max_norm) + 1e-8)
        post_sq = 0.0
        for p in self.quant_model.parameters():
            if p.grad is not None:
                post_sq += float(p.grad.detach().float().pow(2).sum().item())
        post_norm = float(post_sq**0.5)
        return pre_norm, hit_clip, post_norm

    def _run_tail_repair_after_rollout(
        self,
        cond: torch.Tensor,
        batch_size: int,
        device: torch.device,
        epoch: int,
    ) -> None:
        """
        在完整 DDIM rollout 結束後執行：使用 rollout 內快取之真實 x_t（見 ddim_sample_with_training），
        與相同 cond，對尾段 timestep 做額外小步更新。
        使用暫時縮放 param_group lr（tail_repair_lr_scale），結束後還原；不呼叫 lr_scheduler.step()。
        """
        if not self.tail_repair_enable:
            self._last_tail_repair_stats = None
            return

        n_steps = int(self.num_timesteps)
        t_list = [t for t in self.tail_repair_t_range if 0 <= t < n_steps]
        if len(t_list) == 0:
            LOGGER.warning(
                "[TailRepair] no valid timesteps in t_range=%s (num_timesteps=%d)",
                self.tail_repair_t_range,
                n_steps,
            )
            self._last_tail_repair_stats = None
            return

        cache = getattr(self, "_tail_repair_xt_cache", {}) or {}
        missing = [t for t in t_list if int(t) not in cache]
        if len(missing) > 0:
            LOGGER.warning(
                "[TailRepair] missing cached x_t for timesteps %s (have keys %s); skip tail repair",
                missing,
                list(cache.keys()),
            )
            self._last_tail_repair_stats = None
            return

        LOGGER.info(
            "[TailRepair] start epoch=%d | outer_repeats=%d | t_range=%s | lr_scale(param_groups)=%.6f | source=rollout_xt_cache",
            epoch,
            self.tail_repair_steps,
            t_list,
            self.tail_repair_lr_scale,
        )

        was_training = self.quant_model.training
        self.quant_model.train()

        _saved_lrs = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]
        try:
            for g in self.optimizer.param_groups:
                g["lr"] = float(g["lr"]) * float(self.tail_repair_lr_scale)

            micro_step = 0
            loss_vals: List[float] = []

            for rep in range(self.tail_repair_steps):
                for t_step in t_list:
                    micro_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    t = torch.full((batch_size,), int(t_step), device=device, dtype=torch.long)
                    x_cached = cache[int(t_step)]
                    x_t = x_cached.clone().requires_grad_(True)
                    loss = self._compute_distill_loss_mean_for_debug(
                        x=x_t,
                        t=t,
                        model_kwargs={"cond": cond},
                    )
                    loss.backward()
                    pre_n, hit, post_n = self._tail_repair_clip_grad_with_stats(max_norm=0.5)
                    self.optimizer.step()
                    self.ema.update(self.quant_model)

                    lv = float(loss.detach().item())
                    loss_vals.append(lv)
                    LOGGER.info(
                        "[TailRepair] epoch=%d micro_step=%d outer=%d/%d t=%d loss=%.6f | "
                        "grad_pre_clip=%.6e hit_clip=%s grad_post_clip=%.6e (lr scaled x%.6f)",
                        epoch,
                        micro_step,
                        rep + 1,
                        self.tail_repair_steps,
                        int(t_step),
                        lv,
                        pre_n,
                        hit,
                        post_n,
                        float(self.tail_repair_lr_scale),
                    )
        finally:
            for g, old_lr in zip(self.optimizer.param_groups, _saved_lrs):
                g["lr"] = old_lr

        self.optimizer.zero_grad(set_to_none=True)
        if not was_training:
            self.quant_model.eval()

        n_updates = len(loss_vals)
        mean_loss = float(sum(loss_vals) / max(n_updates, 1))
        self._last_tail_repair_stats = {
            "n_updates": n_updates,
            "mean_loss": mean_loss,
            "t_range": list(t_list),
            "outer_repeats": int(self.tail_repair_steps),
            "lr_scale": float(self.tail_repair_lr_scale),
            "lr_mode": "param_group_scale",
            "input_source": "rollout_xt_cache",
        }
        LOGGER.info(
            "[TailRepair] epoch=%d summary | n_updates=%d mean_loss=%.6f",
            epoch,
            n_updates,
            mean_loss,
        )

    def _iter_scale_list_params(self):
        for name, param in self.quant_model.named_parameters():
            if "scale_list" in name:
                yield name, param

    def _ensure_debug_scale_targets(self) -> None:
        if self._debug_scale_list_param_names:
            return
        for name, _ in self._iter_scale_list_params():
            self._debug_scale_list_param_names.append(name)
            if len(self._debug_scale_list_param_names) >= 2:
                break
        if len(self._debug_scale_list_param_names) == 0:
            LOGGER.info("[ScaleListDebug] no scale_list parameter found in quant_model.")

    def _snapshot_debug_scale_values(self):
        snap = {}
        if not self._debug_scale_list_param_names:
            return snap
        named = dict(self.quant_model.named_parameters())
        for name in self._debug_scale_list_param_names:
            p = named.get(name, None)
            if p is None:
                continue
            snap[name] = p.detach().float().clone()
        return snap

    def _log_scale_list_grad_and_delta(self, pre_step_snapshot):
        if not self._debug_scale_list_param_names:
            return
        named = dict(self.quant_model.named_parameters())
        parts = []
        for name in self._debug_scale_list_param_names:
            p = named.get(name, None)
            if p is None:
                continue
            grad = p.grad
            grad_none = grad is None
            grad_norm = float("nan")
            if grad is not None:
                grad_norm = float(grad.detach().float().norm().item())
            delta_absmax = float("nan")
            before = pre_step_snapshot.get(name, None)
            if before is not None:
                after = p.detach().float()
                delta_absmax = float((after - before).abs().max().item())
            parts.append(
                f"{name}: grad_none={grad_none}, grad_norm={grad_norm:.3e}, delta_absmax={delta_absmax:.3e}"
            )
        if parts:
            LOGGER.info(
                "[ScaleListDebug] step=%d | %s",
                int(self.global_optimizer_step + 1),
                " | ".join(parts),
            )

    def _convert_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """將模型內部 timestep 轉到原始範圍（對齊主訓練路徑）。"""
        if hasattr(self, "timestep_map") and len(self.timestep_map) > 0:
            map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
            new_ts = map_tensor[t]
            if hasattr(self, "rescale_timesteps") and self.rescale_timesteps:
                new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            return new_ts
        return t

    def _collect_trainable_optimizer_params(self):
        """收集同時滿足 requires_grad 且存在於 optimizer param_groups 的參數。"""
        opt_param_ids = set()
        for group in self.optimizer.param_groups:
            for p in group.get("params", []):
                if p is not None:
                    opt_param_ids.add(id(p))

        selected = []
        for name, p in self.quant_model.named_parameters():
            if p.requires_grad and id(p) in opt_param_ids:
                selected.append((name, p))
        return selected

    def _pick_representative_grad_params(self):
        """
        選 1~2 個代表性 trainable 參數：
        - 前段偏 input_blocks
        - 後段偏 output_blocks
        並保證 requires_grad=True 且在 optimizer param_groups 中。
        """
        trainable = self._collect_trainable_optimizer_params()
        if len(trainable) == 0:
            return []

        def _is_lora_weight(n: str) -> bool:
            return ("lora" in n) and n.endswith(".weight")

        early_pool = [(n, p) for n, p in trainable if ("input_blocks" in n) and _is_lora_weight(n)]
        late_pool = [(n, p) for n, p in trainable if ("output_blocks" in n) and _is_lora_weight(n)]

        picked = []
        if len(early_pool) > 0:
            early_pool = sorted(early_pool, key=lambda x: x[0])
            picked.append(early_pool[0])
        if len(late_pool) > 0:
            late_pool = sorted(late_pool, key=lambda x: x[0])
            picked.append(late_pool[-1])

        if len(picked) == 0:
            all_lora = [(n, p) for n, p in trainable if _is_lora_weight(n)]
            if len(all_lora) > 0:
                all_lora = sorted(all_lora, key=lambda x: x[0])
                picked.append(all_lora[0])
                if len(all_lora) > 1:
                    picked.append(all_lora[-1])
            else:
                # 最終 fallback：trainable 首尾各一
                trainable = sorted(trainable, key=lambda x: x[0])
                picked.append(trainable[0])
                if len(trainable) > 1:
                    picked.append(trainable[-1])

        # 去重
        uniq = []
        seen = set()
        for n, p in picked:
            if n in seen:
                continue
            seen.add(n)
            uniq.append((n, p))
        return uniq[:2]

    def _compute_distill_loss_mean_for_debug(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """僅用於診斷：計算與正式訓練同定義的 mean MSE distill loss（不做 step）。"""
        converted_t = self._convert_timesteps(t)

        with torch.no_grad():
            teacher_ctx = (
                torch.cuda.amp.autocast(dtype=amp_dtype)
                if (self.teacher_autocast_match and torch.cuda.is_available())
                else nullcontext()
            )
            with teacher_ctx:
                fp_out = self.fp_model.forward(
                    x=x.detach(),
                    t=converted_t,
                    **model_kwargs,
                )
            fp_pred = fp_out.pred

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            quant_out = self.quant_model.forward(
                x=x,
                t=converted_t,
                **model_kwargs,
            )
            quant_pred = quant_out.pred

        quant_pred = quant_pred.float()
        fp_pred = fp_pred.float().detach()
        distill_loss = F.mse_loss(quant_pred, fp_pred, reduction="none")
        distill_loss = distill_loss.mean(dim=[1, 2, 3])
        return distill_loss.mean()

    def _run_timestep_grad_conflict_debug(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        device: torch.device,
        epoch: int,
    ) -> None:
        """Step-1 診斷：固定 batch/cond/noise source，對多個 timestep 構造合法 x_t 並比較梯度方向。"""
        if not self.debug_timestep_grad_conflict:
            return
        if self.debug_timestep_grad_interval <= 0:
            return
        if (epoch % self.debug_timestep_grad_interval) != 0:
            return

        steps = [
            int(s) for s in self.debug_timestep_grad_steps if 0 <= int(s) < int(self.num_timesteps)
        ]
        steps = sorted(list(set(steps)))
        if len(steps) < 2:
            LOGGER.info("[GradConflict] skipped: need at least 2 valid timesteps, got %s", steps)
            return

        reps = self._pick_representative_grad_params()
        if len(reps) == 0:
            LOGGER.info("[GradConflict] skipped: no trainable params in optimizer groups.")
            return

        rep_names = [n for n, _ in reps]
        LOGGER.info("[GradConflict] Epoch %d | selected params: %s", epoch, ", ".join(rep_names))

        was_training = self.quant_model.training
        self.quant_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            cond = self.generate_latent_condition(batch_size, device)
            # 固定同一個 batch/noise source，再按各 timestep 構造合法 x_t = q_sample(x_start, t, noise)
            x_start_ref = torch.tanh(torch.randn(*shape, device=device))
            noise_ref = torch.randn_like(x_start_ref)

        grad_map: Dict[int, Dict[str, torch.Tensor]] = {}
        norm_map: Dict[int, Dict[str, float]] = {}

        for t_step in steps:
            self.optimizer.zero_grad(set_to_none=True)
            t = torch.full((batch_size,), int(t_step), device=device, dtype=torch.long)
            x_t = self.q_sample(x_start_ref, t, noise=noise_ref).detach().requires_grad_(True)
            loss = self._compute_distill_loss_mean_for_debug(
                x=x_t,
                t=t,
                model_kwargs={"cond": cond},
            )
            loss.backward()

            grad_map[t_step] = {}
            norm_map[t_step] = {}
            for pname, p in reps:
                g = p.grad
                if g is None:
                    grad_map[t_step][pname] = None
                    norm_map[t_step][pname] = float("nan")
                else:
                    gv = g.detach().float().reshape(-1).clone()
                    grad_map[t_step][pname] = gv
                    norm_map[t_step][pname] = float(gv.norm().item())

        # Debug backward 與正式訓練完全隔離：診斷後立即清空 grad
        self.optimizer.zero_grad(set_to_none=True)
        if not was_training:
            self.quant_model.eval()

        # 輸出 grad norm
        for t_step in steps:
            parts = []
            for pname, _ in reps:
                parts.append(f"{pname}: norm={norm_map[t_step].get(pname, float('nan')):.6e}")
            LOGGER.info("[GradConflict] t=%d | %s", t_step, " | ".join(parts))

        # 輸出 pairwise cosine
        for pname, _ in reps:
            cos_parts = []
            for a, b in combinations(steps, 2):
                ga = grad_map[a].get(pname, None)
                gb = grad_map[b].get(pname, None)
                if (ga is None) or (gb is None):
                    cos_val = float("nan")
                else:
                    denom = float(ga.norm().item() * gb.norm().item())
                    if denom <= 1e-20:
                        cos_val = float("nan")
                    else:
                        cos_val = float(F.cosine_similarity(ga, gb, dim=0).item())
                cos_parts.append(f"cos(t{a},t{b})={cos_val:.6f}")
            LOGGER.info("[GradConflict] %s | %s", pname, " | ".join(cos_parts))

    def _record_training_step_loss(self, total_loss: float, distill_loss: Optional[float]) -> None:
        """記錄單一 optimizer step 的 loss，並在收滿 chunk 時產生 summary。"""
        self._chunk_total_losses.append(float(total_loss))
        if distill_loss is not None:
            self._chunk_distill_losses.append(float(distill_loss))

        if len(self._chunk_total_losses) < self.loss_chunk_size:
            return

        total_arr = np.asarray(self._chunk_total_losses, dtype=np.float64)
        summary = {
            "chunk_size": int(self.loss_chunk_size),
            "chunk_end_global_step": int(self.global_optimizer_step),
            "chunk_start_global_step": int(self.global_optimizer_step - self.loss_chunk_size + 1),
            "mean_total_loss": float(total_arr.mean()),
            "std_total_loss": float(total_arr.std(ddof=0)),
            "max_total_loss": float(total_arr.max()),
            "min_total_loss": float(total_arr.min()),
        }
        if len(self._chunk_distill_losses) == self.loss_chunk_size:
            distill_arr = np.asarray(self._chunk_distill_losses, dtype=np.float64)
            summary.update(
                {
                    "mean_distill_loss": float(distill_arr.mean()),
                    "std_distill_loss": float(distill_arr.std(ddof=0)),
                    "max_distill_loss": float(distill_arr.max()),
                    "min_distill_loss": float(distill_arr.min()),
                }
            )

        self._completed_loss_chunks.append(summary)
        self._chunk_total_losses = []
        self._chunk_distill_losses = []

    def pop_completed_loss_chunks(self) -> "Any":
        """取出並清空本輪新產生的 chunk summaries。"""
        chunks = self._completed_loss_chunks
        self._completed_loss_chunks = []
        return chunks

    def pop_rollout_step_records(self) -> "Any":
        """取出並清空本輪 rollout 的逐 DDIM step loss 記錄。"""
        records = self._rollout_step_records
        self._rollout_step_records = []
        return records

    def generate_latent_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成潛在條件，模擬 Diff-AE 的潛在空間採樣

        Args:
            batch_size: 批次大小
            device: 設備

        Returns:
            cond: 潛在條件張量
        """
        # 生成隨機潛在雜訊，使用正確的維度
        latent_noise = torch.randn(batch_size, 512, device=device)  # 使用固定的 512 維度
        latent_sampler = self.conf_data._make_latent_diffusion_conf(T=100).make_sampler()

        cond = latent_sampler.sample(
            model=self.fp_model.latent_net,
            noise=latent_noise,
            clip_denoised=self.conf_data.latent_clip_sample,
        )

        # 條件標準化（如果啟用）
        if self.conds_mean is not None and self.conds_std is not None:
            cond = cond * self.conds_std.to(device) + self.conds_mean.to(device)

        return cond

    def p_mean_variance_with_distillation(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Optional[callable] = None,
        model_kwargs: Optional[Dict] = None,
        is_training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        修改版的 p_mean_variance，支援知識蒸餾訓練

        Args:
            model: 模型（量化模型）
            x: 輸入張量
            t: 時間步
            clip_denoised: 是否裁剪去噪結果
            denoised_fn: 去噪函數
            model_kwargs: 模型參數
            is_training: 是否處於訓練模式

        Returns:
            包含預測結果的字典
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        # print(x.shape, t.shape)
        assert t.shape == (B,)

        # 手動處理時間步轉換，模擬 _WrappedModel 的行為
        def convert_timesteps(t: "Any") -> "Any":
            """將時間步轉換為原始範圍"""
            if hasattr(self, "timestep_map") and len(self.timestep_map) > 0:
                map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                new_ts = map_tensor[t]
                if hasattr(self, "rescale_timesteps") and self.rescale_timesteps:
                    new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
                return new_ts
            else:
                # 如果沒有 timestep_map，直接返回原始 t
                return t

        # 轉換時間步
        converted_t = convert_timesteps(t)

        # 浮點模型預測 (teacher)
        with torch.no_grad():
            fp_model_kwargs = model_kwargs.copy()
            teacher_ctx = (
                torch.cuda.amp.autocast(dtype=amp_dtype)
                if (self.teacher_autocast_match and torch.cuda.is_available())
                else nullcontext()
            )
            with teacher_ctx:
                fp_out = self.fp_model.forward(
                    x=x.detach(), t=converted_t, **fp_model_kwargs  # 使用轉換後的時間步
                )
            fp_pred = fp_out.pred

        # 量化模型預測 (student)
        if is_training:
            self.quant_model.train()
        else:
            self.quant_model.eval()

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            quant_out = model.forward(x=x, t=converted_t, **model_kwargs)  # 使用相同的轉換後時間步
            quant_pred = quant_out.pred

        quant_pred = quant_pred
        # 舊版（mean MSE）記錄：
        step_distill_loss = None
        running_mean_ddim_loss = None
        # 新版（sum MSE）記錄：
        # step_distill_loss_eff = None
        # running_mean_ddim_loss_eff = None
        # 計算知識蒸餾損失
        if is_training:
            # 舊版（mean MSE）保留註解供對照：
            distill_loss = F.mse_loss(quant_pred, fp_pred.detach(), reduction="none")
            distill_loss = distill_loss.mean(dim=[1, 2, 3])  # [batch_size]
            distill_loss_mean = distill_loss.mean()

            # 新版：對齊 EfficientDM，使用 sum MSE 作為 backward loss
            # distill_loss_eff = F.mse_loss(quant_pred, fp_pred.detach(), size_average=False)
            # distill_loss_eff.backward()

            # 舊版（mean MSE）反向傳播保留註解：
            distill_loss_mean.backward()

            do_scale_debug = (
                self.debug_scale_list_update
                and self.debug_scale_list_interval > 0
                and ((self.global_optimizer_step + 1) % self.debug_scale_list_interval == 0)
            )
            pre_step_snapshot = {}
            if do_scale_debug:
                self._ensure_debug_scale_targets()
                pre_step_snapshot = self._snapshot_debug_scale_values()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.quant_model.parameters(), max_norm=0.5)

            # 參數更新
            self.optimizer.step()
            if do_scale_debug:
                self._log_scale_list_grad_and_delta(pre_step_snapshot)
            self.lr_scheduler.step()
            self.global_optimizer_step += 1

            # 更新統計
            self.step_count += 1
            # 舊版（mean MSE）統計保留註解：
            step_distill_loss = float(distill_loss_mean.item())
            self.total_distill_loss += step_distill_loss
            running_mean_ddim_loss = self.total_distill_loss / max(self.step_count, 1)

            # step_distill_loss_eff = float(distill_loss_eff.item())
            # self.total_distill_loss_eff += step_distill_loss_eff
            # running_mean_ddim_loss_eff = self.total_distill_loss_eff / max(self.step_count, 1)
            # 目前此訓練路徑 total_loss 與 distill_loss 相同
            self._record_training_step_loss(
                total_loss=step_distill_loss,
                distill_loss=step_distill_loss,
                # total_loss=step_distill_loss_eff,
                # distill_loss=step_distill_loss_eff,
            )

        # 使用原本模型的預測進行後續計算
        model_output = fp_pred

        if self.model_var_type in [ModelVarType.fixed_large, ModelVarType.fixed_small]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.betas[1:], 1),
                    np.log(np.append(self.betas[1:], 1)),
                ),
                ModelVarType.fixed_small: (
                    np.append(self.betas[1:], 1),
                    np.log(np.append(self.betas[1:], 1)),
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # 裁剪預測的 x_start
        def process_xstart(x: "Any") -> "Any":
            """Public function process_xstart."""
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type in [
            ModelMeanType.eps,
        ]:
            if self.model_mean_type == ModelMeanType.eps:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == x.shape

        # 計算後驗方差
        model_variance = self.q_posterior_mean_variance(pred_xstart, x, t)[1]
        model_log_variance = torch.log(model_variance)

        return {
            "pred_xstart": pred_xstart,
            "pred_xstart_mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "fp_pred": fp_pred,
            "quant_pred": quant_pred,
            "converted_t": converted_t,  # 返回轉換後的時間步用於調試
            # 舊 key（mean 路徑）保留兼容，對應新版 eff 值：
            "current_step_loss": step_distill_loss,
            "running_mean_ddim_loss": running_mean_ddim_loss,
            # 新 key（明確標示 eff）：
            # "current_step_loss_eff": step_distill_loss_eff,
            # "running_mean_ddim_loss_eff": running_mean_ddim_loss_eff,
        }

    def ddim_sample_with_training(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        device: torch.device,
        clip_denoised: bool = True,
        denoised_fn: Optional[callable] = None,
        cond_fn: Optional[callable] = None,
        model_kwargs: Optional[Dict] = None,
        eta: float = 0.0,
        progress: bool = False,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        在 DDIM 採樣過程中進行知識蒸餾訓練

        Args:
            batch_size: 批次大小
            shape: 圖像形狀
            device: 設備
            clip_denoised: 是否裁剪去噪結果
            denoised_fn: 去噪函數
            cond_fn: 條件函數
            model_kwargs: 模型參數
            eta: DDIM eta 參數
            progress: 是否顯示進度

        Returns:
            生成的圖像
        """
        # 生成潛在條件
        cond = self.generate_latent_condition(batch_size, device)

        # 準備模型參數
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["cond"] = cond

        # 初始化雜訊圖像
        img = torch.randn(*shape, device=device)

        # 反向時間步
        indices = list(range(self.num_timesteps))[::-1]

        from tqdm.auto import tqdm

        indices = tqdm(indices)
        self._rollout_step_records = []
        if self.tail_repair_enable:
            self._tail_repair_xt_cache = {}

        # 逐步去噪，每一步都進行訓練
        for i in indices:
            t = torch.tensor([i] * batch_size, device=device)

            # 清零梯度
            self.optimizer.zero_grad()

            img = img.detach().requires_grad_(True)
            if self.tail_repair_enable and int(i) in self.tail_repair_t_range:
                self._tail_repair_xt_cache[int(i)] = img.detach().clone()

            # 使用修改版的 p_mean_variance 進行知識蒸餾
            out = self.p_mean_variance_with_distillation(
                model=self.quant_model,
                x=img,
                t=t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                is_training=True,  # 啟用訓練模式
            )
            if out.get("current_step_loss") is not None:
                self._rollout_step_records.append(
                    {
                        "global_step": int(self.global_optimizer_step),
                        "ddim_t": int(i),
                        "current_step_loss": float(out["current_step_loss"]),
                        "running_mean_ddim_loss": float(out["running_mean_ddim_loss"]),
                        # "current_step_loss_eff": float(out["current_step_loss_eff"]),
                        # "running_mean_ddim_loss_eff": float(out["running_mean_ddim_loss_eff"]),
                    }
                )

            # DDIM 更新公式
            pred_xstart = out["pred_xstart"]
            eps = self._predict_eps_from_xstart(img, t, pred_xstart)

            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, img.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, img.shape)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

            # 計算均值預測
            mean_pred = (
                pred_xstart * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
            )

            # 添加雜訊（除了最後一步）
            noise = torch.randn_like(img)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            img = mean_pred + nonzero_mask * sigma * noise

            # 更新 EMA 權重
            self.ema.update(self.quant_model)

            # 定期記錄
            if self.step_count % 20 == 0:
                LOGGER.info(
                    "Step %d - DDIM step %d: current_step_loss=%.6f | running_mean_ddim_loss=%.6f",
                    # "Step %d - DDIM step %d: current_step_loss_eff=%.6f | running_mean_ddim_loss_eff=%.6f",
                    self.step_count,
                    int(i),
                    float(out["current_step_loss"])
                    if out.get("current_step_loss") is not None
                    else float("nan"),
                    float(out["running_mean_ddim_loss"])
                    if out.get("running_mean_ddim_loss") is not None
                    else float("nan"),
                    # float(out["current_step_loss_eff"]) if out.get("current_step_loss_eff") is not None else float("nan"),
                    # float(out["running_mean_ddim_loss_eff"]) if out.get("running_mean_ddim_loss_eff") is not None else float("nan"),
                )

        # Step-2：rollout 結尾 tail-repair（與主 DDIM 步分離；預設關閉）
        if self.tail_repair_enable:
            if epoch is None:
                LOGGER.warning("[TailRepair] enabled but epoch is None; skipping tail repair")
                self._last_tail_repair_stats = None
            else:
                self._run_tail_repair_after_rollout(
                    cond=cond,
                    batch_size=batch_size,
                    device=device,
                    epoch=int(epoch),
                )
        else:
            self._last_tail_repair_stats = None

        return img

    def training_losses_with_inference_distillation(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        conf: Any,
        device: torch.device,
        distill_weight: float = 1.0,
        clip_grad_norm: float = 1.0,
        debug_epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        使用完整 inference 流程進行知識蒸餾訓練

        Args:
            batch_size: 批次大小
            shape: 圖像形狀
            conf: 配置對象
            device: 設備
            distill_weight: 知識蒸餾損失權重
            clip_grad_norm: 梯度裁剪閾值

        Returns:
            {loss: 總損失, distill_loss: 知識蒸餾損失, total_loss_mean: 平均損失, generated_images: 生成圖像}
        """
        # 執行完整的 inference 流程進行訓練
        generated_images = self.ddim_sample_with_training(
            batch_size=batch_size,
            shape=shape,
            device=device,
            clip_denoised=True,
            progress=False,
            epoch=debug_epoch,
        )

        # Step-1 診斷（預設關閉）：timestep gradient conflict
        # - 固定同一 batch/cond/noise source
        # - 對指定 timestep 各自構造合法 x_t 後比較梯度方向
        if debug_epoch is not None:
            self._run_timestep_grad_conflict_debug(
                batch_size=batch_size,
                shape=shape,
                device=device,
                epoch=int(debug_epoch),
            )

        # 計算平均損失
        avg_distill_loss = self.total_distill_loss / max(self.step_count, 1)
        # avg_distill_loss_eff = self.total_distill_loss_eff / max(self.step_count, 1)

        # 清理統計
        self.total_distill_loss = 0.0
        # self.total_distill_loss_eff = 0.0
        self.step_count = 0

        return {
            # 舊欄位名稱保留，但內容改為 eff loss：
            "loss": torch.tensor(avg_distill_loss, device=device),
            "distill_loss": torch.tensor(avg_distill_loss, device=device),
            "total_loss_mean": torch.tensor(avg_distill_loss, device=device),
            # 新欄位（明確）：
            #'distill_loss_eff': torch.tensor(avg_distill_loss_eff, device=device),
            "generated_images": generated_images,
            "loss_chunk_summaries": self.pop_completed_loss_chunks(),
            "rollout_step_records": self.pop_rollout_step_records(),
            "tail_repair": getattr(self, "_last_tail_repair_stats", None),
        }

    def training_step_ts(self, x_start: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        單步 Teacher-Student 訓練函數

        Args:
            batch: 來自 dataloader 的批次資料 (x_start)

        Returns:
            Dict: 包含損失值和統計信息的字典
        """
        device = x_start.device
        batch_size = x_start.size(0)
        # print('t_test', t_test//10 * 10)
        # 1. 生成潛在條件 (cond)
        with torch.no_grad():
            self.fp_model.eval()
            cond = self.generate_latent_condition(x_start.size(0), device)

        # 2. 從 DDIM 序列中隨機抽取時間步索引 k

        k = random.randint(0, len(self.ddim_timesteps) - 1)

        t_ddim = self.ddim_timesteps[k]

        t = torch.tensor([t_ddim] * batch_size, device=device)

        # 3. 前向擾動：q_sample(x_start, t)
        noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        # 4. Teacher 預測 (FP32, no_grad)
        with torch.no_grad():
            self.fp_model.eval()
            # 處理時間步轉換
            if hasattr(self, "rescale_timesteps") and self.rescale_timesteps:
                scaled_t = t.float() * (1000.0 / self.original_num_steps)
            else:
                scaled_t = t

            fp_out = self.fp_model.forward(x=x_t.detach(), t=scaled_t, cond=cond)
            eps_fp = fp_out.pred

        self.optimizer.zero_grad()
        # 5. Student 預測 (W8A8)

        # 設定量化步驟
        # with torch.cuda.amp.autocast(dtype=amp_dtype):
        set_quant_step(self.quant_model, k)
        quant_out = self.quant_model.forward(x=x_t.detach(), t=scaled_t, cond=cond)
        eps_q = quant_out.pred

        # 6. 計算損失 - 使用全精度
        eps_fp_detached = eps_fp.detach().float()  # 轉為全精度
        eps_q_float = eps_q.float()  # 轉為全精度
        noise_float = noise.float()  # 轉為全精度

        # 知識蒸餾損失 (全精度)
        distill_loss = F.mse_loss(eps_q_float, eps_fp_detached, reduction="mean")

        # 噪聲預測損失 (全精度)
        noise_loss = F.mse_loss(eps_q_float, noise_float, reduction="mean")

        # 總損失 (全精度)
        total_loss = self.lambda_distill * distill_loss + self.lambda_noise * noise_loss

        # 7. 反向傳播和參數更新

        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.quant_model.parameters(), max_norm=1.0)

        # 參數更新
        self.optimizer.step()
        self.lr_scheduler.step()

        # EMA 更新
        self.ema.update(self.quant_model)

        # 8. 更新統計
        self.step_count += 1

        # 返回統計信息
        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "noise_loss": noise_loss.item(),
            "timestep_k": k,
            "timestep_t": t_ddim,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def apply_ema_weights(self) -> None:
        """應用 EMA 權重到量化模型"""
        self.ema.ema(self.quant_model)

    def restore_original_weights(self) -> None:
        """恢復原始權重"""
        self.ema.restore(self.quant_model)

    def get_ema_model(self) -> QuantModel_DiffAE_LoRA:
        """獲取應用 EMA 權重的模型副本"""
        # 創建模型副本
        ema_model = deepcopy(self.quant_model)

        # 應用 EMA 權重
        applied_count = 0
        for name, param in ema_model.named_parameters():
            if param.requires_grad and name in self.ema.shadow:
                param.data.copy_(self.ema.shadow[name])
                applied_count += 1

        LOGGER.info(f"EMA 權重應用完成，共應用 {applied_count} 個參數")

        # 推論狀態（fake-quant on）
        if hasattr(ema_model, "set_quant_state"):
            ema_model.set_quant_state(True, True)
        ema_model.eval()

        return ema_model

    @torch.no_grad()
    def profile_unet_step(
        self,
        unet: BeatGANsAutoencModel,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        單步 UNet 前向的 MACs（不訓練、不蒸餾）。
        回傳: (macs_step:int, params:int)
        """
        from fvcore.nn import FlopCountAnalysis, flop_count_str

        device = x.device
        batch_size = 1
        cond = self.generate_latent_condition(batch_size, device)
        model_kwargs = {"cond": cond}
        t = torch.tensor([990] * batch_size, device=device)
        unet.eval()

        flops = FlopCountAnalysis(unet, (x, t, None, None, cond, None, None, None))
        LOGGER.info("Total FLOPs: %f", flops.total())
        LOGGER.info("flop_count_table(flops): \n%s", flop_count_table(flops))


class SpacedDiffusionBeatGans_Sampler(SpacedDiffusionBeatGans):
    """Public class SpacedDiffusionBeatGans_Sampler."""

    def __init__(
        self,
        base_sampler: SpacedDiffusionBeatGans,
        quant_model: QuantModel_DiffAE_LoRA,
        conds_mean: Optional[torch.Tensor] = None,
        conds_std: Optional[torch.Tensor] = None,
        conf: TrainConfig = None,
        ema_helper: Optional[EMAHelper] = None,
    ):
        self.__dict__.update(base_sampler.__dict__)
        self.base_sampler = base_sampler
        self.quant_model = quant_model
        self.conds_mean = conds_mean
        self.conds_std = conds_std
        self.conf_data = conf
        self.ema_helper = ema_helper  # 新增 EMA 輔助器
        self.latent_sampler = self.conf_data._make_latent_diffusion_conf(T=100).make_sampler()

    def sample(
        self,
        ema_model: BeatGANsAutoencModel,
        fp_model: BeatGANsAutoencModel,
        x_T: torch.Tensor,
        noise: torch.Tensor,
        cond: torch.Tensor,
        x_start: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Public function sample."""
        # 如果使用 EMA，臨時應用 EMA 權重

        latent_noise = torch.randn(len(x_T), 512, device=x_T.device)
        cond = self.latent_sampler.sample(
            model=fp_model.latent_net,
            noise=latent_noise,
            clip_denoised=self.conf_data.latent_clip_sample,
        )

        cond = cond * self.conds_std.to(x_T.device) + self.conds_mean.to(x_T.device)

        result = self.base_sampler.sample(model=ema_model, noise=x_T, cond=cond)

        return result


def create_diffae_trainer(
    base_sampler: "Any",
    fp_model: BeatGANsAutoencModel,
    quant_model: QuantModel_DiffAE_LoRA,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    conds_mean: Optional[torch.Tensor] = None,
    conds_std: Optional[torch.Tensor] = None,
    conf: TrainConfig = None,
) -> SpacedDiffusionBeatGans_Trainer:
    """
    工廠函數：創建 Diff-AE + EfficientDM 整合訓練器

    Args:
        base_sampler: 基礎的 SpacedDiffusionBeatGans 採樣器
        fp_model: 浮點教師模型
        quant_model: 量化學生模型
        optimizer: 最佳化器
        lr_scheduler: 學習率調度器
        conds_mean: 潛在條件標準化均值
        conds_std: 潛在條件標準化標準差

    Returns:
        SpacedDiffusionBeatGans_Trainer: 整合訓練器
    """
    trainer = SpacedDiffusionBeatGans_Trainer(
        base_sampler=base_sampler,
        fp_model=fp_model,
        quant_model=quant_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        conds_mean=conds_mean,
        conds_std=conds_std,
        conf=conf,
    )

    LOGGER.info("✅ Diff-AE + EfficientDM 整合訓練器創建成功")
    return trainer


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
