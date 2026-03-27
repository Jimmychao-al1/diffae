"""
Diff-AE + EfficientDM 整合訓練器
結合 Diff-AE 的 SpacedDiffusionBeatGans 和 EfficientDM 的知識蒸餾邏輯
實作完整的 Diff-AE inference 流程進行訓練
"""

from fvcore.nn import flop_count_table
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging
import random
from copy import deepcopy
import copy
from QATcode.quant_model_lora import QuantModel_DiffAE_LoRA
from diffusion.diffusion import *
from model.unet_autoenc import BeatGANsAutoencModel
from model.nn import timestep_embedding
from config import *

LOGGER = logging.getLogger("DiffAE_Trainer")

USE_AMP = True
AMP_DTYPE = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
print(f"使用自動混合精度: {amp_dtype}")

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

    def init_from_state_dict(self, model_state_dict: Dict[str, torch.Tensor], module: nn.Module) -> None:
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
        if hasattr(module, 'act_quantizer') and hasattr(module.act_quantizer, 'current_step'):
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
    
    def __init__(self, 
                base_sampler: SpacedDiffusionBeatGans, 
                fp_model: BeatGANsAutoencModel, 
                quant_model: QuantModel_DiffAE_LoRA, 
                optimizer: torch.optim.Optimizer, 
                lr_scheduler: Any,
                conds_mean: Optional[torch.Tensor] = None,
                conds_std: Optional[torch.Tensor] = None,
                conf : TrainConfig = None):
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
        self.total_distill_loss = 0.0
        self.total_noise_loss = 0.0
        
        # 單步訓練相關參數
        self.lambda_distill = 0.8 # 知識蒸餾損失權重
        self.lambda_noise = 0.2    # 噪聲預測損失權重
        
        # DDIM 時間步序列
        self.ddim_timesteps = make_ddim_timesteps(T_train=1000, S=100)
        self.T_sampler = conf.make_T_sampler()
        
        # 初始化 EMA
        self.ema = EMAHelper(mu=0.999)
        self.ema.register(self.quant_model)
        
        LOGGER.info(f"DDIM 時間步序列長度: {len(self.ddim_timesteps)}")
        LOGGER.info("EMA 初始化完成")
        LOGGER.info("SpacedDiffusionBeatGans_Trainer 初始化完成")
    
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
    
    def p_mean_variance_with_distillation(self, 
                                         model: nn.Module,
                                         x: torch.Tensor,
                                         t: torch.Tensor,
                                         clip_denoised: bool = True,
                                         denoised_fn: Optional[callable] = None,
                                         model_kwargs: Optional[Dict] = None,
                                         is_training: bool = False) -> Dict[str, torch.Tensor]:
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
        #print(x.shape, t.shape)
        assert t.shape == (B,)
        
        # 手動處理時間步轉換，模擬 _WrappedModel 的行為
        def convert_timesteps(t):
            """將時間步轉換為原始範圍"""
            if hasattr(self, 'timestep_map') and len(self.timestep_map) > 0:
                map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                new_ts = map_tensor[t]
                if hasattr(self, 'rescale_timesteps') and self.rescale_timesteps:
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
            fp_out = self.fp_model.forward(
                x=x.detach(),
                t=converted_t,  # 使用轉換後的時間步
                **fp_model_kwargs
            )
            fp_pred = fp_out.pred
        
        # 量化模型預測 (student)
        if is_training:
            self.quant_model.train()
        else:
            self.quant_model.eval()
        
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            quant_out = model.forward(
                x=x,
                t=converted_t,  # 使用相同的轉換後時間步
                **model_kwargs
            )
            quant_pred = quant_out.pred
        
        quant_pred = quant_pred.float()
        # 計算知識蒸餾損失
        if is_training:
            distill_loss = F.mse_loss(quant_pred, fp_pred.detach(), reduction='none')
            distill_loss = distill_loss.mean(dim=[1, 2, 3])  # [batch_size]
            
            # 反向傳播
            distill_loss.mean().backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.quant_model.parameters(), max_norm=0.5)
            
            # 參數更新
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # 更新統計
            self.step_count += 1
            self.total_distill_loss += distill_loss.mean().item()
        
        # 使用原本模型的預測進行後續計算
        model_output = fp_pred
        

        if self.model_var_type in [
                ModelVarType.fixed_large, ModelVarType.fixed_small
        ]:
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
        def process_xstart(x):
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
                    self._predict_xstart_from_eps(x_t=x, t=t,
                                                  eps=model_output))
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape ==
                x.shape)
        
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
        }
    
    def ddim_sample_with_training(self,
                                 batch_size: int,
                                 shape: Tuple[int, ...],
                                 device: torch.device,
                                 clip_denoised: bool = True,
                                 denoised_fn: Optional[callable] = None,
                                 cond_fn: Optional[callable] = None,
                                 model_kwargs: Optional[Dict] = None,
                                 eta: float = 0.0,
                                 progress: bool = False) -> torch.Tensor:
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
        model_kwargs['cond'] = cond
        
        # 初始化雜訊圖像
        img = torch.randn(*shape, device=device)
        
        # 反向時間步
        indices = list(range(self.num_timesteps))[::-1]
        

        from tqdm.auto import tqdm
        indices = tqdm(indices)
        
        # 逐步去噪，每一步都進行訓練
        for i in indices:
            t = torch.tensor([i] * batch_size, device=device)
            
            # 清零梯度
            self.optimizer.zero_grad()

            img = img.detach().requires_grad_(True)
            
            # 使用修改版的 p_mean_variance 進行知識蒸餾
            out = self.p_mean_variance_with_distillation(
                model=self.quant_model,
                x=img,
                t=t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                is_training=True  # 啟用訓練模式
            )
            
            # DDIM 更新公式
            pred_xstart = out["pred_xstart"]
            eps = self._predict_eps_from_xstart(img, t, pred_xstart)
            
            
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, img.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, img.shape)
            sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * 
                     torch.sqrt(1 - alpha_bar / alpha_bar_prev))
            
            # 計算均值預測
            mean_pred = (pred_xstart * torch.sqrt(alpha_bar_prev) +
                         torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
            
            # 添加雜訊（除了最後一步）
            noise = torch.randn_like(img)
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(img.shape) - 1))))
            img = mean_pred + nonzero_mask * sigma * noise

            # 更新 EMA 權重
            self.ema.update(self.quant_model)
            
            # 定期記錄
            if self.step_count % 100 == 0:
                LOGGER.info(f"Step {self.step_count} - DDIM step {i}: "
                           f"Distill loss: {self.total_distill_loss / max(self.step_count, 1):.6f}")
        
        return img
    
    def training_losses_with_inference_distillation(self, 
                                                   batch_size: int,
                                                   shape: Tuple[int, ...],
                                                   conf: Any,
                                                   device: torch.device,
                                                   distill_weight: float = 1.0,
                                                   clip_grad_norm: float = 1.0) -> Dict[str, torch.Tensor]:
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
            progress=False
        )
        
        # 計算平均損失
        avg_distill_loss = self.total_distill_loss / max(self.step_count, 1)
        
        # 清理統計
        self.total_distill_loss = 0.0
        self.step_count = 0
        
        return {
            'loss': torch.tensor(avg_distill_loss, device=device),
            'distill_loss': torch.tensor(avg_distill_loss, device=device),
            'total_loss_mean': torch.tensor(avg_distill_loss, device=device),
            'generated_images': generated_images,
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
        #print('t_test', t_test//10 * 10)
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
            if hasattr(self, 'rescale_timesteps') and self.rescale_timesteps:
                scaled_t = t.float() * (1000.0 / self.original_num_steps)
            else:
                scaled_t = t
                
            fp_out = self.fp_model.forward(
                x=x_t.detach(),
                t=scaled_t,
                cond=cond
            )
            eps_fp = fp_out.pred

        self.optimizer.zero_grad()
        # 5. Student 預測 (W8A8)
        
        # 設定量化步驟
        #with torch.cuda.amp.autocast(dtype=amp_dtype):
        set_quant_step(self.quant_model, k)
        quant_out = self.quant_model.forward(
            x=x_t.detach(),
            t=scaled_t,
            cond=cond
        )
        eps_q = quant_out.pred
        
        # 6. 計算損失 - 使用全精度
        eps_fp_detached = eps_fp.detach().float()  # 轉為全精度
        eps_q_float = eps_q.float()  # 轉為全精度
        noise_float = noise.float()  # 轉為全精度

        # 知識蒸餾損失 (全精度)
        distill_loss = F.mse_loss(eps_q_float, eps_fp_detached, reduction='mean')

        # 噪聲預測損失 (全精度)
        noise_loss = F.mse_loss(eps_q_float, noise_float, reduction='mean')

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
            'total_loss': total_loss.item(),
            'distill_loss': distill_loss.item(),
            'noise_loss': noise_loss.item(),
            'timestep_k': k,
            'timestep_t': t_ddim,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
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
        
        # 同步 int 反量化器到 weight_quantizer（關鍵）
        #for name, module in ema_model.named_modules():
        #    if hasattr(module, 'intn_dequantizer') and hasattr(module, 'weight_quantizer'):
        #        with torch.no_grad():
        #            module.intn_dequantizer.delta.data.copy_(module.weight_quantizer.delta.detach().to(module.intn_dequantizer.delta.device))
        #            module.intn_dequantizer.zero_point.data.copy_(module.weight_quantizer.zero_point.detach().to(module.intn_dequantizer.zero_point.device))

        # 推論狀態（fake-quant on）
        if hasattr(ema_model, 'set_quant_state'):
            ema_model.set_quant_state(True, True)
        ema_model.eval()
        
        return ema_model

    @torch.no_grad()
    def profile_unet_step(self,
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
        model_kwargs = {'cond': cond}
        t = torch.tensor([990] * batch_size, device=device)
        unet.eval()

        flops = FlopCountAnalysis(unet, (x, t,None,None,cond,None,None,None))
        LOGGER.info("Total FLOPs: %f", flops.total())
        LOGGER.info("flop_count_table(flops): \n%s", flop_count_table(flops))


        


class SpacedDiffusionBeatGans_Sampler(SpacedDiffusionBeatGans):
    def __init__(self, base_sampler: SpacedDiffusionBeatGans, quant_model: QuantModel_DiffAE_LoRA, 
                 conds_mean: Optional[torch.Tensor] = None, conds_std: Optional[torch.Tensor] = None, 
                 conf: TrainConfig = None, ema_helper: Optional[EMAHelper] = None):
        self.__dict__.update(base_sampler.__dict__)
        self.base_sampler = base_sampler
        self.quant_model = quant_model
        self.conds_mean = conds_mean
        self.conds_std = conds_std
        self.conf_data = conf
        self.ema_helper = ema_helper  # 新增 EMA 輔助器
        self.latent_sampler = self.conf_data._make_latent_diffusion_conf(T=100).make_sampler()

    def sample(self, ema_model: BeatGANsAutoencModel, fp_model: BeatGANsAutoencModel, x_T: torch.Tensor, noise: torch.Tensor, 
               cond: torch.Tensor, x_start: torch.Tensor, model_kwargs: Optional[Dict] = None, 
               progress: bool = False) -> torch.Tensor:
        
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
    

def create_diffae_trainer(base_sampler, 
                         fp_model: BeatGANsAutoencModel, 
                         quant_model: QuantModel_DiffAE_LoRA,
                         optimizer: torch.optim.Optimizer, 
                         lr_scheduler: Any,
                         conds_mean: Optional[torch.Tensor] = None,
                         conds_std: Optional[torch.Tensor] = None,
                         conf : TrainConfig = None) -> SpacedDiffusionBeatGans_Trainer:
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
        conf=conf
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