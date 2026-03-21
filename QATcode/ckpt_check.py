import torch
import logging
from pathlib import Path

# 設置 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('QATcode/ckpt_check/ckpt_check.log')
    ],
)
logger = logging.getLogger(__name__)

# 加載 checkpoint
ckpt_path = "QATcode/diffae_step6_lora_best.pth"
#ckpt_path = "QATcode/convert_int_ckpt/checkpoint_B.pth"
logger.info(f"=== 加載 checkpoint: {ckpt_path} ===")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 檢查頂層鍵
logger.info("=== Checkpoint 頂層鍵 ===")
logger.info(f"頂層鍵: {list(ckpt.keys())}")

# 如果是 state_dict（直接保存的模型）
if isinstance(ckpt, dict) and 'state_dict' not in ckpt:
    logger.info("\n=== 這是 state_dict（直接保存的模型）===")
    logger.info(f"總共 {len(ckpt)} 個鍵")
    
    # 記錄所有鍵
    logger.info("\n=== 所有鍵的完整列表 ===")
    for i, key in enumerate(ckpt.keys(), 1):
        logger.info(f"{i:5d}. {key}")
    
    logger.info("\n前 30 個鍵：")
    for i, key in enumerate(list(ckpt.keys())[:30]):
        logger.info(f"  {i+1}. {key}")
    
    # 統計不同類型的鍵
    encoder_keys = [key for key in ckpt.keys() if 'encoder' in key]
    ema_keys = [key for key in ckpt.keys() if 'ema_model' in key]
    lora_keys = [key for key in ckpt.keys() if 'lora' in key.lower()]
    intn_keys = [key for key in ckpt.keys() if 'intn_dequantizer' in key]
    
    logger.info(f"\n含有'encoder'的鍵的數量: {len(encoder_keys)}")
    if encoder_keys:
        logger.info("前 30 個含有'encoder'的鍵：")
        for i, key in enumerate(encoder_keys[:30]):
            logger.info(f"  {i+1}. {key}")
    
    logger.info(f"\n含有'ema_model'的鍵的數量: {len(ema_keys)}")
    if ema_keys:
        logger.info("前 30 個含有'ema_model'的鍵：")
        for i, key in enumerate(ema_keys[:30]):
            logger.info(f"  {i+1}. {key}")
    
    logger.info(f"\n含有'lora'的鍵的數量: {len(lora_keys)}")
    if lora_keys:
        logger.info("前 30 個含有'lora'的鍵：")
        for i, key in enumerate(lora_keys[:30]):
            logger.info(f"  {i+1}. {key}")
    
    logger.info(f"\n含有'intn_dequantizer'的鍵的數量: {len(intn_keys)}")
    if intn_keys:
        logger.info("前 30 個含有'intn_dequantizer'的鍵：")
        for i, key in enumerate(intn_keys[:30]):
            logger.info(f"  {i+1}. {key}")
    
    # 檢查每個 tensor 的形狀和 dtype
    logger.info("\n=== 前 30 個參數的詳細信息 ===")
    for i, (key, value) in enumerate(list(ckpt.items())[:30]):
        if isinstance(value, torch.Tensor):
            logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}, size={value.numel()}")
        else:
            logger.info(f"{key}: {type(value)}")

# 如果是完整 checkpoint（包含 state_dict, optimizer 等）
elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    logger.info("\n=== 這是完整 checkpoint ===")
    for key in ckpt.keys():
        if key == 'state_dict':
            logger.info(f"{key}: {len(ckpt[key])} 個參數")
            # 記錄 state_dict 中的所有鍵
            logger.info("\n=== state_dict 中的所有鍵 ===")
            for i, sd_key in enumerate(ckpt[key].keys(), 1):
                logger.info(f"{i:5d}. {sd_key}")
        else:
            logger.info(f"{key}: {type(ckpt[key])}")

logger.info("\n=== 檢查完成 ===")