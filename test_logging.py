#!/usr/bin/env python
import sys
import logging

# 模擬原始程式的結構
class TestConfig:
    LOG_FILE = "QATcode/sample_lora_intmodel.log"
    
    @classmethod
    def setup_environment(cls):
        # 清除現有的 handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        
        # 重新配置 logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler(cls.LOG_FILE)],
            force=True
        )

# 在模組載入時就初始化 LOGGER（這是問題所在）
LOGGER = logging.getLogger("QuantTraining")

if __name__ == "__main__":
    # 解析參數
    if len(sys.argv) > 1:
        TestConfig.LOG_FILE = sys.argv[1]
    
    print(f"[INFO] Setting log file to: {TestConfig.LOG_FILE}")
    
    # 設置環境
    TestConfig.setup_environment()
    
    # 檢查 handlers
    print(f"[DEBUG] LOGGER handlers: {LOGGER.handlers}")
    print(f"[DEBUG] Root logger handlers: {logging.getLogger().handlers}")
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            print(f"[DEBUG] FileHandler is writing to: {handler.baseFilename}")
    
    # 測試寫入
    LOGGER.info("Test message from LOGGER")
    logging.info("Test message from root logger")
    
    print(f"[INFO] Log should be in: {TestConfig.LOG_FILE}")
