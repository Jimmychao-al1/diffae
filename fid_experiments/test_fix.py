#!/usr/bin/env python3
"""
快速測試修改是否正確
"""

import sys
from pathlib import Path

# 從 fid_experiments/ 執行，所以 ROOT 是上一層（DiffAE root）
ROOT = Path(__file__).resolve().parent.parent

def test_modifications():
    """驗證所有必要的修改"""
    errors = []
    
    # 檢查 experiment.py
    exp_py = (ROOT / 'experiment.py').read_text()
    
    # 1. eval_num_images 修改
    if 'conf.eval_num_images = 1' in exp_py:
        errors.append('❌ experiment.py 仍有 eval_num_images = 1 (應改為 EVAL_SAMPLES)')
    elif 'conf.eval_num_images = EVAL_SAMPLES' not in exp_py:
        errors.append('❌ experiment.py 缺少 eval_num_images = EVAL_SAMPLES')
    else:
        print('✅ experiment.py: eval_num_images 已正確修改為 EVAL_SAMPLES')
    
    # 2. resume_from_checkpoint 修改
    if 'resume_from_checkpoint=resume' in exp_py:
        errors.append('❌ experiment.py 仍使用 resume_from_checkpoint (應改為 ckpt_path)')
    elif 'ckpt_path=resume' not in exp_py:
        errors.append('❌ experiment.py 缺少 ckpt_path=resume')
    else:
        print('✅ experiment.py: resume_from_checkpoint 已正確改為 ckpt_path')
    
    # 3. EVAL_SAMPLES 全域變數
    if 'EVAL_SAMPLES = 50_000' not in exp_py:
        errors.append('❌ experiment.py 缺少 EVAL_SAMPLES 全域變數')
    else:
        print('✅ experiment.py: EVAL_SAMPLES 全域變數存在')
    
    # 檢查 run_ffhq128.py
    run_ffhq = (ROOT / 'run_ffhq128.py').read_text()
    if 'eval_samples=args.eval_samples' not in run_ffhq:
        errors.append('❌ run_ffhq128.py 未正確傳遞 eval_samples 參數')
    else:
        print('✅ run_ffhq128.py: 正確傳遞 eval_samples 參數')
    
    # 總結
    print()
    if errors:
        print('發現問題：')
        for e in errors:
            print(f'  {e}')
        return False
    else:
        print('🎉 所有修改都正確！')
        print()
        print('可以執行的測試命令：')
        print('  cd fid_experiments && python run_experiments.py --steps 20 --k 5 --dry-run')
        print('  python run_ffhq128.py --eval_samples 1000 --steps 20')
        return True

if __name__ == '__main__':
    success = test_modifications()
    sys.exit(0 if success else 1)
