import torch
ckpt = torch.load('QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth', map_location='cpu')
newsd = {}
for k, v in ckpt.items():
    if 'scale_list' in k or 'zp_list' in k:
        # newv = v[::12][:20]
        newv = v[::5]
    
    else:
        newv = v
    
    newsd[k] = newv

torch.save(newsd, 'QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth')