import torch
ckpt = torch.load('QATcode/diffae_step6_lora_best_int.pth', map_location='cpu')
newsd = {}
for k, v in ckpt.items():
    if 'delta_list' in k or 'zp_list' in k:
        # newv = v[::12][:20]
        newv = v[::5]
    
    else:
        newv = v
    
    newsd[k] = newv

torch.save(newsd, 'QATcode/diffae_step6_lora_best_int_20steps.pth')