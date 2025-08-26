# code based on Skimmed_CFG by Extraltodeus (https://github.com/Extraltodeus/Skimmed_CFG)
import torch

@torch.no_grad()
def get_skimming_mask(x_orig, cond, uncond, cond_scale, return_denoised=False, disable_flipping_filter=False):
    # Calculate the denoised tensor based on CFG formula
    denoised = x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))
    
    # Check if signs of (cond - uncond) match signs of cond
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    
    # Check if signs of cond match signs after CFG scaling
    matching_diff_after = cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    
    if disable_flipping_filter:
        # Combine masks without deviation filter
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        # Check if signs of denoised match signs of (denoised - x_orig)
        deviation_influence = (denoised.sign() == (denoised - x_orig).sign())
        # Combine all masks to determine influenced elements
        outer_influence = matching_pred_signs & matching_diff_after & deviation_influence
    
    if return_denoised:
        # Return mask and denoised tensor if requested
        return outer_influence, denoised
    else:
        # Return only the mask
        return outer_influence

@torch.no_grad()
def skimmed_CFG(x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False):
    # Get mask and denoised tensor from get_skimming_mask
    outer_influence, denoised = get_skimming_mask(x_orig, cond, uncond, cond_scale, True, disable_flipping_filter)
    
    # Calculate denoised tensor with skimming scale
    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond)))
    
    # Compute difference between original and skimmed denoised tensors
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    
    # Adjust cond tensor where mask is True, scaling by cond_scale
    cond[outer_influence] = cond[outer_influence] - (low_cfg_denoised_outer_difference[outer_influence] / cond_scale)
    
    # Return modified cond tensor
    return cond