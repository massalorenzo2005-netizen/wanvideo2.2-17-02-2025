# file: cfg_utils.py
#
# This file contains the core processing logic for various CFG modification
# algorithms. Each function is self-contained and handles one specific method,
# ensuring the main sampler code remains clean and readable.

# code based on Skimmed_CFG by Extraltodeus (https://github.com/Extraltodeus/Skimmed_CFG)

import torch
import math
from torchvision.transforms.functional import gaussian_blur
from typing import Tuple, Union
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Helper Functions  ---

@torch.no_grad()
def get_skimming_mask(x_orig: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor,
                      cond_scale: float, return_denoised: bool = False,
                      disable_flipping_filter: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Calculates a mask to identify tensor elements with high guidance influence.

    This function identifies where the guidance might "overshoot" or cause
    sign flips, which often correspond to visual artifacts at high CFG scales.
    The logic is based on the original implementation by Extraltodeus.

    Args:
        x_orig (torch.Tensor): The original noisy tensor (latent).
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        cond_scale (float): The classifier-free guidance scale.
        return_denoised (bool): If True, also returns the denoised tensor.
        disable_flipping_filter (bool): If True, disables one of the mask criteria.

    Returns:
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: 
        The boolean mask of influenced elements. If return_denoised is True,
        it returns a tuple containing the mask and the denoised tensor.
    """
    # Standard CFG formula to calculate the denoised prediction.
    denoised = x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))
    
    # Criterion 1: Check if the guidance direction (cond - uncond) aligns with the conditional prediction.
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    
    # Criterion 2: Check if the sign of the conditional prediction is preserved after applying CFG.
    matching_diff_after = cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    
    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        # Criterion 3: Check for deviation by comparing the denoised result's sign
        # with the direction of the change (denoised - x_orig).
        deviation_influence = (denoised.sign() == (denoised - x_orig).sign())
        outer_influence = matching_pred_signs & matching_diff_after & deviation_influence
    
    if return_denoised:
        return outer_influence, denoised
    return outer_influence

# --- Logic for Each CFG Method ---

def apply_skim(x_orig: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor,
               cond_scale: float, skimming_scale: float, disable_flipping_filter: bool) -> torch.Tensor:
    """
    Helper function to apply the core skimming calculation to a tensor.

    It calculates the difference between a high-CFG and low-CFG (skimmed)
    denoised result and subtracts this difference from the conditional prediction
    in the areas identified by the mask.

    Args:
        x_orig (torch.Tensor): The original noisy tensor (latent).
        cond (torch.Tensor): The conditional prediction to be modified.
        uncond (torch.Tensor): The opposing unconditional prediction.
        cond_scale (float): The original CFG scale.
        skimming_scale (float): The lower CFG scale used for skimming.
        disable_flipping_filter (bool): Flag to pass to the mask generation.

    Returns:
        torch.Tensor: The modified conditional prediction tensor.
    """
    outer_influence, denoised = get_skimming_mask(x_orig, cond, uncond, cond_scale, True, disable_flipping_filter)
    
    # Calculate the denoised state using the lower 'skimming_scale'.
    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond)))
    
    # Find the difference caused by the lower CFG scale.
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    
    # Apply the correction to the 'cond' tensor where the mask is active.
    cond[outer_influence] = cond[outer_influence] - (low_cfg_denoised_outer_difference[outer_influence] / cond_scale)
    return cond

def apply_single_scale_skim(z: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor,
                            cfg_scale: float, skim_cfg: float, full_skim_neg: bool,
                            disable_flip: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the 'Single Scale' skimming logic to both cond and uncond tensors.

    Args:
        z (torch.Tensor): The original noisy tensor (latent).
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        cfg_scale (float): The original CFG scale.
        skim_cfg (float): The lower CFG scale for skimming.
        full_skim_neg (bool): If True, fully skims the negative prompt.
        disable_flip (bool): Flag for the mask generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    uncond = apply_skim(z, uncond, cond, cfg_scale, skim_cfg if not full_skim_neg else 0, disable_flip)
    cond = apply_skim(z, cond, uncond, cfg_scale - 1, skim_cfg, disable_flip)
    return cond, uncond

def apply_replace_skim(z: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor,
                       cfg_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the 'Replace' skimming logic.

    In areas of high influence, the unconditional prediction is replaced by the
    conditional prediction, effectively neutralizing the negative guidance there.

    Args:
        z (torch.Tensor): The original noisy tensor (latent).
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        cfg_scale (float): The original CFG scale.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    skim_mask = get_skimming_mask(z, cond, uncond, cfg_scale)
    uncond[skim_mask] = cond[skim_mask]
    skim_mask = get_skimming_mask(z, uncond, cond, cfg_scale - 1)
    uncond[skim_mask] = cond[skim_mask]
    return cond, uncond

def apply_linear_interpolation_skim(z: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor,
                                    cfg_scale: float, skim_cfg: float, skim_cfg_neg: float,
                                    is_dual: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the 'Linear Interpolation' skimming logic.

    Blends the unconditional and conditional predictions in high-influence areas
    based on the provided skimming scales.

    Args:
        z (torch.Tensor): The original noisy tensor (latent).
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        cfg_scale (float): The original CFG scale.
        skim_cfg (float): The positive skimming scale.
        skim_cfg_neg (float): The negative skimming scale (for dual mode).
        is_dual (bool): True if using separate positive and negative scales.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    fallback_weight_uncond = (skim_cfg - 1) / (cfg_scale - 1)
    fallback_weight_cond = (skim_cfg_neg - 1) / (cfg_scale - 1) if is_dual else fallback_weight_uncond

    # Blend where uncond influences cond
    skim_mask = get_skimming_mask(z, cond, uncond, cfg_scale)
    uncond[skim_mask] = torch.lerp(uncond[skim_mask], cond[skim_mask], 1 - fallback_weight_uncond)

    # Blend where cond influences uncond
    skim_mask = get_skimming_mask(z, uncond, cond, cfg_scale)
    uncond[skim_mask] = torch.lerp(uncond[skim_mask], cond[skim_mask], 1 - fallback_weight_cond)
    return cond, uncond

def apply_ami_cfg(cond: torch.Tensor, uncond: torch.Tensor, alpha_base: float, power: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Attention-Masked Interpolation CFG (AMI-CFG).

    This method uses the magnitude of the conditional tensor as a proxy for
    semantic importance. It then interpolates the unconditional prediction
    towards the conditional one more strongly in less important areas.

    Args:
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        alpha_base (float): The base interpolation factor (strength of the effect).
        power (float): An exponent to control the influence of the importance map.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    # Create an importance map from the absolute magnitude of the conditional tensor.
    importance_map = torch.abs(cond).sum(dim=1, keepdim=True)
    
    # Normalize the map to a [0, 1] range for consistent application.
    map_min, map_max = importance_map.min(), importance_map.max()
    if map_max > map_min:
        importance_map = (importance_map - map_min) / (map_max - map_min)
    
    # Calculate a spatially-varying alpha. Regions of low importance get stronger interpolation.
    alpha_tensor = alpha_base * (1.0 - importance_map).pow(power)
    
    # Apply linear interpolation (lerp) using the calculated alpha tensor.
    uncond = uncond + (cond - uncond) * alpha_tensor
    return cond, uncond

def apply_pag_cfg(cond: torch.Tensor, uncond: torch.Tensor, percentile: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Perceptual Adaptive Guidance (PAG-CFG).

    This method limits the magnitude of the guidance vector (`cond - uncond`) by
    clipping its values based on a high percentile. This prevents oversaturation
    and artifacts without losing the core guidance direction.

    Args:
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        percentile (float): The percentile (0.0 to 1.0) for magnitude clipping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    guidance = cond - uncond
    magnitude = torch.abs(guidance)
    
    # Determine the threshold from a high percentile of the guidance magnitudes.
    threshold = torch.quantile(magnitude, percentile)
    
    # Create scaling factors that will reduce only the values exceeding the threshold.
    scaling_mask = magnitude > threshold
    scaling_factors = torch.ones_like(magnitude)
    scaling_factors[scaling_mask] = threshold / magnitude[scaling_mask]
    
    # Reconstruct the conditional prediction with the clipped guidance.
    modified_guidance = guidance * scaling_factors
    cond = uncond + modified_guidance
    return cond, uncond

def apply_ssdt_cfg_skimming(z: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor, cfg_scale: float,
                            progress: float, start_scale: float, end_scale: float, schedule: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the scheduled skimming part of SSDT-CFG.

    This function interpolates the skimming strength over the course of the
    sampling process, often applying a stronger effect at the beginning.

    Args:
        z (torch.Tensor): The original noisy tensor (latent).
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        cfg_scale (float): The original CFG scale.
        progress (float): The current sampling progress (0.0 to 1.0).
        start_scale (float): Skimming strength at the start of sampling.
        end_scale (float): Skimming strength at the end of sampling.
        schedule (str): The type of interpolation ("linear" or "cosine").

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    if schedule == 'cosine':
        progress = (1 - math.cos(progress * math.pi)) / 2

    # Calculate the current skimming strength based on the schedule.
    current_skim_strength = torch.lerp(torch.tensor(start_scale), torch.tensor(end_scale), progress).item()
    
    # Apply skimming as a linear interpolation.
    skim_mask = get_skimming_mask(z, cond, uncond, cfg_scale)
    uncond[skim_mask] = torch.lerp(cond[skim_mask], uncond[skim_mask], current_skim_strength)
    return cond, uncond

def apply_ssdt_dynamic_thresholding(noise_pred: torch.Tensor, percentile: float) -> torch.Tensor:
    """
    Applies the dynamic thresholding part of SSDT-CFG to the final noise prediction.

    This clamps the final combined prediction based on a percentile of its
    magnitudes, preventing extreme values that can cause artifacts.

    Args:
        noise_pred (torch.Tensor): The final noise prediction after CFG.
        percentile (float): The percentile (0.0 to 1.0) for clamping.

    Returns:
        torch.Tensor: The clamped noise prediction tensor.
    """
    magnitude = torch.abs(noise_pred)
    threshold = torch.quantile(magnitude, percentile)
    return torch.clamp(noise_pred, -threshold, threshold)

def apply_lss_cfg(uncond: torch.Tensor, beta: float, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Applies Latent Space Smoothing (LSS-CFG).

    This method regularizes the unconditional prediction by blending it with a
    Gaussian-blurred version of itself. This can reduce noise and improve coherence.

    Args:
        uncond (torch.Tensor): The unconditional noise prediction.
        beta (float): The strength of the smoothing effect (blend factor).
        kernel_size (int): The kernel size for the Gaussian blur (must be odd).
        sigma (float): The sigma value for the Gaussian blur.

    Returns:
        torch.Tensor: The modified unconditional tensor.
    """
    # Create a smoothed version of the unconditional prediction.
    uncond_smoothed = gaussian_blur(uncond, kernel_size=kernel_size, sigma=sigma)
    
    # Interpolate between the original and the smoothed version.
    return torch.lerp(uncond, uncond_smoothed, beta)

def dispatch_cfg_modification(cond: torch.Tensor, uncond: torch.Tensor, z: torch.Tensor,
                              cfg_scale: float, progress: float, cfg_args: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Acts as a central dispatcher for all CFG modification algorithms.

    This function reads the 'mode' from the cfg_args dictionary and calls the
    appropriate helper function, passing only the necessary parameters. This
    isolates the sampler from the specific implementation details of each
    CFG modification technique.

    Args:
        cond (torch.Tensor): The conditional noise prediction.
        uncond (torch.Tensor): The unconditional noise prediction.
        z (torch.Tensor): The original noisy tensor (latent).
        cfg_scale (float): The current CFG scale.
        progress (float): The current sampling progress (0.0 to 1.0).
        cfg_args (dict): The dictionary containing all settings from the UI nodes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified cond and uncond tensors.
    """
    mode = cfg_args.get("mode")

    # Dispatch to the appropriate handler based on the selected mode
    if mode == "Single Scale":
        return apply_single_scale_skim(
            z, cond, uncond, cfg_scale,
            cfg_args.get("Skimming_CFG"),
            cfg_args.get("full_skim_negative"),
            cfg_args.get("disable_flipping_filter")
        )
    elif mode == "Replace":
        return apply_replace_skim(z, cond, uncond, cfg_scale)
    elif mode == "Linear Interpolation":
        return apply_linear_interpolation_skim(
            z, cond, uncond, cfg_scale,
            cfg_args.get("Skimming_CFG"),
            cfg_args.get("Skimming_CFG_negative"),
            is_dual=False
        )
    elif mode == "Linear Interpolation Dual Scales":
        return apply_linear_interpolation_skim(
            z, cond, uncond, cfg_scale,
            cfg_args.get("Skimming_CFG"),
            cfg_args.get("Skimming_CFG_negative"),
            is_dual=True
        )
    elif mode == "AMI-CFG":
        return apply_ami_cfg(
            cond, uncond,
            cfg_args.get("alpha_base"),
            cfg_args.get("power")
        )
    elif mode == "PAG-CFG":
        return apply_pag_cfg(cond, uncond, cfg_args.get("percentile"))
    elif mode == "SSDT-CFG":
        return apply_ssdt_cfg_skimming(
            z, cond, uncond, cfg_scale, progress,
            cfg_args.get("start_scale"),
            cfg_args.get("end_scale"),
            cfg_args.get("schedule")
        )
    elif mode == "LSS-CFG":
        # LSS-CFG only modifies the unconditional tensor
        modified_uncond = apply_lss_cfg(
            uncond,
            cfg_args.get("beta"),
            cfg_args.get("kernel_size"),
            cfg_args.get("sigma")
        )
        return cond, modified_uncond

    # If mode is not recognized, return original tensors without modification
    return cond, uncond

def get_cfg_log_details(cfg_args: dict) -> str:
    """
    Creates a formatted string of key parameters for the active CFG mode.

    This helper function is called by the sampler to generate informative log
    messages without cluttering the sampler's own logic.

    Args:
        cfg_args (dict): The dictionary containing all settings from the UI.

    Returns:
        str: A formatted string for logging, e.g., "alpha_base=0.50, power=2.0".
    """
    mode = cfg_args.get("mode")
    details = []

    try:
        if mode in ["Single Scale", "Linear Interpolation", "Linear Interpolation Dual Scales"]:
            details.append(f"Skimming CFG={cfg_args.get('Skimming_CFG'):.2f}")
            if mode == "Single Scale":
                details.append(f"Full skimm negative={cfg_args.get('full_skim_negative')}")
                details.append(f"Disable fliping filter={cfg_args.get('disable_flipping_filter')}")
            elif mode == "Linear Interpolation Dual Scales":
                details.append(f"Skimming CFG negative=: {cfg_args.get('Skimming_CFG_negative', 0.0):.2f}")
        elif mode == "AMI-CFG":
            details.append(f"alpha_base={cfg_args.get('alpha_base'):.2f}")
            details.append(f"power={cfg_args.get('power'):.1f}")
        elif mode == "PAG-CFG":
            details.append(f"percentile={cfg_args.get('percentile'):.2f}")
        elif mode == "SSDT-CFG":
            details.append(f"schedule='{cfg_args.get('schedule')}'")
            details.append(f"scale=[{cfg_args.get('start_scale'):.2f} -> {cfg_args.get('end_scale'):.2f}]")
        elif mode == "LSS-CFG":
            details.append(f"beta={cfg_args.get('beta'):.2f}")
            details.append(f"kernel={cfg_args.get('kernel_size')}")
            details.append(f"sigma={cfg_args.get('sigma'):.1f}")
    except (TypeError, KeyError) as e:
        log.warning(f"Could not format log details for mode '{mode}' due to missing key: {e}")
        return "invalid params"

    return ", ".join(details)
# --- End of cfg_utils.py ---