# file: cfg_nodes.py
#
# This file defines nodes for advanced CFG modifications in the WanVideoWrapper framework.
# It includes a base class for parameter nodes, individual parameter nodes for each
# CFG algorithm, and a controller node (AdvancedCFGArgs) to manage them.
# The NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS dictionaries register
# the nodes with the framework for dynamic instantiation and UI display.

import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CFGArgsNodeBase:
    """A base class for CFG parameter nodes to reduce code duplication."""
    RETURN_TYPES = ("CFG_ARGS",)
    RETURN_NAMES = ("cfg_settings",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/CFG Modifiers"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        """Processes the node's inputs."""
        kwargs["_node_name"] = self.__class__.__name__
        return (kwargs,)


class CFGSkimmingArgs(CFGArgsNodeBase):
    """
    Manages classic CFG Skimming methods. These algorithms identify areas where
    high CFG might cause artifacts (e.g., color burn) by analyzing sign agreement
    between predictions. They then reduce guidance strength in those specific areas.
    """
    DESCRIPTION = "Manages classic CFG Skimming methods. These algorithms identify areas where high CFG might cause artifacts (e.g., color burn) and reduce the guidance strength in those specific areas. 'Single Scale' corrects both prompts, 'Replace' neutralizes the negative prompt, and 'Linear Interpolation' blends the two."
    
    @classmethod
    def INPUT_TYPES(cls):
        """Defines the inputs for the original skimming methods."""
        return {"required": {
            "mode": ("COMBO", {
                "options": ["Single Scale", "Replace", "Linear Interpolation", "Linear Interpolation Dual Scales"],
                "default": "Single Scale",
                "tooltip": "Selects the specific skimming algorithm. 'Single Scale' is a balanced correction, 'Replace' is aggressive, and 'Linear Interpolation' offers a smooth blend."
            }),
            "Skimming_CFG": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 7.0, "step": 0.5, "tooltip": "The target lower CFG value to blend towards. Lower values (e.g., 1.0-2.0) provide stronger artifact reduction but may decrease detail."}),
            "Skimming_CFG_negative": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 7.0, "step": 0.5, "tooltip": "The negative scale, used ONLY in 'Linear Interpolation Dual Scales' mode to control negative prompt blending."}),
            "full_skim_negative": ("BOOLEAN", {"default": False, "tooltip": "If enabled, aggressively neutralizes the negative prompt's influence in high-conflict areas by setting its skimming scale to 0."}),
            "disable_flipping_filter": ("BOOLEAN", {"default": False, "tooltip": "Disables one of the internal criteria for mask generation (deviation influence), resulting in a broader and often stronger effect."}),
        }}

NODE_CLASS_MAPPINGS["CFGSkimmingArgs"] = CFGSkimmingArgs
NODE_DISPLAY_NAME_MAPPINGS["CFGSkimmingArgs"] = "CFG Skimming Args WIP+BETA"


class AMICFGArgs(CFGArgsNodeBase):
    """
    Controls AMI-CFG (Attention-Masked Interpolation). This method creates a
    semantic importance map based on prediction magnitude and applies stronger
    guidance reduction (blending) in less important/detailed areas.
    """
    DESCRIPTION = "Controls AMI-CFG. This method creates a semantic importance map (based on prediction magnitude) and applies stronger guidance reduction (blending) in less important/detailed areas, preserving detail in key subjects."

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the inputs for AMI-CFG."""
        return {"required": {
            "alpha_base": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Base strength of the blending applied to low-importance areas. Higher values lead to a more pronounced reduction of negative prompt influence."}),
            "power": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Contrast exponent for the importance map. Higher values create a sharper distinction between important (less affected) and unimportant (more affected) areas."}),
        }}

NODE_CLASS_MAPPINGS["AMICFGArgs"] = AMICFGArgs
NODE_DISPLAY_NAME_MAPPINGS["AMICFGArgs"] = "CFG AMI-CFG Args WIP+BETA"


class PAGCFGArgs(CFGArgsNodeBase):
    """
    Controls PAG-CFG (Perceptual Adaptive Guidance). This method analyzes the
    entire guidance vector ('cond' - 'uncond') and scales down only the most
    extreme values that exceed a certain percentile, preventing pixel 'burn-out'.
    """
    DESCRIPTION = "Controls PAG-CFG. This method analyzes the entire guidance vector ('cond' - 'uncond') and scales down only the most extreme values that exceed a certain percentile. This prevents pixel 'burn-out' without altering the overall guidance direction."

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the inputs for PAG-CFG."""
        return {"required": {
            "percentile": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for clipping. At 0.95, any guidance signal in the top 5% of intensity will be scaled down to the 95th percentile level. Lowering this value strengthens the clipping effect."}),
        }}

NODE_CLASS_MAPPINGS["PAGCFGArgs"] = PAGCFGArgs
NODE_DISPLAY_NAME_MAPPINGS["PAGCFGArgs"] = "CFG PAG-CFG Args WIP+BETA"


class SSDTCFGArgs(CFGArgsNodeBase):
    """
    Controls SSDT-CFG (Scheduled Skimming w/ Dynamic Thresholding). This is a
    two-stage process: 1. Applies a guidance-reducing effect that weakens over
    time (Scheduled Skimming). 2. Applies Dynamic Thresholding to the final output.
    """
    DESCRIPTION = "Controls SSDT-CFG. This is a two-stage process. First, it applies a guidance-reducing effect that weakens over time (Scheduled Skimming). Second, it applies Dynamic Thresholding to the final output, clamping the brightest and darkest values to prevent 'sparkle' artifacts."

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the inputs for SSDT-CFG."""
        return {"required": {
            "start_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the skimming effect at the start of sampling. 1.0 is minimal effect, 0.0 is a very strong blend towards the conditional prompt."}),
            "end_scale": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the skimming effect at the end of sampling. 1.0 is minimal effect, 0.0 is a very strong blend."}),
            "schedule": ("COMBO", {"options": ["linear", "cosine"], "default": "linear", "tooltip": "The interpolation curve for the effect's strength as it progresses from start_scale to end_scale."}),
            "threshold_percentile": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Final output clamp. At 0.99, it clamps the brightest/darkest 1% of values. Use this to remove extreme 'firefly' pixels without affecting the main image."}),
        }}

NODE_CLASS_MAPPINGS["SSDTCFGArgs"] = SSDTCFGArgs
NODE_DISPLAY_NAME_MAPPINGS["SSDTCFGArgs"] = "CFG SSDT-CFG Args WIP+BETA"


class LSSCFGArgs(CFGArgsNodeBase):
    """
    Controls LSS-CFG (Latent Space Smoothing). This method applies a gentle
    Gaussian blur to the negative conditioning ('uncond'), which has a regularizing
    effect, reducing high-frequency noise and improving smoothness.
    """
    DESCRIPTION = "Controls LSS-CFG. This method applies a gentle Gaussian blur to the negative conditioning ('uncond'). This has a regularizing effect, reducing high-frequency noise and improving overall image smoothness and coherence."
    
    @classmethod
    def INPUT_TYPES(cls):
        """Defines the inputs for LSS-CFG."""
        return {"required": {
            "beta": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend factor between the original and blurred negative prompt. At 0.5, the negative prompt is a 50/50 mix. Higher values increase the smoothing effect."}),
            "kernel_size": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2, "tooltip": "The size of the blur filter in pixels. Larger values result in a stronger, more spread-out blur. Must be an odd number."}),
            "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "The standard deviation of the Gaussian blur. Higher values create a more intense blur for a given kernel size."}),
        }}
    
    def process(self, **kwargs):
        """Ensures the kernel size is always an odd number."""
        if "kernel_size" in kwargs and kwargs["kernel_size"] % 2 == 0:
            kwargs["kernel_size"] += 1
        return super().process(**kwargs)

NODE_CLASS_MAPPINGS["LSSCFGArgs"] = LSSCFGArgs
NODE_DISPLAY_NAME_MAPPINGS["LSSCFGArgs"] = "CFG LSS-CFG Args WIP+BETA"


class AdvancedCFGArgs:
    """
    The central hub for activating and managing CFG modifications. It takes the
    settings from a connected parameter node and passes them to the sampler. Its
    sole purpose is to route the configuration.
    """
    DESCRIPTION = "The central hub for activating and managing CFG modifications. It takes the settings from a connected parameter node and passes them to the sampler. Its sole purpose is to route the configuration."
    
    RETURN_TYPES = ("CFG_ARGS",)
    RETURN_NAMES = ("cfg_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True
    
    MODE_TO_NODE_MAP = {
        "Single Scale": "CFGSkimmingArgs", "Replace": "CFGSkimmingArgs",
        "Linear Interpolation": "CFGSkimmingArgs", "Linear Interpolation Dual Scales": "CFGSkimmingArgs",
        "AMI-CFG": "AMICFGArgs", "PAG-CFG": "PAGCFGArgs",
        "SSDT-CFG": "SSDTCFGArgs", "LSS-CFG": "LSSCFGArgs"
    }

    def __init__(self):
        """
        Initializes the node and pre-computes the reverse mapping 
        from node name to mode for efficient lookups.
        """
        self.NODE_TO_MODE_MAP = {v: k for k, v in self.MODE_TO_NODE_MAP.items()}

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the inputs for the main controller node."""
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "A master switch to enable or disable the entire advanced CFG modification process."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enables detailed logging of the CFG modification process in the console. Essential for debugging and fine-tuning."}),
                "cfg_settings": ("CFG_ARGS", {"tooltip": "Input for a single parameter node (e.g., 'CFG Skimming Args'). The type of node connected determines the algorithm that will be used by the sampler."})
            }
        }

    def process(self, enabled: bool, verbose: bool, cfg_settings: dict) -> Tuple[dict]:
        """Validates inputs and packages them for the sampler."""
        if not enabled:
            return ({"enabled": False},)
        
        final_args = {"enabled": True, "verbose": verbose}
        node_name = cfg_settings.get("_node_name")
        
        if node_name == "CFGSkimmingArgs":
            mode = cfg_settings.get("mode")
            if mode not in self.MODE_TO_NODE_MAP:
                log.warning(f"AdvancedCFGArgs: Invalid mode '{mode}' selected in CFGSkimmingArgs. The effect will be bypassed.")
                final_args["enabled"] = False
                return (final_args,)
        else:
            mode = self.NODE_TO_MODE_MAP.get(node_name)
            if not mode:
                log.warning(f"AdvancedCFGArgs: Connected node '{node_name}' is not recognized. The effect will be bypassed.")
                final_args["enabled"] = False
                return (final_args,)
        
        if node_name != self.MODE_TO_NODE_MAP.get(mode):
            log.warning(f"AdvancedCFGArgs: Mode '{mode}' requires a '{self.MODE_TO_NODE_MAP.get(mode)}' node, but '{node_name}' is connected. The effect will be bypassed.")
            final_args["enabled"] = False
            return (final_args,)
        
        final_args["mode"] = mode
        final_args.update(cfg_settings)
        return (final_args,)

# Node registration for the framework
# Maps node identifiers to their classes and display names for UI integration
# Note: Display names include 'WIP+BETA' to indicate work-in-progress status
NODE_CLASS_MAPPINGS["AdvancedCFGArgs"] = AdvancedCFGArgs
NODE_DISPLAY_NAME_MAPPINGS["AdvancedCFGArgs"] = "CFG Advanced Control WIP+BETA"
# --- End of cfg_nodes.py ---