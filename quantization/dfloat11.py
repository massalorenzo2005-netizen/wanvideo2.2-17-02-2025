import os
import folder_paths
from typing import Optional, Union, Dict
import torch.nn as nn
from transformers import AutoConfig

def rename_diffusers_to_comfy(state_dict):
    new_state_dict = {}
    
    # Handle non-block keys first
    non_block_keys = {
        'patch_embedding.weight': 'patch_embedding.weight',
        'patch_embedding.bias': 'patch_embedding.bias',
        'proj_out.weight': 'head.head.weight',
        'proj_out.bias': 'head.head.bias',
        'condition_embedder.text_embedder.linear_1.weight': 'text_embedding.0.weight',
        'condition_embedder.text_embedder.linear_1.bias': 'text_embedding.0.bias',
        'condition_embedder.text_embedder.linear_2.weight': 'text_embedding.2.weight',
        'condition_embedder.text_embedder.linear_2.bias': 'text_embedding.2.bias',
        'condition_embedder.time_embedder.linear_1.weight': 'time_embedding.0.weight',
        'condition_embedder.time_embedder.linear_1.bias': 'time_embedding.0.bias',
        'condition_embedder.time_embedder.linear_2.weight': 'time_embedding.2.weight',
        'condition_embedder.time_embedder.linear_2.bias': 'time_embedding.2.bias',
        'condition_embedder.time_proj.weight': 'time_projection.1.weight',
        'condition_embedder.time_proj.bias': 'time_projection.1.bias'
    }
    
    for old_key, new_key in non_block_keys.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]
    
    # Process block parameters
    for key in state_dict:
        if key.startswith('blocks.'):
            parts = key.split('.')
            block_num = parts[1]
            remaining = '.'.join(parts[2:])
            
            # Handle different block components
            if remaining.startswith('attn1.'):
                # Self attention
                attn_part = remaining[6:]
                if attn_part.startswith('norm_'):
                    # Norm layers
                    norm_type = attn_part[5]
                    new_key = f'blocks.{block_num}.self_attn.norm_{norm_type}.weight'
                elif attn_part.startswith('to_'):
                    # Projection layers
                    proj_type = attn_part[3]
                    if attn_part.endswith('bias'):
                        new_key = f'blocks.{block_num}.self_attn.{proj_type}.bias'
                    else:
                        new_key = f'blocks.{block_num}.self_attn.{proj_type}.weight'
                elif attn_part == 'to_out.0.bias':
                    new_key = f'blocks.{block_num}.self_attn.o.bias'
                else:
                    continue
                    
            elif remaining.startswith('attn2.'):
                # Cross attention
                attn_part = remaining[6:]
                if attn_part.startswith('norm_'):
                    # Norm layers
                    norm_type = attn_part[5]
                    new_key = f'blocks.{block_num}.cross_attn.norm_{norm_type}.weight'
                elif attn_part.startswith('to_'):
                    # Projection layers
                    proj_type = attn_part[3]
                    if attn_part.endswith('bias'):
                        new_key = f'blocks.{block_num}.cross_attn.{proj_type}.bias'
                    else:
                        new_key = f'blocks.{block_num}.cross_attn.{proj_type}.weight'
                elif attn_part == 'to_out.0.bias':
                    new_key = f'blocks.{block_num}.cross_attn.o.bias'
                else:
                    continue
                    
            elif remaining.startswith('ffn.'):
                # Feed-forward network
                ffn_part = remaining[4:]
                if ffn_part.startswith('net.0.proj.bias'):
                    new_key = f'blocks.{block_num}.ffn.0.bias'
                elif ffn_part.startswith('net.0.proj.weight'):
                    new_key = f'blocks.{block_num}.ffn.0.weight'
                elif ffn_part.startswith('net.2.bias'):
                    new_key = f'blocks.{block_num}.ffn.2.bias'
                elif ffn_part.startswith('net.2.weight'):
                    new_key = f'blocks.{block_num}.ffn.2.weight'
                else:
                    continue
                    
            elif remaining.startswith('norm2.'):
                # Norm layer
                norm_part = remaining[6:]
                if norm_part == 'weight':
                    new_key = f'blocks.{block_num}.norm3.weight'
                elif norm_part == 'bias':
                    new_key = f'blocks.{block_num}.norm3.bias'
                else:
                    continue
                    
            elif remaining in ['gaps', 'luts', 'output_positions', 'split_positions', 
                               'encoded_exponent', 'sign_mantissa']:
                new_key = f'blocks.{block_num}.{remaining.split(".")[-1]}'
            elif remaining in ['scale_shift_table']:
                new_key = f'blocks.{block_num}.modulation'
            else:
                continue
                
            new_state_dict[new_key] = state_dict[key]
    
    return new_state_dict

def locate_dfloat11(model):
    base_path = os.path.abspath(folder_paths.base_path)
    model_dir = os.path.join(base_path, "models", "dfloat11", model)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Subdirectory {model_dir} not found.")
    return model_dir

# ===

class DFloat11Model:
    """
    Wrapper class for loading and using models with DFloat11 compressed weights.
    DFloat11 is a custom 11-bit floating point format that provides memory efficiency
    while maintaining numerical accuracy for LLM weights.
    """
    @classmethod
    def from_pretrained(
        cls,
        dfloat11_model_name_or_path: str,
        device: Optional[str] = None,
        bfloat16_model: Optional[nn.Module] = None,
        cpu_offload: bool = False,
        pin_memory: bool = True,
        **kwargs,
    ):
        """
        Load a model with DFloat11 compressed weights from local path or Hugging Face Hub.
        
        Args:
            dfloat11_model_name_or_path: Local path or HF Hub model name
            device: Target device for the model
            device_map: Strategy for distributing model across devices
            max_memory: Maximum memory allocation per device
            bfloat16_model: Optional pre-initialized model to load weights into
            cpu_offload: Enables CPU offloading; only keeps a single block of weights in GPU at once
            pin_memory: Enables memory-pinning/page-locking when using CPU offloading
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_config
            
        Returns:
            Model with DFloat11 compressed weights configured for on-the-fly decompression
        """
        # Resolve model path, downloading from HF Hub if needed
        if os.path.exists(dfloat11_model_name_or_path):
            dfloat11_model_path = dfloat11_model_name_or_path
        else:
            raise FileNotFoundError(f"DFloat11 model not found at {dfloat11_model_name_or_path}")
            
        # Load model configuration
        config = AutoConfig.from_pretrained(dfloat11_model_path)
        model = bfloat16_model
        assert model is not None
        
        # Verify model has DFloat11 configuration
        assert hasattr(config, 'dfloat11_config')
        dfloat11_config = config.dfloat11_config

        # Load compressed weights and configure decompression
        load_and_replace_tensors(model, dfloat11_model_path, dfloat11_config, cpu_offload=cpu_offload, pin_memory=pin_memory)

        if not cpu_offload:
            # Calculate and report model size
            model_bytes = 0
            for param in model.state_dict().values():
                model_bytes += param.nbytes

            print(f"Total model size: {model_bytes / 1e9:0.4f} GB", file=stderr)

        # Move model to specified device ~~or distribute across multiple devices~~
        # -- KJ wrapper will take care of it
        model = model.to(device)

        return model