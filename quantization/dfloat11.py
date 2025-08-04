from ..utils import log
import os
import folder_paths
from typing import Optional, Union, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig
from tqdm import tqdm
from safetensors.torch import load_file
from .dfloat11_core import get_hook
import re

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

# dfloat11 function, but with state dict renaming to comfy model layers
def load_and_replace_tensors(model, directory_path, dfloat11_config, cpu_offload=False, pin_memory=False):
    """
    Loads DFloat11 compressed weights from safetensors files and configures the model
    to use them with on-the-fly decompression.
    
    Args:
        model: The PyTorch model to load weights into
        directory_path: Path to the directory containing safetensors files
        dfloat11_config: Configuration for DFloat11 compression
        
    Returns:
        The model with configured DFloat11 compression
    """
    threads_per_block = dfloat11_config['threads_per_block']
    bytes_per_thread  = dfloat11_config['bytes_per_thread']
    pattern_dict      = dfloat11_config['pattern_dict']
    
    # Get all .safetensors files in the directory
    safetensors_files = [f for f in os.listdir(directory_path) if f.endswith('.safetensors')]
    loading_desc = 'Loading DFloat11 safetensors'

    for file_name in tqdm(safetensors_files, desc=loading_desc):
        file_path = os.path.join(directory_path, file_name)
        
        # Load the tensors from the file
        loaded_tensors = load_file(file_path)
        loaded_tensors = rename_diffusers_to_comfy(loaded_tensors) # -- the only change

        # Iterate over each tensor in the file
        for tensor_name, tensor_value in loaded_tensors.items():
            # Check if this tensor exists in the model's state dict
            if tensor_name in model.state_dict():
                # Get the parameter or buffer
                if tensor_name in dict(model.named_parameters()):
                    # It's a parameter, we can set it directly
                    param = dict(model.named_parameters())[tensor_name]
                    if param.shape == tensor_value.shape:
                        param.data.copy_(tensor_value)
                    else:
                        log.error(f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}")
                else:
                    # It's a buffer, we can also set it directly
                    buffer = dict(model.named_buffers())[tensor_name]
                    if buffer.shape == tensor_value.shape:
                        buffer.copy_(tensor_value)
                    else:
                        log.error(f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}")
            else:
                # Split the tensor name to get module path
                parts = tensor_name.split('.')
                module = model
                
                # Navigate to the correct module
                for i, part in enumerate(parts[:-1]):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        log.error(f"Cannot find module path for {tensor_name}")
                        break
                else:
                    if parts[-1] == 'split_positions':
                        setattr(module, 'split_positions', tensor_value.tolist())
                    else:
                        # Register the buffer to the found module
                        module.register_buffer(parts[-1], tensor_value)

                    # Set up decompression for encoded weights
                    if parts[-1] == 'encoded_exponent':
                        # Register the decode hook to decompress weights during forward pass
                        module.register_forward_pre_hook(get_hook(threads_per_block, bytes_per_thread))

                        # Configure weight injection based on module type
                        for pattern, attr_names in pattern_dict.items():
                            if re.fullmatch(pattern, '.'.join(parts[:-1])):
                                if isinstance(module, nn.Embedding):
                                    # Remove weight attribute from embedding layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                elif isinstance(module, nn.Linear):
                                    # Remove weight attribute from linear layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                else:
                                    # Handle special case for multi-module weight injection
                                    setattr(module, 'weight_injection_modules', [])
                                    for attr_path in attr_names:
                                        parts = attr_path.split('.')
                                        target = module
                                        for p in parts:
                                            target = getattr(target, p)

                                        tmp = target.weight
                                        delattr(target, 'weight')
                                        del tmp
                                        module.weight_injection_modules.append(target)
                    elif parts[-1] == 'output_positions':
                        # Calculate required shared memory size for CUDA kernel
                        output_positions_np = tensor_value.view(torch.uint32).numpy()
                        setattr(
                            module,
                            'shared_mem_size',
                            threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
                        )
    
    return model

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
        pin_memory: bool = False,
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

            log.info(f"Total model size: {model_bytes / 1e9:0.4f} GB")

        # Move model to specified device ~~or distribute across multiple devices~~
        # -- KJ wrapper will take care of it
        model = model.to(device)

        return model
