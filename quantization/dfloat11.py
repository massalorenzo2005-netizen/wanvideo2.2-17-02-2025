import os
from comfy.cli_args import args

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
    base_path = os.path.abspath(args.base_directory)
    model_dir = os.path.join(base_path, "models", "dfloat11", model)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Subdirectory {model_dir} not found.")
    return model_dir
