# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2025 Tianyi Zhang
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import torch
import torch.nn as nn
import math
from ..utils import log

# Load CUDA kernel for custom DFloat11 tensor decoding
import pkg_resources
ptx_path = pkg_resources.resource_filename("dfloat11", "decode.ptx")
_decode = cp.RawModule(path=ptx_path).get_function('decode')

offloaded_tensor_names = (
    'encoded_exponent', 'sign_mantissa'
)


class TensorManager:
    """
    Static utility class that manages tensor allocation and reuse
    to minimize memory allocation overhead during tensor reconstruction.
    """
    # Static class variables to store tensors
    _tensors = {}  # Maps device to tensor

    @staticmethod
    def allocate_bfloat16(device, n_elements):
        """
        Get a bfloat16 tensor with at least n_elements on the specified device.

        If a tensor already exists on the device and is larger than n_elements,
        a slice of the tensor with exactly n_elements is returned. If n_elements 
        exceeds the size of the existing tensor, the existing tensor is deallocated 
        and a larger one is allocated.

        Args:
            device: The device to allocate the tensor on (e.g., 'cuda:0')
            n_elements: The exact number of elements required

        Returns:
            A bfloat16 tensor with exactly n_elements on the specified device
        """
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check if we already have a tensor for this device
        if device in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[device]
            
            # If existing tensor is large enough, return a slice of it
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]
            
            # Otherwise, delete the existing tensor to free up memory
            del TensorManager._tensors[device]
            torch.cuda.empty_cache()  # Ensure memory is freed
        
        # Allocate a new tensor
        new_tensor = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        log.info(f'Allocated {n_elements} bf16 on device {device}')
        
        # Store the tensor
        TensorManager._tensors[device] = new_tensor
        
        return new_tensor


def get_hook(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly.
    
    This hook reconstructs full-precision weights from compressed representations
    using a custom CUDA kernel during the forward pass.
    
    Args:
        threads_per_block: CUDA thread configuration 
        bytes_per_thread: Number of bytes processed per CUDA thread
        
    Returns:
        A forward pre-hook function for PyTorch modules
    """
    threads_per_block = tuple(threads_per_block)

    def decode_hook(module, _):
        device = module.luts.device

        # Load offloaded tensors to GPU if not already there
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name, tensor in module.offloaded_tensors.items():
                if not (
                    hasattr(module, tensor_name) and (getattr(module, tensor_name).device == device)
                ):
                    module.register_buffer(tensor_name, tensor.to(device, non_blocking=True))

        # Get dimensions for tensor reconstruction
        n_elements = module.sign_mantissa.numel()
        n_bytes = module.encoded_exponent.numel()
        n_luts = module.luts.shape[0]

        # Get output tensor for reconstructed weights
        reconstructed = TensorManager.allocate_bfloat16(device, n_elements)

        # Configure CUDA grid dimensions for the kernel launch
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        # Launch CUDA kernel to decode the compressed weights
        with cp.cuda.Device(device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=module.shared_mem_size, args=[
                module.luts.data_ptr(),
                module.encoded_exponent.data_ptr(),
                module.sign_mantissa.data_ptr(),
                module.output_positions.data_ptr(),
                module.gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])

        # Inject reconstructed weights into the appropriate module
        if isinstance(module, nn.Linear):
            module.weight = reconstructed.view(
                module.out_features, module.in_features
            )
        elif isinstance(module, nn.Embedding):
            module.weight = reconstructed.view(
                module.num_embeddings, module.embedding_dim
            )
        else:
            # Handle special case where weights need to be split across multiple submodules
            weights = torch.tensor_split(reconstructed, module.split_positions)
            for sub_module, weight in zip(module.weight_injection_modules, weights):
                sub_module.weight = weight.view(sub_module.out_features, sub_module.in_features)

        # Delete tensors from GPU if offloading is enabled
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name in module.offloaded_tensors.keys():
                if hasattr(module, tensor_name):
                    tmp = getattr(module, tensor_name)
                    delattr(module, tensor_name)
                    del tmp

    return decode_hook
