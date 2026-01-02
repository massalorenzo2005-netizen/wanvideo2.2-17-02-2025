# amd_fix.py - AMD GPU compatibility fix for tensor device mismatches
import torch

# Store the original concat function
original_concat = torch.concat

# Create a safe replacement that ensures tensors are on the same device
def safe_concat(tensors, *args, **kwargs):
    if not isinstance(tensors, (list, tuple)) or len(tensors) <= 1:
        return original_concat(tensors, *args, **kwargs)
    
    # Find the first CUDA device in the list
    target_device = None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.device.type == 'cuda':
            target_device = t.device
            break
    
    # If no CUDA device found, use the first tensor's device
    if target_device is None and isinstance(tensors[0], torch.Tensor):
        target_device = tensors[0].device
    
    # Only move tensors if needed
    if target_device is not None:
        fixed_tensors = []
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.device != target_device:
                fixed_tensors.append(t.to(target_device))
            else:
                fixed_tensors.append(t)
        return original_concat(fixed_tensors, *args, **kwargs)
    else:
        return original_concat(tensors, *args, **kwargs)

# Apply the patch
def apply_fix():
    try:
        # Only apply this when running on AMD GPU
        if torch.cuda.is_available() and "amd" in torch.cuda.get_device_name(0).lower():
            # Apply our patch
            torch.concat = safe_concat
            print("Applied device-safe concatenation patch for AMD GPUs")
        return True
    except Exception as e:
        print(f"Failed to apply AMD fix: {str(e)}")
        return False

# Apply fix when imported
apply_fix()
