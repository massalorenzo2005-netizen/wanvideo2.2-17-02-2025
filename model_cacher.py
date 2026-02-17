import os
import time
from huggingface_hub import hf_hub_download

# Configuration
MOUNT_POINT = "/runpod-volume"
MODELS_DIR = os.path.join(MOUNT_POINT, "checkpoints", "WanVideo", "WanMove")
VAE_DIR = os.path.join(MOUNT_POINT, "vae", "wanvideo")
T5_DIR = os.path.join(MOUNT_POINT, "clip", "wanvideo") # T5 loaded as CLIP/TextEncoder

# Ensure directories exist (if mount allows writing)
for d in [MODELS_DIR, VAE_DIR, T5_DIR]:
    os.makedirs(d, exist_ok=True)

def check_and_download(repo_id, filename, target_dir):
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        print(f"[CACHE] Found {filename} in {target_dir}. Skipping download.")
        return
    
    print(f"[CACHE] MISSING {filename}. Downloading from {repo_id}...")
    try:
        # Download directly to target path
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print(f"[CACHE] Downloaded {filename} successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")

if __name__ == "__main__":
    print("--- Starting Auto-Populate ---")
    
    # 1. WanMove Model (Quantized or Full)
    # Repo: Kijai/WanVideo-wrapper (Example, replace with real source if needed)
    # Using specific file from previous context: Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors
    check_and_download(
        "Kijai/WanVideo-wrapper", 
        "Wan2.1-WanMove_fp8_scaled_e4m3fn.safetensors", # Double check filename on HF
        MODELS_DIR
    )

    # 2. VAE
    check_and_download(
        "Kijai/WanVideo-wrapper",
        "Wan2_1_VAE_bf16.safetensors",
        VAE_DIR
    )

    # 3. T5 Encoder
    check_and_download(
        "Kijai/WanVideo-wrapper",
        "umt5-xxl-enc-bf16.safetensors",
        T5_DIR
    )

    print("--- Auto-Populate Complete ---")
