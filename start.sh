#!/bin/bash
echo "--- [START] RunPod WanVideo Worker ---"

# 1. Setup Volume Paths
VOLUME_PATH="/runpod-volume"
mkdir -p $VOLUME_PATH/checkpoints/WanVideo
mkdir -p $VOLUME_PATH/vae/wanvideo
mkdir -p $VOLUME_PATH/clip/wanvideo

# 2. Check & Download Models (Resumable)
# Questa parte scarica SOLO se i file non sono già sul volume da 100GB
echo "--- [CHECK] Verifying Models on Volume ---"

# WanMove Checkpoint
if [ ! -f "$VOLUME_PATH/checkpoints/WanVideo/Wan2.1-WanMove.safetensors" ]; then
    echo "--- [DOWNLOAD] WanMove Model (Missing) ---"
    wget -c -q --show-progress -O "$VOLUME_PATH/checkpoints/WanVideo/Wan2.1-WanMove.safetensors" https://huggingface.co/Kijai/WanVideo-wrapper/resolve/main/Wan2.1-WanMove_fp8_scaled_e4m3fn.safetensors
else
    echo "--- [OK] WanMove Model Found ---"
fi

# VAE
if [ ! -f "$VOLUME_PATH/vae/wanvideo/vae.safetensors" ]; then
    echo "--- [DOWNLOAD] VAE (Missing) ---"
    wget -c -q --show-progress -O "$VOLUME_PATH/vae/wanvideo/vae.safetensors" https://huggingface.co/Kijai/WanVideo-wrapper/resolve/main/Wan2_1_VAE_bf16.safetensors
else
    echo "--- [OK] VAE Found ---"
fi

# 3. Configuration Linking
echo "--- [CONFIG] Linking Models to ComfyUI ---"
echo "comfyui:
    base_path: $VOLUME_PATH
    checkpoints: checkpoints
    clip: clip
    vae: vae" > $VOLUME_PATH/extra_model_paths.yaml

# Link extra_model_paths so ComfyUI sees the Volume
ln -sf $VOLUME_PATH/extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml

# 4. Start Handler
echo "--- [EXEC] Starting RunPod Handler ---"
python3 -u /rp_handler.py
