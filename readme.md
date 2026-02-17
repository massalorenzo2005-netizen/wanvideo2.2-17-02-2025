# RunPod Serverless Worker - WanMove (v2.2)

This repository contains a **Serverless Worker** optimized for [WanVideo 2.2](https://github.com/Wan-Video/Wan2.1) using Kijai's ComfyUI Wrapper.

## 🚀 Deployment Guide

1.  **Create Endpoint** in RunPod.
2.  **Select this Repository** (`massalorenzo2005-netizen/wanvideo2.2-...`).
3.  **Use Dockerfile**: Select `Dockerfile` (root).
4.  **Network Volume**: attach a volume (min 100GB) to `/runpod-volume`.

## ⚙️ How it Works (Automagic)

*   **First Boot**: The internal script `start.sh` checks if models exist on your volume.
    *   If missing, it **downloads them automatically** (WanMove + VAE + CLIP).
    *   This takes ~15 mins depending on speed.
*   **Subsequent Boots**: It sees models are present and starts in <5 seconds.

## 📡 API Usage

**Input Payload:**
```json
{
  "input": {
    "prompt": "a cybernetic android dancing, neon lights",
    "subject_image_base64": "...",
    "motion_video_base64": "..."
  }
}
```

**Output:**
```json
{
  "video_base64": "...",
  "format": "mp4"
}
```
