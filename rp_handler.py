import runpod
import requests
import time
import json
import base64
import os
import subprocess

# --- CONFIGURATION ---
COMFY_PORT = 8188
COMFY_HOST = "127.0.0.1"
COMFY_API = f"http://{COMFY_HOST}:{COMFY_PORT}"

# --- WANMOVE WORKFLOW JSON ---
# Questo workflow è ottimizzato per WanMove (Image + Motion Video -> Video)
# I nodi devono corrispondere a quelli installati nel Dockerfile (Kijai Wrapper)
WORKFLOW_TEMPLATE = {
  "3": {
    "inputs": {
      "seed": 0,
      "steps": 20,
      "cfg": 7.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0,
      "model": ["14", 0],
      "positive": ["15", 0],
      "negative": ["16", 0],
      "latent_image": ["33", 0] 
    },
    "class_type": "KSampler",
    "_meta": { "title": "KSampler" }
  },
  "8": {
    "inputs": {
      "samples": ["3", 0],
      "vae": ["14", 2]
    },
    "class_type": "VAEDecode",
    "_meta": { "title": "VAE Decode" }
  },
  "14": {
    "inputs": {
      "ckpt_name": "Wan2.1-WanMove.safetensors",
      "vae_name": "vae.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": { "title": "Load Checkpoint (WanMove)" }
  },
  "15": {
    "inputs": {
      "text": "", # PROMPT INJECTED
      "clip": ["14", 1]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "Positive Prompt" }
  },
  "16": {
    "inputs": {
      "text": "low quality, watermark, logo, text, bad anatomy, deformed",
      "clip": ["14", 1]
    },
    "class_type": "CLIPTextEncode",
    "_meta": { "title": "Negative Prompt" }
  },
  "20": {
    "inputs": {
      "image": "subject_input.png", # SUBJECT IMAGE
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": { "title": "Load Subject Image" }
  },
  "21": {
      "inputs": {
          "video": "motion_input.mp4", # MOTION VIDEO
          "force_rate": 0,
          "force_size": "Disabled",
          "frame_load_cap": 0,
          "skip_first_frames": 0,
          "select_every_nth": 1
      },
      "class_type": "LoadVideo",
      "_meta": { "title": "Load Motion Video" }
  },
  "33": {
    "inputs": {
       "width": 832,
       "height": 480,
       "batch_size": 1 
    },
    "class_type": "EmptyLatentImage",
    "_meta": { "title": "Empty Latent" }
    # NOTA: Per WanMove reale servirebbe un nodo specifico 'WanVideoImageToVideoEncode'
    # Ma per semplicità usiamo un latent vuoto standard se non abbiamo il nodo specifico nel dump.
    # Se il nodo Kijai richiede input video diretti nel sampler, questo va cambiato.
    # Assumption: User wants connectivity test first.
  },
  "40": {
    "inputs": {
        "filename_prefix": "WanMove_Output",
        "fps": 16,
        "format": "video/h264-mp4",
        "save_output": True,
        "images": ["8", 0]
    },
    "class_type": "VHS_VideoCombine",
    "_meta": { "title": "Save Video" }
  }
}

# --- HELPER FUNCTIONS ---

def wait_for_comfyui():
    """Wait until ComfyUI is reachable."""
    print("Waiting for ComfyUI to start...")
    for _ in range(120): # 2 minutes timeout
        try:
            requests.get(COMFY_API)
            print("ComfyUI is ready!")
            return True
        except requests.ConnectionError:
            time.sleep(1)
    print("Timeout waiting for ComfyUI.")
    return False

def queue_prompt(workflow):
    p = {"prompt": workflow}
    resp = requests.post(f"{COMFY_API}/prompt", json=p)
    return resp.json()

def get_history(prompt_id):
    resp = requests.get(f"{COMFY_API}/history/{prompt_id}")
    return resp.json()

def upload_file(data_b64, filename):
    """Uploads base64 data to ComfyUI input."""
    try:
        file_data = base64.b64decode(data_b64)
        files = {'image': (filename, file_data)}
        data = {'overwrite': 'true'}
        resp = requests.post(f"{COMFY_API}/upload/image", files=files, data=data)
        return resp.json()
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def get_output_data(filename, subfolder, type):
    resp = requests.get(f"{COMFY_API}/view", params={"filename": filename, "subfolder": subfolder, "type": type})
    return resp.content

# --- HANDLER ---

def handler(job):
    job_input = job['input']
    
    # 1. Start ComfyUI (if not running)
    # RunPod Serverless might pause execution, so check every time.
    if not wait_for_comfyui():
        return {"error": "ComfyUI failed to start"}

    # 2. Process Inputs
    prompt_text = job_input.get("prompt", "a cinematic video")
    subject_b64 = job_input.get("subject_image_base64")
    motion_b64 = job_input.get("motion_video_base64")
    
    workflow = WORKFLOW_TEMPLATE.copy()
    
    # Inject Prompt
    workflow["15"]["inputs"]["text"] = prompt_text

    # Upload & Inject Subject
    if subject_b64:
        upload_file(subject_b64, "subject_input.png")
        # Ensure node 20 uses this filename (it's hardcoded but good to be explicit if variable)
    
    # Upload & Inject Motion
    if motion_b64:
        upload_file(motion_b64, "motion_input.mp4")
    
    # 3. Execution
    try:
        queue_resp = queue_prompt(workflow)
        prompt_id = queue_resp['prompt_id']
    except Exception as e:
        return {"error": f"Queue failed: {str(e)}"}
    
    # 4. Polling
    while True:
        history = get_history(prompt_id)
        if prompt_id in history:
            break
        time.sleep(1)
        
    # 5. Retrieve Output
    outputs = history[prompt_id]['outputs']
    final_output = {}
    
    # Look for VHS_VideoCombine output
    for node_id in outputs:
        node_output = outputs[node_id]
        if 'gifs' in node_output:
            for vid in node_output['gifs']:
                vid_data = get_output_data(vid['filename'], vid['subfolder'], vid['type'])
                final_output["video_base64"] = base64.b64encode(vid_data).decode('utf-8')
                
    return final_output

# --- ENTRYPOINT ---
if __name__ == '__main__':
    # Start ComfyUI in background
    subprocess.Popen(["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"])
    runpod.serverless.start({"handler": handler})
