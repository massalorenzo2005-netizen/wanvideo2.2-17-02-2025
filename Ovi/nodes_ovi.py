import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import folder_paths
from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
from PIL import Image

# Add the local Ovi directory to the Python path
script_directory = os.path.dirname(os.path.abspath(__file__))
if script_directory not in sys.path:
    sys.path.insert(0, script_directory)

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


# ----------------------------
# Helpers
# ----------------------------
def _get_vae_scale(vae_obj):
    """
    Try common attribute names on wrapper and inner model.
    Returns 1.0 if not found.
    """
    for holder in (vae_obj, getattr(vae_obj, "model", None)):
        if holder is None:
            continue
        for name in ("scaling_factor", "scale_factor", "vae_scale", "latent_scaling", "latent_scale"):
            if hasattr(holder, name):
                val = getattr(holder, name)
                try:
                    return float(val)
                except Exception:
                    pass
    return 1.0


def _ensure_cf_hw_video(decoded: torch.Tensor) -> torch.Tensor:
    """
    Make sure decoded video is 4D [C, F, H, W].
    Accepts: [B,C,F,H,W], [B,F,C,H,W], [C,F,H,W], [F,C,H,W], and list/tuple wrappers.
    """
    x = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
    if x.ndim == 5:
        x = x[0]  # drop batch
    if x.ndim != 4:
        raise RuntimeError(f"Unexpected decoded video ndim {x.ndim}; expected 4 or 5.")
    if x.shape[0] in (1, 3, 4):
        return x  # [C,F,H,W]
    if x.shape[1] in (1, 3, 4):
        return x.permute(1, 0, 2, 3).contiguous()  # [F,C,H,W] -> [C,F,H,W]
    # fallback heuristic
    if x.shape[0] <= 4 and x.shape[1] > 4:
        return x
    return x.permute(1, 0, 2, 3).contiguous()


class OviMMAudioVAELoader:
    """Loads MMAudio VAE for audio encoding/decoding in Ovi"""

    @classmethod
    def INPUT_TYPES(s):
        vae_files = folder_paths.get_filename_list("vae")
        vae_16k_files = [f for f in vae_files if "16" in f.lower() and any(ext in f.lower() for ext in ['.pth', '.pt'])]
        vocoder_files = [f for f in vae_files if "vocoder" in f.lower() or "netg" in f.lower()]

        return {
            "required": {
                "vae_16k": (vae_16k_files if vae_16k_files else vae_files,
                           {"tooltip": "MMAudio VAE 16k model (v1-16.pth) from models/vae"}),
                "vocoder": (vocoder_files if vocoder_files else vae_files,
                          {"tooltip": "BigVGAN vocoder (best_netG.pt) from models/vae"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MMAUDIOVAE",)
    RETURN_NAMES = ("mmaudio_vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Loads MMAudio VAE for Ovi audio generation"

    def loadmodel(self, vae_16k, vocoder, precision):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        vae_path = folder_paths.get_full_path("vae", vae_16k)
        vocoder_path = folder_paths.get_full_path("vae", vocoder)

        from .ovi.modules.mmaudio.features_utils import FeaturesUtils

        vae = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            bigvgan_vocoder_ckpt=vocoder_path,
            mode='16k',
            need_vae_encoder=True
        )

        vae.to(device=offload_device, dtype=dtype)
        vae.eval()

        return (vae,)


class OviFusionModelLoader:
    """Loads Ovi Fusion Model using shared WanVideoWrapper components"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fusion_model": (folder_paths.get_filename_list("diffusion_models"),
                               {"tooltip": "Ovi fusion model (e.g., ovi_11B_bf16.safetensors)"}),
                "wan_vae": ("WANVAE",
                          {"tooltip": "Wan VAE from WanVideoVAELoader"}),
                "mmaudio_vae": ("MMAUDIOVAE",
                              {"tooltip": "MMAudio VAE from OviMMAudioVAELoader"}),
                "base_precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "lora": ("WANVIDLORA", {"default": None,
                        "tooltip": "Optional LoRA weights"}),
                "cpu_offload": ("BOOLEAN", {"default": False,
                               "tooltip": "Enable CPU offload to reduce VRAM usage"}),
            }
        }

    RETURN_TYPES = ("OVIMODEL",)
    RETURN_NAMES = ("ovi_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Loads Ovi Fusion model with shared components from WanVideoWrapper"

    def loadmodel(self, fusion_model, wan_vae, mmaudio_vae,
                  base_precision, lora=None, cpu_offload=False):

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        # Load fusion model config
        video_config_path = os.path.join(script_directory, "ovi/configs/model/dit/video.json")
        audio_config_path = os.path.join(script_directory, "ovi/configs/model/dit/audio.json")

        with open(video_config_path) as f:
            video_config = json.load(f)
        with open(audio_config_path) as f:
            audio_config = json.load(f)

        # Initialize fusion model
        from .ovi.modules.fusion import FusionModel

        fusion_model_obj = FusionModel(video_config, audio_config)

        # Load weights
        model_path = folder_paths.get_full_path("diffusion_models", fusion_model)

        # Load checkpoint
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = load_torch_file(model_path, safe_load=True)

        # Load state dict
        missing, unexpected = fusion_model_obj.load_state_dict(state_dict, strict=True)
        print(f"Loaded Ovi fusion model from {model_path}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        del state_dict

        # Move to device
        fusion_model_obj.to(device=device if not cpu_offload else offload_device, dtype=dtype)
        fusion_model_obj.eval()
        fusion_model_obj.set_rope_params()

        # Apply LoRA if provided
        if lora is not None:
            from ..utils import apply_lora
            print(f"Applying LoRA to Ovi fusion model...")
            apply_lora(fusion_model_obj, lora)

        # Package everything together
        ovi_model = {
            "fusion_model": fusion_model_obj,
            "wan_vae": wan_vae,
            "mmaudio_vae": mmaudio_vae,
            "dtype": dtype,
            "device": device,
            "cpu_offload": cpu_offload,
            "video_config": video_config,
            "audio_config": audio_config,
        }

        return (ovi_model,)


class OviSampler:
    """Main sampling node for Ovi video+audio generation (KSampler-style IO)"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ovi_model": ("OVIMODEL",),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (used for both video & audio)"}),
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "video_guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "audio_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1,
                                    "tooltip": "Flow matching shift parameter"}),
                "slg_layer": ("INT", {"default": 11, "min": 0, "max": 40, "step": 1,
                                      "tooltip": "Skip Layer Guidance layer index"}),
                "solver": (["unipc", "dpm++", "euler"], {"default": "unipc"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional input image for I2V mode"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32,
                                   "tooltip": "Video height (T2V mode only)"}),
                "width": ("INT", {"default": 992, "min": 64, "max": 2048, "step": 32,
                                  "tooltip": "Video width (T2V mode only)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("frames", "audio",)
    FUNCTION = "generate"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Generate video and audio using Ovi fusion model (takes CONDITIONING like KSampler)"

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _extract_cond_embedding(cond_obj) -> torch.Tensor:
        """
        Accepts many CONDITIONING shapes that appear in Comfy:
          - list of [tensor, extras] (typical Comfy)
          - list/tuple/dict variants with keys
          - raw tensor
        Returns a [L, C] tensor.
        """
        def _pick_from_dict(d):
            # try common keys used by different nodes
            for k in ("conditioning", "cond", "embeds", "embedding"):
                if k in d and torch.is_tensor(d[k]):
                    return d[k]
            return None

        x = None

        # raw tensor?
        if torch.is_tensor(cond_obj):
            x = cond_obj

        # list/tuple container?
        elif isinstance(cond_obj, (list, tuple)) and len(cond_obj) > 0:
            for item in cond_obj:
                # typical Comfy: [tensor, {...}]
                if isinstance(item, (list, tuple)) and len(item) > 0 and torch.is_tensor(item[0]):
                    x = item[0]
                    break
                # dict-like item
                if isinstance(item, dict):
                    cand = _pick_from_dict(item)
                    if cand is not None:
                        x = cand
                        break

        # dict container?
        elif isinstance(cond_obj, dict):
            x = _pick_from_dict(cond_obj)

        if x is None:
            # helpful debug message (keys/types we saw)
            raise RuntimeError("Could not find 'conditioning' tensor in CONDITIONING input. "
                               f"Got type={type(cond_obj)} with sample={cond_obj[0] if isinstance(cond_obj, (list, tuple)) and cond_obj else 'n/a'}")

        # Normalize shape to [L, C]
        if x.dim() == 3:     # [B, L, C]
            if x.size(0) == 1:
                x = x.squeeze(0)
            else:
                # use the first item if multiple batches
                x = x[0]
        if x.dim() != 2:
            raise RuntimeError(f"Unexpected conditioning shape {tuple(x.shape)}; expected [L, C] or [1, L, C].")

        return x


    # ---- main ------------------------------------------------------------
    def generate(self, ovi_model, positive, negative, seed, steps,
                 video_guidance_scale, audio_guidance_scale, shift, slg_layer, solver,
                 image=None, height=512, width=992):

        fusion_model = ovi_model["fusion_model"]
        wan_vae = ovi_model["wan_vae"]
        mmaudio_vae = ovi_model["mmaudio_vae"]
        cpu_offload = ovi_model["cpu_offload"]
        video_config = ovi_model["video_config"]
        audio_config = ovi_model["audio_config"]

        # extract text embeddings from CONDITIONING
        text_embeddings_pos = self._extract_cond_embedding(positive).to(device, ovi_model["dtype"])
        text_embeddings_neg = self._extract_cond_embedding(negative).to(device, ovi_model["dtype"])
        # use same negative for both streams (matches KSampler’s single NEG input)
        text_embeddings_video_neg = text_embeddings_neg
        text_embeddings_audio_neg = text_embeddings_neg

        # Determine mode
        is_i2v = image is not None

        # Schedulers
        from .ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from .ovi.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
        from diffusers import FlowMatchEulerDiscreteScheduler

        if solver == "unipc":
            scheduler_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            scheduler_video.set_timesteps(steps, device=device, shift=shift)
            timesteps = scheduler_video.timesteps

            scheduler_audio = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            scheduler_audio.set_timesteps(steps, device=device, shift=shift)
        elif solver == "dpm++":
            scheduler_video = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(steps, shift=shift)
            timesteps, _ = retrieve_timesteps(scheduler_video, device=device, sigmas=sampling_sigmas)
            scheduler_audio = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            _ = retrieve_timesteps(scheduler_audio, device=device, sigmas=sampling_sigmas)
        else:
            scheduler_video = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, _ = retrieve_timesteps(scheduler_video, steps, device=device)
            scheduler_audio = FlowMatchEulerDiscreteScheduler(shift=shift)
            _ = retrieve_timesteps(scheduler_audio, steps, device=device)

        # I2V first-frame encode (optional)
        latents_images_t = None
        if is_i2v:
            first_frame = image[0].permute(2, 0, 1).to(device=device, dtype=ovi_model["dtype"])  # [C,H,W]
            wan_vae.model = wan_vae.model.to(device)
            with torch.no_grad():
                # Wan VAE expects [-1,1] and [C, T, H, W] with T=1 at dim=1
                ff = (first_frame * 2 - 1).clamp(-1, 1).unsqueeze(0).unsqueeze(2)  # [1,C,1,H,W]
                latents_images = wan_vae.single_encode(ff, device=device, pbar=False)
                if isinstance(latents_images, (list, tuple)): latents_images = latents_images[0]
                if latents_images.ndim == 5: latents_images = latents_images[0]
                if latents_images.ndim == 4 and latents_images.shape[1] == 1:
                    latents_images_t = latents_images.to(ovi_model["dtype"])          # [C,1,H,W]
                else:
                    latents_images_t = latents_images.to(ovi_model["dtype"]).unsqueeze(1)
            video_latent_h, video_latent_w = latents_images_t.shape[2], latents_images_t.shape[3]
            if cpu_offload:
                wan_vae.model = wan_vae.model.to(offload_device); torch.cuda.empty_cache()
        else:
            video_latent_h = height // 16
            video_latent_w = width // 16

        # Init noises
        video_latent_channel = video_config["in_dim"]
        audio_latent_channel = audio_config["in_dim"]
        video_latent_length = 31
        audio_latent_length = 157

        g = torch.Generator(device=device).manual_seed(seed)
        video_noise = torch.randn((video_latent_channel, video_latent_length, video_latent_h, video_latent_w),
                                  device=device, dtype=ovi_model["dtype"], generator=g)
        audio_noise = torch.randn((audio_latent_length, audio_latent_channel),
                                  device=device, dtype=ovi_model["dtype"], generator=g)

        # seq lens
        max_seq_len_audio = audio_noise.shape[0]
        _ps_h = fusion_model.video_model.patch_size[1]
        _ps_w = fusion_model.video_model.patch_size[2]
        max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_ps_h * _ps_w)

        if cpu_offload:
            fusion_model = fusion_model.to(device)

        # sampling
        pbar = ProgressBar(steps)
        with torch.amp.autocast('cuda', enabled=ovi_model["dtype"] != torch.float32, dtype=ovi_model["dtype"]):
            for t in timesteps:
                t_in = torch.full((1,), t, device=device)

                if is_i2v:
                    video_noise[:, :1] = latents_images_t  # keep first frame clean

                # cond
                pred_vid_pos, pred_audio_pos = fusion_model(
                    vid=[video_noise], audio=[audio_noise], t=t_in,
                    vid_context=[text_embeddings_pos], audio_context=[text_embeddings_pos],
                    vid_seq_len=max_seq_len_video, audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=is_i2v
                )
                # uncond
                pred_vid_neg, pred_audio_neg = fusion_model(
                    vid=[video_noise], audio=[audio_noise], t=t_in,
                    vid_context=[text_embeddings_video_neg], audio_context=[text_embeddings_audio_neg],
                    vid_seq_len=max_seq_len_video, audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=is_i2v, slg_layer=slg_layer
                )

                # cfg
                pv = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                pa = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                # steps
                video_noise = scheduler_video.step(pv.unsqueeze(0), t, video_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)
                audio_noise = scheduler_audio.step(pa.unsqueeze(0), t, audio_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)
                pbar.update(1)

        if cpu_offload:
            fusion_model = fusion_model.to(offload_device); torch.cuda.empty_cache()

        # decode
        wan_vae.model = wan_vae.model.to(device)
        mmaudio_vae = mmaudio_vae.to(device)

        if is_i2v:
            video_noise[:, :1] = latents_images_t

        vae_scale = _get_vae_scale(wan_vae)
        latents_for_decode = video_noise / vae_scale

        with torch.no_grad():
            decoded = wan_vae.single_decode(latents_for_decode, device=device, pbar=False)
        video = _ensure_cf_hw_video(decoded).to(torch.float32)
        vmin, vmax = video.min(), video.max()
        if (vmin >= 0.0) and (vmax <= 1.0):
            pass
        elif (vmin >= -1.1) and (vmax <= 1.1):
            video = (video + 1.0) * 0.5
        else:
            video = (video - vmin) / (vmax - vmin + 1e-8)
        video = video.clamp(0.0, 1.0).cpu()
        frames = video.permute(1, 2, 3, 0).contiguous().to(torch.float32).cpu()

        audio_latents = audio_noise.unsqueeze(0).transpose(1, 2)  # [1, C, L]
        with torch.no_grad():
            wav = mmaudio_vae.wrapped_decode(audio_latents)
        wav = wav.squeeze().cpu().float().numpy()
        audio_dict = {"waveform": torch.from_numpy(wav).unsqueeze(0).unsqueeze(0), "sample_rate": 16000}

        return (frames, audio_dict)

NODE_CLASS_MAPPINGS = {
    "OviMMAudioVAELoader": OviMMAudioVAELoader,
    "OviFusionModelLoader": OviFusionModelLoader,
    "OviSampler": OviSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OviMMAudioVAELoader": "Ovi MMAudio VAE Loader",
    "OviFusionModelLoader": "Ovi Fusion Model Loader",
    "OviSampler": "Ovi Sampler",
}
