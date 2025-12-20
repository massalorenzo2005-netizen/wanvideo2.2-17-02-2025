import torch
from ..utils import log
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


class WanVideoLongCatAvatarExtendEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prev_latents": ("LATENT", {"tooltip": "Previous latents to be used to continue generation"}),
                    "audio_embeds": ("MULTITALK_EMBEDS", {"tooltip": "Full length audio embeddings"}),
                    "num_frames": ("INT", {"default": 93, "min": 1, "max": 256, "step": 1, "tooltip": "Number of new frames to generate" }),
                    "overlap": ("INT", {"default": 13, "min": 0, "max": 16, "step": 1, "tooltip": "Number of overlapping frames from previous latents" }),
                    "frames_processed": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Number of frames already processed in the video" }),
                    "if_not_enough_audio": (["pad_with_start", "mirror_from_end"], {"default": "pad_with_start", "tooltip": "What to do if there are not enough frames in pose_images for the window"}),
                },
                "optional": {
                    "ref_latent": ("LATENT", {"default": None, "tooltip": "Reference latent for the first frame (used for consistency)"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, prev_latents, audio_embeds, num_frames, overlap, if_not_enough_audio, frames_processed=0, ref_latent=None):

        new_audio_embed = audio_embeds.copy()

        audio_features = torch.stack(new_audio_embed["audio_features"])
        print("audio_features shape: ", audio_features.shape)
        if audio_features.shape[1] < frames_processed + num_frames:
            deficit = frames_processed + num_frames - audio_features.shape[1]
            if if_not_enough_audio == "pad_with_start":
                pad = audio_features[:, :1].repeat(1, deficit, 1, 1, 1)
                audio_features = torch.cat([audio_features, pad], dim=1)
            elif if_not_enough_audio == "mirror_from_end":
                to_add = audio_features[:, -deficit:, :].flip(dims=[1])
                audio_features = torch.cat([audio_features, to_add], dim=1)
            log.info(f"Not enough audio features, extended from {new_audio_embed['audio_features'].shape[1]} to {audio_features.shape[1]} frames.")

        ref_target_masks = new_audio_embed.get("ref_target_masks", None)
        if ref_target_masks is not None:
            new_audio_embed["ref_target_masks"] = ref_target_masks[:, frames_processed:frames_processed+num_frames, :]

        latent_overlap = (overlap - 1) // 4 + 1
        print("prev_latents shape: ", prev_latents["samples"].shape, "latent_overlap: ", latent_overlap)
        prev_samples = prev_latents["samples"][:, :, -latent_overlap:].clone()

        ref_sample = None
        if ref_latent is not None:
            ref_sample = ref_latent["samples"][0, :, :1].clone()

        log.info(f"Previous latents shape: {prev_samples.shape}, using last {latent_overlap} latent frames for overlap.")

        new_latent_frames = (num_frames - 1) // 4 + 1
        target_shape = (16, new_latent_frames, prev_samples.shape[-2], prev_samples.shape[-1])
        print("target_shape: ", target_shape)

        audio_stride = 2
        indices = torch.arange(2 * 2 + 1) - 2

        if frames_processed == 0:
            audio_start_idx = 0
        else:
            audio_start_idx = (frames_processed - overlap) * audio_stride
        audio_end_idx = audio_start_idx + num_frames * audio_stride

        log.info(f"Extracting audio embeddings from index {audio_start_idx} to {audio_end_idx}")

        #center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        #center_indices = torch.clamp(center_indices, min=0, max=audio_features.shape[0]-1)
        #audio_emb = audio_features[center_indices][None,...]
        audio_embs = []
        for human_idx in range(len(audio_features)):
            center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=audio_features[human_idx].shape[0] - 1)

            audio_emb = audio_features[human_idx][center_indices].unsqueeze(0).to(device)
            audio_embs.append(audio_emb)
        audio_emb = torch.cat(audio_embs, dim=0)

        new_audio_embed["audio_features"] = None
        new_audio_embed["audio_emb_slice"] = audio_emb

        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "extra_latents": [{"samples": prev_samples, "index": 0}],
            "multitalk_embeds": new_audio_embed,
            "longcat_ref_latent": ref_sample,
        }

        return (embeds,)


NODE_CLASS_MAPPINGS = {
    "WanVideoLongCatAvatarExtendEmbeds": WanVideoLongCatAvatarExtendEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLongCatAvatarExtendEmbeds": "WanVideo LongCat Avatar Extend Embeds",
    }
