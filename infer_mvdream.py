import argparse

import torch
import os
import random
from torchvision.utils import save_image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.utils.import_utils import is_xformers_available

from mvdream.unet import UNet2DConditionModel as MVDreamUNet2DConditionModel
from mvdream.camera_proj import CameraMatrixEmbedding
from mvdream.attention_processor import CrossViewAttnProcessor as MVD_CrossViewAttnProcessor
from mvdream.attention_processor import XFormersCrossViewAttnProcessor as MVD_XFormersCrossViewAttnProcessor
from mvdream.utils import get_camera
from mvdream.pipeline_mvdream import set_self_attn_processor

from einops import rearrange, repeat
from torchvision.transforms import functional as F


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt):

    captions = [prompt]
    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


@torch.no_grad()
def inference(vae, unet, noise_scheduler, prompt_embed, device, weight_dtype, camera_embeds, image_resolution):
    bsz = prompt_embed.size(0)
    input_shape = (bsz * 4, 4, image_resolution // 8, image_resolution // 8)
    input_noise = torch.randn(input_shape, device=device, dtype=weight_dtype)

    prompt_embed = prompt_embed.to(device, weight_dtype)
    
    prompt_embed = repeat(prompt_embed, "bs n_token c -> (bs nview) n_token c", nview=4)
    camera_embeds = rearrange(camera_embeds, "bs nview c -> (bs nview) c")

    pred_original_sample = predict_original(unet, noise_scheduler, input_noise, prompt_embed, camera_embeds)
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
    image = rearrange(image, "(bs nview) c h w -> bs c h (nview w)", nview=4)
    image = (image.detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return image


def predict_original(unet, noise_scheduler, input_noise, prompt_embeds, camera_embeds):
    max_timesteps = torch.ones((input_noise.shape[0],), dtype=torch.int64, device=input_noise.device)
    max_timesteps = max_timesteps * (noise_scheduler.config.num_train_timesteps - 1)

    alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047) ** 0.5
    model_pred = unet(input_noise, max_timesteps, prompt_embeds, camera_embeds).sample

    latents = (input_noise - sigma_T * model_pred) / alpha_T
    # latents = torch.nn.functional.interpolate(latents, scale_factor=2, mode='bilinear')
    return latents


def set_unet_self_attn_cross_view_processor(unet_model, num_views=4):
    attn_procs_cls = MVD_XFormersCrossViewAttnProcessor if is_xformers_available() else MVD_CrossViewAttnProcessor
    set_self_attn_processor(unet_model, attn_procs_cls(num_views=num_views))


def main(args):
    device, dtype = "cuda", torch.float16
    unet = MVDreamUNet2DConditionModel.from_pretrained(args.checkpoint_path)


    # unet = MVDreamUNet2DConditionModel.from_pretrained("/workspace/SwiftBrush/output_switfbrush_mvdream3/checkpoint-10000/unet_ema")
    set_unet_self_attn_cross_view_processor(unet_model=unet, num_views=4)
    

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None, use_fast=False
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    camera_proj = CameraMatrixEmbedding.from_pretrained(args.pretrained_model_name_or_path, subfolder="camera_proj", revision=None)
    camera_proj = camera_proj.to(device)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    camera_proj.requires_grad_(False)

    vae.to(device, dtype=torch.float32)
    camera_proj.to(device, dtype=torch.float32)
    text_encoder.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.prompt_list is not None:
        with open(args.prompt_list) as f:
            prompts = f.read().splitlines()
    else:
        prompts = [args.prompt]
    
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Process prompts in batches
    batch_size = args.batch_size  # Define batch size in `args`
    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        # Encode prompts in batch
        prompt_embeds = [encode_prompt(tokenizer, text_encoder, p)["prompt_embeds"] for p in batch_prompts]
        prompt_embeds = torch.cat(prompt_embeds).to(device)  # Stack and move to device

        # Generate batched camera matrices
        elevation_cam = torch.randint(0, 30, (len(batch_prompts),), device=device)
        azimuth_start_cam = torch.randint(0, 360, (len(batch_prompts),), device=device)
        cam_dist = 1.0
        c2ws = torch.stack([
                            get_camera(4, elevation=e.item(), azimuth_start=a.item(), cam_dist=cam_dist)
                            for e, a in zip(elevation_cam, azimuth_start_cam)
                        ])
        c2ws = c2ws.to(device, dtype=camera_proj.dtype)

        camera_matrix_embeds_default = camera_proj(c2ws.flatten(-2, -1))  # Batch operation
        camera_matrix_embeds_default = camera_matrix_embeds_default.to(device, dtype=dtype)

        # Perform inference in batch
        images = inference(vae,
                            unet,
                            noise_scheduler,
                            prompt_embeds,
                            device=device,
                            weight_dtype=dtype,
                            camera_embeds=camera_matrix_embeds_default,
                            image_resolution=args.resolution
                            ) / 255

        # Save the images
        for i, prompt in enumerate(batch_prompts):
            sanitized_prompt = prompt.replace("/", "_")[:100]
            if args.four_view:
                save_image(images[i], os.path.join(out_dir, f"{sanitized_prompt}.png"))
            else:
                split_image = rearrange(images[i], "c h (nview w) -> nview c h w", nview=4)
                for view_idx, image in enumerate(split_image):
                    save_image(image, os.path.join(out_dir, f"{sanitized_prompt}_{view_idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of an inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="thuanz123/swiftbrush",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Text prompt used for inference.",
    )
    parser.add_argument(
        "--four_view",
        action="store_true",
        default=False,
        help="Save four view in one image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        required=False,
        help="Random seed used for inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        required=False,
        help="Random seed used for inference.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        required=False,
        help="Image size for inference.",
    )
    parser.add_argument(
        "--prompt_list",
        type=str,
        default=None,
        required=True,
        help="A .txt file containing all prompts used for training.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="The output directory where the model checkpoint was stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swiftbrush-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()
    main(args)
