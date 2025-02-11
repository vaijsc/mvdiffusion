#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import argparse
import gc
import glob
import logging
import math
import os
import random
import shutil
from pathlib import Path

from itertools import chain
import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F_vision
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import torchvision

from mvdream.unet import UNet2DConditionModel as MVDreamUNet2DConditionModel
from mvdream.camera_proj import CameraMatrixEmbedding
from mvdream.attention_processor import CrossViewAttnProcessor as MVD_CrossViewAttnProcessor
from mvdream.attention_processor import XFormersCrossViewAttnProcessor as MVD_XFormersCrossViewAttnProcessor
from mvdream.utils import get_camera
from mvdream.pipeline_mvdream import set_self_attn_processor

from einops import rearrange, repeat

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__)


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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        nargs="+",
        help=("The path to a text file containing all training prompts"),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompts"
            " `args.validation_prompts` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swiftbrush-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for unet.",
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for lora teacher.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="The classifier-free guidance scale.")
    parser.add_argument("--guidance_scale_sd", type=float, default=7.5, help="The classifier-free guidance for sd.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help=("The alpha constant of the LoRA update matrices."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_dir is None:
        raise ValueError("Need a training folder.")

    return args


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

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
def inference(vae, unet, noise_scheduler, encoded_embeds, generator, device, weight_dtype, camera_matrix_embeds):
    input_shape = (4, 4, args.resolution // 8, args.resolution // 8)
    input_noise = torch.randn(input_shape, generator=generator, device=device, dtype=weight_dtype)

    prompt_embed = encoded_embeds["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)
    
    prompt_embed = repeat(prompt_embed, "bs n_token c -> (bs nview) n_token c", nview=4)
    camera_embeds = repeat(camera_matrix_embeds, "nview c -> (bs nview) c", bs=1)

    pred_original_sample = predict_original(unet, noise_scheduler, input_noise, prompt_embed, camera_embeds)
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
    image = rearrange(image, "(bs nview) c h w -> bs c h (nview w)", nview=4)
    image = (image[0].detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return image


def predict_original(unet, noise_scheduler, input_noise, prompt_embeds, camera_embeds):
    max_timesteps = torch.ones((input_noise.shape[0],), dtype=torch.int64, device=input_noise.device)
    max_timesteps = max_timesteps * (noise_scheduler.config.num_train_timesteps - 1)

    alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047) ** 0.5
    model_pred = unet(input_noise, max_timesteps, prompt_embeds, camera_embeds).sample

    latents = (input_noise - sigma_T * model_pred) / alpha_T
    return latents


class PromptDataset(Dataset):
    def __init__(self, train_data_dir):
        self.train_data_paths = []
        for data_dir in train_data_dir:
            self.train_data_paths += list(glob.glob(data_dir + "/*.npy"))
        print(f"TOTAL DATA LENGTH: {len(self.train_data_paths)}")

    def __len__(self):
        return len(self.train_data_paths)

    def __getitem__(self, index):
        data = {"prompt_embeds": torch.from_numpy(np.load(self.train_data_paths[index], allow_pickle=True))}
        return data

    def shuffle(self, *args, **kwargs):
        random.shuffle(self.train_data_paths)
        return self

    def select(self, selected_range):
        self.train_data_paths = [self.train_data_paths[idx] for idx in selected_range]
        return self


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    visualize_dir = os.path.join(logging_dir, "images")
    os.makedirs(visualize_dir, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    camera_proj = CameraMatrixEmbedding.from_pretrained(args.pretrained_model_name_or_path, subfolder="camera_proj", revision=args.revision)
    
    mv_teacher_lora = MVDreamUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    sd_teacher_lora = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base" , subfolder="unet", revision=args.revision)

    # UNet use cross-view attention
    def set_unet_self_attn_cross_view_processor(unet_model, num_views=4):
        attn_procs_cls = MVD_XFormersCrossViewAttnProcessor if is_xformers_available() else MVD_CrossViewAttnProcessor
        set_self_attn_processor(unet_model, attn_procs_cls(num_views=num_views))

    # Set up LORA
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v"],
    )
    mv_teacher_lora.add_adapter(config)
    sd_teacher_lora.add_adapter(config)

    set_unet_self_attn_cross_view_processor(unet_model=mv_teacher_lora, num_views=4)

    # Freeze vae, text encoders and teacher.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    camera_proj.requires_grad_(False)

    # Set unet and lora teacher as trainable.
    mv_teacher_lora.train()
    sd_teacher_lora.train()

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    camera_proj.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            mv_teacher_lora.enable_xformers_memory_efficient_attention()
            sd_teacher_lora.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if args.gradient_checkpointing:
        mv_teacher_lora.enable_gradient_checkpointing()
        sd_teacher_lora.enable_gradient_checkpointing()



    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.learning_rate_lora = (
            args.learning_rate_lora
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.Adam8bit
        optimizer_lora_class = bnb.optim.Adam8bit
    else:
        optimizer_class = torch.optim.Adam
        optimizer_lora_class = torch.optim.Adam

    # Optimizer creation
    # Init particles
    image_path = "/workspace/env/data/mvdream-diffusers/output_mvdream/a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes_4.png"
    image = torchvision.io.read_image(image_path) /255.0 * 2 -1 
    image = rearrange(image, "c h  (nview w) -> nview c h w", nview=4)
    image = image.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        rgb_latent = vae.encode(image).latent_dist.sample()
        rgb_latent = rgb_latent * vae.config.scaling_factor

        # latent = rgb_latent / vae.config.scaling_factor
        # rgb = vae.decode(latent).sample.float()
        # torchvision.utils.save_image(rgb * 0.5 + 0.5, "test.png")

    # particles = torch.nn.functional.interpolate(rgb_latent.detach().clone(), scale_factor=2, mode="bilinear")
    # particles.requires_grad = True
    # particles_to_optimize = [particles]

    input_shape = (args.train_batch_size * 4, 4, args.resolution // 8, args.resolution // 8)
    particles = torch.randn(*input_shape, dtype=weight_dtype, device=accelerator.device)
    particles.requires_grad = True
    particles_to_optimize = [particles]

    
    optimizer = optimizer_class(
        particles_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    #random camera
    with torch.no_grad():
        elevation_cam = 10
        azimuth_start_cam = 30
        cam_dist = 1.0
        c2ws = get_camera(4, elevation=elevation_cam, azimuth_start=azimuth_start_cam, cam_dist=cam_dist).to(accelerator.device, dtype=camera_proj.dtype)
        camera_matrix_embeds = camera_proj(c2ws.flatten(1, 2)) # 4 x 1280 = nview x dim
        camera_matrix_embeds = camera_matrix_embeds.to(accelerator.device, dtype=weight_dtype)

    optimizer_mv_lora = optimizer_lora_class(
        mv_teacher_lora.parameters(),
        lr=args.learning_rate_lora,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_sd_lora = optimizer_lora_class(
        sd_teacher_lora.parameters(),
        lr=args.learning_rate_lora,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if type(args.validation_prompts) is not list:
        args.validation_prompts = [args.validation_prompts]

    # Get null-text embedding
    null_dict = encode_prompt([""], text_encoder, tokenizer)
    validation_dicts = [encode_prompt([prompt], text_encoder, tokenizer) for prompt in args.validation_prompts]
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(8000 / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    lr_scheduler_mv_lora = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_mv_lora,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler_sd_lora = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sd_lora,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    particles = accelerator.prepare(particles)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    mv_teacher_lora, lr_scheduler_mv_lora, optimizer_mv_lora  = accelerator.prepare(mv_teacher_lora, lr_scheduler_mv_lora, optimizer_mv_lora)
    sd_teacher_lora, lr_scheduler_sd_lora, optimizer_sd_lora  = accelerator.prepare(sd_teacher_lora, lr_scheduler_sd_lora, optimizer_sd_lora)

   
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(8000 / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("train_data_dir")
        accelerator.init_trackers("swiftbrush", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Get alphas cummulative product
    alphas_cumprod = noise_scheduler.alphas_cumprod
    alphas_cumprod = alphas_cumprod.to(accelerator.device, dtype=weight_dtype)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss_vsd = train_loss_lora = 0.0
        for step in range(80000):

            if step % 3 == 0:
                mode = "mv"
            else:
                mode = "image"

            with accelerator.accumulate(particles, mv_teacher_lora, sd_teacher_lora):
                batch = validation_dicts[0]
                bsz = batch["prompt_embeds"].shape[0]

                if mode == "mv":
                    # enable cross view attention
                    set_unet_self_attn_cross_view_processor(unet_model=mv_teacher_lora, num_views=4)
                    ##########################

                    # Get the text embeddings
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                    prompt_null_embeds = (
                        null_dict["prompt_embeds"].repeat(bsz, 1, 1).to(accelerator.device, dtype=weight_dtype)
                    )

                    # Get predicted original samples from unet
                    prompt_embeds = repeat(prompt_embeds, "bs n_token c -> (bs nview) n_token c", nview=4)
                    prompt_null_embeds = repeat(prompt_null_embeds, "bs n_token c -> (bs nview) n_token c", nview=4)
                    camera_embeds = repeat(camera_matrix_embeds, "nview c -> (bs nview) c", bs=bsz)

                    up_pred_original_samples = particles
                    pred_original_samples = torch.nn.functional.interpolate(up_pred_original_samples, scale_factor=0.5, mode="bilinear")

                    # Sample noise that we'll add to the predicted original samples
                    noise = torch.randn_like(pred_original_samples)

                    # Sample a random timestep for each image
                    timesteps_range = torch.tensor([0.02, 0.981]) * noise_scheduler.config.num_train_timesteps
                    timesteps = torch.randint(*timesteps_range.long(), (bsz,), device=accelerator.device).long()             # bsz * nview
                    timesteps = repeat(timesteps, "bs -> (bs nview)", nview=4)
                    

                    # Add noise to the predicted original samples according to the noise magnitude at each timestep
                    noisy_samples = noise_scheduler.add_noise(pred_original_samples, noise, timesteps)

                    # Prepare outputs from the teacher
                    with torch.no_grad():
                        accelerator.unwrap_model(mv_teacher_lora).disable_adapters()
                        teacher_pred_cond = mv_teacher_lora(noisy_samples, timesteps, prompt_embeds, camera_embeds).sample
                        teacher_pred_uncond = mv_teacher_lora(noisy_samples, timesteps, prompt_null_embeds, camera_embeds).sample

                        accelerator.unwrap_model(mv_teacher_lora).enable_adapters()
                        lora_pred_cond = mv_teacher_lora(noisy_samples, timesteps, prompt_embeds, camera_embeds).sample

                        # Apply classifier-free guidance to the teacher prediction
                        teacher_pred = teacher_pred_uncond + args.guidance_scale * (
                            teacher_pred_cond - teacher_pred_uncond
                        )
                        lora_pred = lora_pred_cond

                    # Compute the score gradient for updating the model
                    sigma_t = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)
                    score_gradient = torch.nan_to_num(sigma_t**2 * (teacher_pred - lora_pred))

                    # Compute the VSD loss for the model
                    target = (pred_original_samples - score_gradient).detach()
                    loss_vsd = 0.5 * F.mse_loss(pred_original_samples.float(), target.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss_vsd = accelerator.gather(loss_vsd.repeat(args.train_batch_size)).mean()
                    train_loss_vsd += avg_loss_vsd.item() / args.gradient_accumulation_steps

                    # Backpropagate for the unet
                    accelerator.backward(loss_vsd)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # LoRA loss
                    # Sample noise that we'll add to the predicted original samples
                    noise = torch.randn_like(pred_original_samples.detach())

                    # Sample a random timestep for each image
                    timesteps_range = torch.tensor([0, 1]) * noise_scheduler.config.num_train_timesteps
                    timesteps = torch.randint(*timesteps_range.long(), (bsz,), device=accelerator.device).long()
                    timesteps = repeat(timesteps, "bs -> (bs nview)", nview=4)

                    # Add noise to the predicted original samples according to the noise magnitude at each timestep
                    noisy_samples = noise_scheduler.add_noise(pred_original_samples.detach(), noise, timesteps)

                    # Compute output for updating the LoRA teacher
                    encoder_hidden_states = prompt_null_embeds if random.random() < 0.1 else prompt_embeds
                    lora_pred = mv_teacher_lora(noisy_samples, timesteps, encoder_hidden_states, camera_embeds).sample

                    alpha_t = (alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
                    lora_pred = alpha_t * lora_pred
                    target = alpha_t * noise

                    # Compute the loss for LoRA teacher
                    loss_lora = F.mse_loss(lora_pred.float(), target.float(), reduction="mean") 

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss_lora = accelerator.gather(loss_lora.repeat(args.train_batch_size)).mean()
                    train_loss_lora += avg_loss_lora.item() / args.gradient_accumulation_steps

                    # Backpropagate for the LoRA teacher
                    accelerator.backward(loss_lora)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(mv_teacher_lora.parameters(), args.max_grad_norm)
                    optimizer_mv_lora.step()
                    lr_scheduler_mv_lora.step()
                    optimizer_mv_lora.zero_grad(set_to_none=args.set_grads_to_none)

                else:

                    # Get the text embeddings
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                    prompt_null_embeds = (
                        null_dict["prompt_embeds"].repeat(bsz, 1, 1).to(accelerator.device, dtype=weight_dtype)
                    )

                    # Get predicted original samples from unet
                    prompt_embeds = repeat(prompt_embeds, "bs n_token c -> (bs nview) n_token c", nview=4)
                    prompt_null_embeds = repeat(prompt_null_embeds, "bs n_token c -> (bs nview) n_token c", nview=4)
                    camera_embeds = repeat(camera_matrix_embeds, "nview c -> (bs nview) c", bs=bsz)

                    pred_original_samples = particles
                    # Sample noise that we'll add to the predicted original samples
                    noise = torch.randn_like(pred_original_samples)

                    # Sample a random timestep for each image
                    timesteps_range = torch.tensor([0.02, 0.981]) * noise_scheduler.config.num_train_timesteps
                    timesteps = torch.randint(*timesteps_range.long(), (bsz * 4,), device=accelerator.device).long()             # bsz * nview
                    

                    # Add noise to the predicted original samples according to the noise magnitude at each timestep
                    noisy_samples = noise_scheduler.add_noise(pred_original_samples, noise, timesteps)

                    # Prepare outputs from the teacher
                    with torch.no_grad():
                        accelerator.unwrap_model(sd_teacher_lora).disable_adapters()
                        teacher_pred_cond = sd_teacher_lora(noisy_samples, timesteps, prompt_embeds).sample
                        teacher_pred_uncond = sd_teacher_lora(noisy_samples, timesteps, prompt_null_embeds).sample

                        accelerator.unwrap_model(sd_teacher_lora).enable_adapters()
                        lora_pred_cond = sd_teacher_lora(noisy_samples, timesteps, prompt_embeds).sample

                        # Apply classifier-free guidance to the teacher prediction
                        teacher_pred = teacher_pred_uncond + args.guidance_scale * (
                            teacher_pred_cond - teacher_pred_uncond
                        )
                        lora_pred = lora_pred_cond

                    # Compute the score gradient for updating the model
                    sigma_t = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)
                    score_gradient = torch.nan_to_num(sigma_t**2 * (teacher_pred - lora_pred))

                    # Compute the VSD loss for the model
                    target = (pred_original_samples - score_gradient).detach()
                    loss_vsd = 0.5 * F.mse_loss(pred_original_samples.float(), target.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss_vsd = accelerator.gather(loss_vsd.repeat(args.train_batch_size)).mean()
                    train_loss_vsd += avg_loss_vsd.item() / args.gradient_accumulation_steps

                    # Backpropagate for the unet
                    accelerator.backward(loss_vsd)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # LoRA loss
                    # Sample noise that we'll add to the predicted original samples
                    noise = torch.randn_like(pred_original_samples.detach())

                    # Sample a random timestep for each image
                    timesteps_range = torch.tensor([0, 1]) * noise_scheduler.config.num_train_timesteps
                    timesteps = torch.randint(*timesteps_range.long(), (bsz * 4,), device=accelerator.device).long()

                    # Add noise to the predicted original samples according to the noise magnitude at each timestep
                    noisy_samples = noise_scheduler.add_noise(pred_original_samples.detach(), noise, timesteps)

                    # Compute output for updating the LoRA teacher
                    encoder_hidden_states = prompt_null_embeds if random.random() < 0.1 else prompt_embeds
                    lora_pred = sd_teacher_lora(noisy_samples, timesteps, encoder_hidden_states).sample

                    alpha_t = (alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
                    lora_pred = alpha_t * lora_pred
                    target = alpha_t * noise

                    # Compute the loss for LoRA teacher
                    loss_lora = F.mse_loss(lora_pred.float(), target.float(), reduction="mean") 

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss_lora = accelerator.gather(loss_lora.repeat(args.train_batch_size)).mean()
                    train_loss_lora += avg_loss_lora.item() / args.gradient_accumulation_steps

                    # Backpropagate for the LoRA teacher
                    accelerator.backward(loss_lora)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(sd_teacher_lora.parameters(), args.max_grad_norm)
                    optimizer_sd_lora.step()
                    lr_scheduler_sd_lora.step()
                    optimizer_sd_lora.zero_grad(set_to_none=args.set_grads_to_none)

            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss_vsd": train_loss_vsd, "train_loss_lora": train_loss_lora}, step=global_step
                )
                train_loss_vsd = train_loss_lora = 0.0

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        if args.validation_prompts is not None and args.num_validation_images > 0:
                            logger.info(
                                "Running validation... \nGenerating {} images with prompts:\n  {}".format(
                                    args.num_validation_images, "\n  ".join(args.validation_prompts)
                                )
                            )

                            with torch.no_grad():
                                pred_original_sample = particles.detach() / vae.config.scaling_factor
                                image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
                                image = rearrange(image, "(bs nview) c h w -> bs c h (nview w)", nview=4)
                                image = (image[0].detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                                torchvision.utils.save_image(image.unsqueeze(0)/255, os.path.join(visualize_dir, f"step_{global_step}.png"))

                        
            logs = {
                "step_loss_vsd": loss_vsd.detach().item(),
                "step_loss_lora": loss_lora.detach().item(),
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            revision=args.revision,
        )

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config)
        pipeline.save_pretrained(args.output_dir + "/final/")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
