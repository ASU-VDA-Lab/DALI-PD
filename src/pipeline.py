#BSD 3-Clause License
#
#Copyright (c) 2025, ASU-VDA-Lab
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


###########################################################################
# References:
# https://github.com/huggingface/diffusers/
###########################################################################
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.doc_utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.training_utils import EMAModel
from src import EDAUNet, VAE, PinMacroControlNet
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy


import matplotlib.pyplot as plt
import os


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

# ============================
# Helpers for non-overlapping macro sampling (guaranteed termination)
# ============================
def _random_space_partition(operate_x0: int, operate_y0: int, operate_x1: int, operate_y1: int,
                            min_leaf_area: int, target_leaf_count: int, max_splits: int = 10000,
                            margin_px: int = 0):
    """
    Randomly partition the working area into disjoint leaf rectangles using guillotine/KD-style splits.
    All rectangles are axis-aligned and represented with integer coordinates in half-open form (x1,y1 exclusive).

    Returns a list of (x0, y0, x1, y1) leaves. Guaranteed to terminate.
    """
    leaves = [(operate_x0, operate_y0, operate_x1, operate_y1)]

    def rect_area(r):
        return max(0, r[2] - r[0]) * max(0, r[3] - r[1])

    splits = 0
    rng = np.random.default_rng()
    while len(leaves) < target_leaf_count and splits < max_splits:
        # pick a splittable leaf
        idxs = list(range(len(leaves)))
        rng.shuffle(idxs)
        split_done = False
        for idx in idxs:
            x0, y0, x1, y1 = leaves[idx]
            W = x1 - x0
            H = y1 - y0
            if W <= 1 or H <= 1:
                continue

            # choose direction heuristically but randomly
            if W > H:
                # prefer vertical
                prefer_vertical = True if rng.random() < 0.7 else False
            else:
                prefer_vertical = True if rng.random() < 0.3 else False

            def try_vertical():
                # ensure both sides >= min_leaf_area
                if H <= 0:
                    return None
                k = int(np.ceil(min_leaf_area / max(1, H)))
                # account for margin gap between leaves
                if W - (2 * k + margin_px) <= 0:
                    return None
                # split coordinate must leave room for k on both sides plus the margin gap
                m_left = margin_px // 2
                m_right = margin_px - m_left
                cx_min = x0 + k + m_left
                cx_max = x1 - k - m_right
                if cx_min >= cx_max:
                    return None
                cx = rng.integers(cx_min, cx_max)
                left = (x0, y0, int(cx) - m_left, y1)
                right = (int(cx) + m_right, y0, x1, y1)
                if rect_area(left) >= min_leaf_area and rect_area(right) >= min_leaf_area:
                    return left, right
                return None

            def try_horizontal():
                if W <= 0:
                    return None
                k = int(np.ceil(min_leaf_area / max(1, W)))
                if H - (2 * k + margin_px) <= 0:
                    return None
                m_top = margin_px // 2
                m_bottom = margin_px - m_top
                cy_min = y0 + k + m_top
                cy_max = y1 - k - m_bottom
                if cy_min >= cy_max:
                    return None
                cy = rng.integers(cy_min, cy_max)
                top = (x0, y0, x1, int(cy) - m_top)
                bottom = (x0, int(cy) + m_bottom, x1, y1)
                if rect_area(top) >= min_leaf_area and rect_area(bottom) >= min_leaf_area:
                    return top, bottom
                return None

            result = None
            if prefer_vertical:
                result = try_vertical() or try_horizontal()
            else:
                result = try_horizontal() or try_vertical()

            if result is not None:
                a, b = result
                leaves[idx] = a
                leaves.append(b)
                splits += 1
                split_done = True
                break

        if not split_done:
            break

    return leaves


def _allocate_rectangles_in_leaves(leaves, macros: int, macro_area: float,
                                   min_area: int = 300, max_area: int = 10000,
                                   aspect_ratio_max: float = 5.0, min_edge_px: int = 1,
                                   max_width_px: int | None = None, max_height_px: int | None = None):
    """
    Allocate up to `macros` rectangles, one per leaf, within the given leaves without overlap.
    Each rectangle respects area bounds and aspect ratio, and fits within its leaf.

    Returns a list of boxes [(x0,y0,x1,y1)] with integer coordinates (x1,y1 exclusive).
    """
    rng = np.random.default_rng()

    def rect_area(r):
        return max(0, r[2] - r[0]) * max(0, r[3] - r[1])

    eligible = []
    for r in leaves:
        if rect_area(r) < min_area:
            continue
        lw = r[2] - r[0]
        lh = r[3] - r[1]
        if min(lw, lh) < min_edge_px:
            continue
        eligible.append(r)
    if len(eligible) == 0:
        return []

    if len(eligible) >= macros:
        chosen = list(eligible)
        rng.shuffle(chosen)
        chosen = chosen[:macros]
    else:
        # not enough leaves; take what we have
        chosen = sorted(eligible, key=rect_area, reverse=True)[:macros]

    # Dirichlet weights for randomness but cap by constraints per leaf
    alpha = np.ones(len(chosen), dtype=float)
    weights = rng.dirichlet(alpha)
    target_areas = (weights * macro_area).tolist()

    boxes = []
    for i, leaf in enumerate(chosen):
        x0, y0, x1, y1 = leaf
        W = x1 - x0
        H = y1 - y0
        leaf_cap = max(1, int(0.95 * W * H))
        Ai = int(np.clip(target_areas[i], min_area, min(max_area, leaf_cap)))

        # sample aspect ratio log-uniform in [1/aspect_ratio_max, aspect_ratio_max]
        log_r = rng.uniform(np.log(1.0 / aspect_ratio_max), np.log(aspect_ratio_max))
        r = float(np.exp(log_r))

        # compute initial w,h from area and ratio, enforce min edge
        w = max(min_edge_px, int(np.sqrt(Ai * r)))
        h = max(min_edge_px, int(np.ceil(Ai / max(1, w))))

        # fit global max dims if provided
        if max_width_px is not None:
            w = min(w, max_width_px)
        if max_height_px is not None:
            h = min(h, max_height_px)

        # fit into leaf by scaling down if necessary
        if w > W or h > H:
            sx = W / max(1, w)
            sy = H / max(1, h)
            s = min(sx, sy)
            w = max(1, int(np.floor(w * s)))
            h = max(1, int(np.floor(h * s)))

        # constrain aspect ratio to <= aspect_ratio_max
        if max(w, h) / max(1, min(w, h)) > aspect_ratio_max:
            if w > h:
                w = int(min(max(w, min_edge_px), aspect_ratio_max * h))
            else:
                h = int(min(max(h, min_edge_px), aspect_ratio_max * w))

        # final guardrails: ensure within leaf
        w = min(max(min_edge_px, w), W)
        h = min(max(min_edge_px, h), H)
        if max_width_px is not None:
            w = min(w, max_width_px)
        if max_height_px is not None:
            h = min(h, max_height_px)

        # sample position inside leaf
        if W - w > 0:
            px = int(rng.integers(x0, x1 - w))
        else:
            px = x0
        if H - h > 0:
            py = int(rng.integers(y0, y1 - h))
        else:
            py = y0

        bx0 = px
        by0 = py
        bx1 = px + w
        by1 = py + h
        boxes.append([bx0, by0, bx1, by1])

    return boxes

class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`VAE`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: VAE,
        unet: Union[UNet2DConditionModel, EDAUNet, EMAModel],
        pin_macro_controlnet: Union[PinMacroControlNet, EMAModel],
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        """
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        """
        self.register_modules(
            vae=vae,
            unet=unet,
            pin_macro_controlnet=pin_macro_controlnet,
            scheduler=scheduler
        )
        block_out_channels = [
            128,
            256,
            512,
            512
        ]
        self.vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        clock_period,
        utilization,
        macros,
        pins,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
        """

        model_dtype = self.unet.shadow_params[0].dtype if isinstance(self.unet, EMAModel) else self.unet.dtype
        prompt = []
        for i in range(len(macros)):
            prompt.append([utilization, clock_period, macros[i][0], macros[i][1], macros[i][2], macros[i][3]])
        if len(prompt) < 100:
            for _ in range(100 - len(prompt)):
                prompt.append([utilization, clock_period, 0, 0, 0, 0])
        prompt = np.array(prompt)
        bbox_embeds = torch.from_numpy(prompt[:, 2:]).to(dtype=torch.float32, device=device)
        para_embeds = torch.from_numpy(prompt[:, :2]).to(dtype=torch.float32, device=device)
        bbox_embeds = bbox_embeds[None, ...]
        para_embeds = para_embeds[None, ...]

        bs_embed, seq_len, _ = bbox_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bbox_embeds = bbox_embeds.repeat(1, num_images_per_prompt, 1)
        bbox_embeds = bbox_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        para_embeds = para_embeds.repeat(1, num_images_per_prompt, 1)
        para_embeds = para_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # process pins
        temp_pin_list = list()
        pin_list = list()
        for side in range(4):
            temp_pin_list.append(list())
        pin_side_count = [len(pins[0][0]), len(pins[1][0]), len(pins[2][0]), len(pins[3][0])]
        #print(pin_side_count)
        for side in range(4):
            for i in range(pin_side_count[side]):
                temp_pin_list[side].append([pins[side][0][i][0], pins[side][0][i][1], pins[side][0][i][2]])
            if pin_side_count[side] == 0:
                temp_pin_list[side] = [[0.0, 0.0, 0.0]]
        for side in range(4):
            temp_pin_tensor = torch.from_numpy(np.array(temp_pin_list[side])).to(dtype=torch.float32, device=device)[None, ...]
            temp_pin_tensor = temp_pin_tensor.repeat(1, num_images_per_prompt, 1)
            temp_pin_tensor = temp_pin_tensor.view(bs_embed * num_images_per_prompt, pin_side_count[side], -1)
            pin_list.append(temp_pin_tensor)
        del temp_pin_list

        if do_classifier_free_guidance:
            negative_bbox = []
            negative_para = []
            for _ in range(100):
                negative_bbox.append([0, 0, 0, 0])
                negative_para.append([0.0, 0.0])
            negative_bbox_embeds = torch.tensor(
                negative_bbox,
                dtype=model_dtype,
                device=device
            )[None, ...]
            negative_para_embeds = torch.tensor(
                negative_para,
                dtype=model_dtype,
                device=device
            )[None, ...]
            negative_bbox_embeds = negative_bbox_embeds.repeat(1, num_images_per_prompt, 1)
            negative_bbox_embeds = negative_bbox_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            negative_para_embeds = negative_para_embeds.repeat(1, num_images_per_prompt, 1)
            negative_para_embeds = negative_para_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            bbox_embeds = torch.cat([bbox_embeds, negative_bbox_embeds], dim=0)
            para_embeds = torch.cat([para_embeds, negative_para_embeds], dim=0)

            # process pins
            final_pin_list = list()
            for side in range(4):
                temp_pin_list = list()
                for i in range(pin_side_count[side]):
                    temp_pin_list.append([0, 0, 0])
                if pin_side_count[side] == 0:
                    temp_pin_list = [[0.0, 0.0, 0.0]]
                negative_pins = torch.from_numpy(np.array(temp_pin_list)).to(dtype=torch.float32, device=device)[None, ...]
                negative_pins = negative_pins.repeat(1, num_images_per_prompt, 1)
                negative_pins = negative_pins.view(bs_embed * num_images_per_prompt, pin_side_count[side], -1)
                final_pin_list.append(torch.cat([pin_list[side], negative_pins], dim=0))

            pin_list = final_pin_list

        return bbox_embeds, para_embeds, pin_list

    def decode_latents(self, latents):
        image = self.vae.decoder(latents)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        utilization,
        clock_period,
        macros,
        callback_steps,
    ):
        if utilization <= 0 or utilization > 1:
            raise ValueError(f"`utilization` has to be greater than 0 and less than or equal to 1 but is {utilization}.")
        if clock_period <= 0:
            raise ValueError(f"`clock_period (ns)` has to be greater than 0 but is {clock_period}.")
        
        if isinstance(macros, list):
            for macro in macros:
                if len(macro) != 4:
                    raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers but is {macro}.")
                else:
                    for i in range(4):
                        if not isinstance(macro[i], float) and not isinstance(macro[i], np.float64):
                            raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers but is {macro}.")
        elif isinstance(macros, int):
            if macros <= 0:
                raise ValueError(f"`macros` has to be greater than or equal to 0 but is {macros}.")
        elif isinstance(macros, np.ndarray):
            for macro in macros:
                if len(macro) != 4:
                    raise ValueError(f"`macros` has to be a list of lists with 4 floating numbers but is {macro}.")
                else:
                    for i in range(4):
                        if not isinstance(macro[i], float) and not isinstance(macro[i], np.float64):
                            raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers but is {macro}. with type {type(macro[i])}")
        else:
            raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers or an integer but is {macros}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        height_ = height
        width_ = width
        if height_ % 8 != 0:
            height_ = height_ + height_ % 8
        if width_ % 8 != 0:
            width_ = width_ + width_ % 8

        latent_height = int(height_ // self.vae_scale_factor)
        latent_width = int(width_ // self.vae_scale_factor)
            
        shape = (int(batch_size), int(num_channels_latents), int(latent_height), int(latent_width))

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            #torch.manual_seed(100)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_control_latents(self, batch_size, height, width, macros, pins, dtype, device, do_classifier_free_guidance):
        height_ = height
        width_ = width
        if height_ % 8 != 0:
            height_ = height_ + height_ % 8
        if width_ % 8 != 0:
            width_ = width_ + width_ % 8

        latent_height = int(height_)
        latent_width = int(width_)
        macro_numpy = np.zeros((latent_height, latent_width))
        pin_numpy = np.zeros((latent_height, latent_width))

        #assign macro blocks
        for macro_count in range(len(macros)):
            x0, y0, x1, y1 = macros[macro_count]
            x0 = int(x0 * (latent_width - 1))
            y0 = int(y0 * (latent_height - 1))
            x1 = int(x1 * (latent_width - 1))
            y1 = int(y1 * (latent_height - 1))
            for x in range(x0, x1+1):
                for y in range(y0, y1+1):
                    macro_numpy[y, x] = 1

        #assign pins
        for side in range(4):
            for pins_ in pins[side][0]:
                x, y = pins_[1], pins_[2]
                x = int(x * (latent_width - 1))
                y = int(y * (latent_height - 1))
                pin_numpy[y, x] += 1

        pin_numpy /= pin_numpy.max()
        
        macro_numpy = macro_numpy[None, ...]
        pin_numpy = pin_numpy[None, ...]
        control_net_numpy = np.concatenate((macro_numpy, pin_numpy), axis=0)[None, ...]
        if do_classifier_free_guidance:
            empty_control_net_numpy = np.zeros((1, 2, latent_height, latent_width))
            control_net_numpy = np.concatenate((control_net_numpy, empty_control_net_numpy), axis=0)

        if batch_size > 1:
            control_net_numpy = control_net_numpy.repeat(batch_size, axis=0)

        control_net_numpy = torch.from_numpy(control_net_numpy).to(dtype=dtype, device=device)
        return control_net_numpy

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        clock_period: Optional[float] = None,
        utilization: Optional[float] = None,
        macros: Optional[int] = None,
        pins: Optional[float] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "numpy",
        return_split_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_scale: Optional[float] = 1.0,
        index: Optional[int] = 0,
        restriction: bool = False,
        circuitnet_data_count: Optional[int] = -1,
        name: Optional[str] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_split_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a dictionary with the generated images split into their respective components.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        do_classifier_free_guidance = True if guidance_scale > 1.0 else False
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            utilization, clock_period, macros, callback_steps
        )

        if isinstance(macros, int):
            macros = self.sample_macro_bounding_boxes(macros, height, width, utilization)
        if isinstance(pins[0], int):
            pins = self.sample_pins(pins)
        # 2. Define call parameters
        batch_size = 1

        device = self.unet.shadow_params[0].device if isinstance(self.unet, EMAModel) else self.unet.device

        # 3. Encode input prompt
        bbox_embeds, para_embeds, pin_list = self._encode_prompt(
            clock_period = clock_period,
            utilization = utilization,
            macros = macros,
            pins = pins,
            device = device,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = do_classifier_free_guidance
        )

        not_meet_break_condition = True
        iterations = 0
        while not_meet_break_condition:
            latents = None
            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            in_channels = 4
            num_channels_latents = in_channels
            
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                bbox_embeds.dtype,
                device,
                generator,
                latents,
            )

            control_net_embeds = self.prepare_control_latents(
                batch_size = batch_size,
                height = height,
                width = width,
                macros = macros,
                pins = pin_list,
                dtype = bbox_embeds.dtype,
                device = device,
                do_classifier_free_guidance = do_classifier_free_guidance,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # predict the noise residual
                    control_residuals = self.pin_macro_controlnet(
                        control_net_embeds,
                        t,
                        bbox_tensor=bbox_embeds,
                        pin_tensor_list=pin_list,
                    )

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        bbox_tensor=bbox_embeds,
                        para_tensor=para_embeds,
                        mid_block_additional_residual=control_residuals[0],
                        up_block_additional_residuals=control_residuals[1:],
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
            iterations += 1
            noisefig = latents.clone().detach().cpu().numpy()[0]

            if output_type == "numpy":
                # 8. Post-processing
                image = self.decode_latents(latents)
            else:
                image = latents

            if not return_split_dict:
                return image

            height = int(height)
            width = int(width)
            return_dict = {
                "cell_density": [np.array(image[i][0][:height, :width]) for i in range(len(image))],
                "macro_region": [np.array(image[i][1][:height, :width]) for i in range(len(image))],
                "RUDY": [np.array(image[i][2][:height, :width]) for i in range(len(image))],
                "IR_drop": [np.array(image[i][3][:height, :width]) for i in range(len(image))],
                "power_all": [np.array(image[i][4][:height, :width]) for i in range(len(image))],
                "power_sca": [np.array(image[i][5][:height, :width]) for i in range(len(image))]
            }

            for image_key, image_value in return_dict.items():
                for i in range(len(image_value)): # batch size 
                    return_dict[image_key][i] = self.shift_value_to_zero(image_value[i], rgb=False)

            # Post-processing
            for i in range(len(return_dict["macro_region"])): # batch size 
                return_dict["macro_region"][i], return_dict["cell_density"][i] = self.macro_cell_post_processing(return_dict["macro_region"][i],return_dict["cell_density"][i])
                return_dict["IR_drop"][i], return_dict["power_all"][i], return_dict["power_sca"][i] = self.power_IR_post_processing(
                    return_dict["IR_drop"][i], return_dict["power_all"][i], return_dict["power_sca"][i], return_dict["macro_region"][i])
                return_dict["cell_density"][i], return_dict["power_all"][i], return_dict["power_sca"][i] = self.power_calibration(
                    return_dict["cell_density"][i], return_dict["power_all"][i], return_dict["power_sca"][i], return_dict["macro_region"][i])
                        
            not_meet_break_condition = self.checker(return_dict, utilization, height, width)             
            
            if restriction:
                if iterations > 25:
                    not_meet_break_condition = False

        return return_dict, iterations

    def checker(self, return_dict, utilization, height, width):
        for i in range(len(return_dict["cell_density"])): # batch size
            # use IR_drop > eps_ir to define IR region, and do closing + dilation
            # to eliminate noise small holes and expand one circle to cover conservatively
            if np.sum(np.where((return_dict["cell_density"][i] > 0) & (return_dict["macro_region"][i] == 0), 1, 0)) >= \
                utilization * 0.8 * ((height - 30) * (width - 30) - np.sum(np.where(return_dict["macro_region"][i] == 1, 1, 0))):
                if np.sum(np.where((return_dict["cell_density"][i] > 0) & (return_dict["macro_region"][i] == 0), 1, 0)) <= \
                utilization * 1.25 * ((height - 30) * (width - 30) - np.sum(np.where(return_dict["macro_region"][i] == 1, 1, 0))):
                    if np.sum(np.where(return_dict["power_all"][i] > 0, 1, 0)) <= np.sum(np.where(return_dict["IR_drop"][i] > 0, 1, 0)):
                        if np.sum(np.where(return_dict["power_all"][i] > 0, 1, 0)) * 1.25 >= np.sum(np.where(return_dict["IR_drop"][i] > 0, 1, 0)):
                            if np.sum(return_dict["macro_region"][i]) >= 0.1 * (height - 30) * (width - 30):
                                return False
        return True

    def find_best_cut(
        self,
        labels,
        index,
        x0,
        y0,
        x1,
        y1,
        util_based_split: bool = True
    ):
        """Helper function for finding the best cut point, optimized with Numba."""
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        max_area_diff = -1.0 # Use float for Numba compatibility if needed
        cut_x, cut_y = -1, -1
        direction = 0 # 0: none, 1: vertical, 2: horizontal

        if util_based_split:
            component_mask = (labels == index)

            for x in range(x0 + 1, x1):
                # Vertical cut simulation
                first_part_mask_v = component_mask.copy()
                first_part_mask_v[:, x:] = False
                second_part_mask_v = component_mask.copy()
                second_part_mask_v[:, :x] = False

                first_coords_v = np.where(first_part_mask_v)
                second_coords_v = np.where(second_part_mask_v)

                if first_coords_v[0].size > 0 and second_coords_v[0].size > 0:
                    first_part_area_x0 = np.min(first_coords_v[1])
                    first_part_area_x1 = np.max(first_coords_v[1])
                    first_part_area_y0 = np.min(first_coords_v[0])
                    first_part_area_y1 = np.max(first_coords_v[0])
                    second_part_area_x0 = np.min(second_coords_v[1])
                    second_part_area_x1 = np.max(second_coords_v[1])
                    second_part_area_y0 = np.min(second_coords_v[0])
                    second_part_area_y1 = np.max(second_coords_v[0])

                    # Ensure positive dimensions before calculating area
                    fp_h = first_part_area_y1 - first_part_area_y0 + 1
                    fp_w = first_part_area_x1 - first_part_area_x0 + 1
                    sp_h = second_part_area_y1 - second_part_area_y0 + 1
                    sp_w = second_part_area_x1 - second_part_area_x0 + 1

                    if fp_h > 0 and fp_w > 0 and sp_h > 0 and sp_w > 0:
                        first_part_area_bbox = fp_w * fp_h
                        second_part_area_bbox = sp_w * sp_h
                        area_diff = abs(first_part_area_bbox + second_part_area_bbox - bbox_area)
                        if area_diff > max_area_diff:
                            max_area_diff = area_diff
                            cut_x = x
                            cut_y = -1 # Mark y as unused for vertical
                            direction = 1 # vertical

            for y in range(y0 + 1, y1):
                # Horizontal cut simulation
                first_part_mask_h = component_mask.copy()
                first_part_mask_h[y:, :] = False
                second_part_mask_h = component_mask.copy()
                second_part_mask_h[:y, :] = False

                first_coords_h = np.where(first_part_mask_h)
                second_coords_h = np.where(second_part_mask_h)

                if first_coords_h[0].size > 0 and second_coords_h[0].size > 0:
                    first_part_area_x0 = np.min(first_coords_h[1])
                    first_part_area_x1 = np.max(first_coords_h[1])
                    first_part_area_y0 = np.min(first_coords_h[0])
                    first_part_area_y1 = np.max(first_coords_h[0])
                    second_part_area_x0 = np.min(second_coords_h[1])
                    second_part_area_x1 = np.max(second_coords_h[1])
                    second_part_area_y0 = np.min(second_coords_h[0])
                    second_part_area_y1 = np.max(second_coords_h[0])

                    # Ensure positive dimensions
                    fp_h = first_part_area_y1 - first_part_area_y0 + 1
                    fp_w = first_part_area_x1 - first_part_area_x0 + 1
                    sp_h = second_part_area_y1 - second_part_area_y0 + 1
                    sp_w = second_part_area_x1 - second_part_area_x0 + 1

                    if fp_h > 0 and fp_w > 0 and sp_h > 0 and sp_w > 0:
                        first_part_area_bbox = fp_w * fp_h
                        second_part_area_bbox = sp_w * sp_h
                        area_diff = abs(first_part_area_bbox + second_part_area_bbox - bbox_area)
                        if area_diff > max_area_diff:
                            max_area_diff = area_diff
                            cut_x = -1 # Mark x as unused for horizontal
                            cut_y = y
                            direction = 2 # horizontal

        else:
            cut_x = (x0 + x1) // 2
            cut_y = (y0 + y1) // 2
            direction = 0
            if abs(abs(x1 - cut_x + 1) - abs(y1 - y0 + 1)) <= abs(abs(x1 - x0 + 1) - abs(y1 - cut_y + 1)):
                cut_y = -1
                direction = 1
            else:
                cut_x = -1
                direction = 2

        return cut_x, cut_y, direction

    def macro_cell_post_processing(
        self,
        macro_region: np.ndarray,
        cell_density: np.ndarray,
    ):
        if np.sum(macro_region) == 0:
            return macro_region, cell_density
        macro_img = macro_region * 255
        macro_img = macro_img.astype(np.uint8)
        macro_img = cv2.resize(macro_img, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)


        erosion_kernel_1 = np.ones((3, 3), np.uint8)
        erosion_kernel_2 = np.ones((5, 5), np.uint8)
        dilation_kernel_1 = np.ones((5, 5), np.uint8)
        dilation_kernel_2 = np.ones((3, 3), np.uint8)

        macro_img = cv2.erode(macro_img, erosion_kernel_1, iterations=1)
        macro_img = cv2.dilate(macro_img, dilation_kernel_1, iterations=1)
        macro_img = cv2.erode(macro_img, erosion_kernel_2, iterations=1)
        macro_img = cv2.dilate(macro_img, dilation_kernel_2, iterations=1)

        macro_img = cv2.resize(
            macro_img,
            (macro_region.shape[1], macro_region.shape[0]),
            interpolation=cv2.INTER_AREA
        )

        erosion_kernel = np.ones((3, 3), np.uint8)
        dilation_kernel_1 = np.ones((5, 5), np.uint8)
        macro_img = cv2.erode(macro_img, erosion_kernel, iterations=1)

        macro_img = macro_img / 255
        macro_img[macro_img < 0.9] = 0.0
        macro_img[macro_img >= 0.9] = 1.0
        macro_img = macro_img * 255
        macro_img = macro_img.astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(macro_img)
        index = 1
        constraint_area = min(labels.shape[0], labels.shape[1]) ** 2
        while index < num_labels:
            
            coords = np.where(labels == index)
            if coords[0].size > 0:
                x0 = np.min(coords[1])
                y0 = np.min(coords[0])
                x1 = np.max(coords[1])
                y1 = np.max(coords[0])
                bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                if (np.sum(np.where(labels == index, 1, 0)) / bbox_area) > 0.9:
                    macro_img[y0:y1+1, x0:x1+1] = 255
                    labels[y0:y1+1, x0:x1+1] = index
                    index += 1
                elif (np.sum(np.where(labels == index, 1, 0)) / constraint_area) < 0.04:
                    macro_img[y0:y1+1, x0:x1+1] = 255
                    labels[y0:y1+1, x0:x1+1] = index
                    index += 1
                else:
                    cut_x, cut_y, direction = self.find_best_cut(labels, index, x0, y0, x1, y1, True)
                    if direction == 0:
                        cut_x, cut_y, direction = self.find_best_cut(labels, index, x0, y0, x1, y1, False)
                    new_label_id = num_labels
                    component_mask_coords = np.where(labels == index)
                    if direction == 1:
                        mask_to_change = (component_mask_coords[1] >= cut_x)
                        coords_to_change = (component_mask_coords[0][mask_to_change], component_mask_coords[1][mask_to_change])
                        labels[coords_to_change] = new_label_id
                        num_labels += 1
                    elif direction == 2:
                        mask_to_change = (component_mask_coords[0] >= cut_y)
                        coords_to_change = (component_mask_coords[0][mask_to_change], component_mask_coords[1][mask_to_change])
                        labels[coords_to_change] = new_label_id
                        num_labels += 1

            else:
                index += 1
        macro_img = cv2.dilate(macro_img, dilation_kernel_1, iterations=1)
        macro_img = macro_img/255
        macro_img[macro_img < 0.9] = 0.0
        macro_img[macro_img >= 0.9] = 1.0
        
        cell_img = deepcopy(cell_density)
        cell_img = cell_img * 255
        cell_img = cell_img * macro_img
        if np.sum(cell_img) == 0:
            return macro_img, cell_density
        cell_img = cell_img.astype(np.int64)
        cell_img_temp = deepcopy(cell_img)
        macro_img_most_frequent_value = np.bincount(cell_img_temp.flatten())[1:].argmax() + 1
        # from int to float
        macro_img_filter = np.where(cell_img == macro_img_most_frequent_value, 1, 0)
        macro_pixel_amount = np.sum(macro_img_filter)
        macro_pixel_value = np.sum(cell_img[macro_img_filter == 1]) / macro_pixel_amount
        macro_pixel_value = macro_pixel_value / 255

        cell_img = deepcopy(cell_density)
        cell_img[macro_img == 1.0] = 0.08333333333333331 # macro_pixel_value
        return macro_img, cell_img

    def power_IR_post_processing(
        self,
        IR_drop: np.ndarray,
        power_all: np.ndarray,
        power_sca: np.ndarray,
        macro_region: np.ndarray
    ):
        # get smaller macro filter
        if np.sum(macro_region) == 0:
            return IR_drop, power_all, power_sca
        macro_img_small = deepcopy(macro_region) 
        macro_img_small = macro_img_small * 255
        macro_img_small = macro_img_small.astype(np.uint8)
        dilation_kernel = np.ones((3, 3), np.uint8)
        macro_img_small = cv2.erode(macro_img_small, dilation_kernel, iterations=1)
        macro_img_small = macro_img_small / 255
        macro_img_small[macro_img_small < 0.9] = 0.0
        macro_img_small[macro_img_small >= 0.9] = 1.0
        
        IR_drop_img = deepcopy(IR_drop)
        IR_drop_img1 = IR_drop_img * macro_img_small
        max_value = np.max(IR_drop_img1)
        filter = np.where(IR_drop_img <= max_value, 1, 0)
        IR_drop_img[filter == 1] = 0
        # power_all
        power_all_img = deepcopy(power_all)
        power_all_img1 = power_all_img * macro_img_small
        max_value = np.max(power_all_img1)
        filter = np.where(power_all_img <= max_value, 1, 0)
        power_all_img[filter == 1] = 0
        # power_sca
        power_sca_img = deepcopy(power_sca)
        power_sca_img1 = power_sca_img * macro_img_small
        max_value = np.max(power_sca_img1)
        filter = np.where(power_sca_img <= max_value, 1, 0)
        power_sca_img[filter == 1] = 0
            
        return IR_drop_img, power_all_img, power_sca_img

    def power_calibration(
        self,
        cell_density: np.ndarray,
        power_all: np.ndarray,
        power_sca: np.ndarray,
        macro_region: np.ndarray
    ):

        intersection = np.where((power_all > 0) & (power_sca > 0) & (cell_density > 0), 1, 0)

        power_all = power_all * intersection
        power_sca = power_sca * intersection

        union = np.where((intersection == 1) | (macro_region == 1), 1, 0)

        cell_density = cell_density * union
        return cell_density, power_all, power_sca

    def shift_value_to_zero(
        self,
        image: np.ndarray,
        rgb: bool = False,
    ):
        # check the min value of the 2d np array, if it is not zero, then get the delta of it, and shift all the values in the array by the delta so the min is zero
        min_value = np.min(image)
        if min_value != 0:
            delta = -min_value
            image = image + delta
        
        # normalize values between 0 and 1 if max value > 1
        max_value = np.max(image)
        if rgb:
            if max_value > 255:
                image = image / max_value
            else:
                image = image / 255.0
        else:
            if max_value > 1:
                image = image / max_value
            
        return image

    def sample_macro_bounding_boxes(
        self,
        macros: int,
        height: int,
        width: int,
        utilization: float,
    ):
        # Operating region with 15px inset
        operate_x0, operate_y0 = 15, 15
        operate_x1, operate_y1 = width - 15, height - 15

        # Target total macro area
        macro_area = float(utilization) * float((height - 30) * (width - 30)) * 0.9

        # Partition space into sufficient leaves (no overlap by construction)
        min_leaf_area = 160  # slightly larger than min macro area to leave margin
        leaves = _random_space_partition(operate_x0, operate_y0, operate_x1, operate_y1,
                                         min_leaf_area=min_leaf_area,
                                         target_leaf_count=max(macros * 2, macros + 1),
                                         margin_px=5)

        # Allocate rectangles within leaves (non-overlapping)
        boxes = _allocate_rectangles_in_leaves(
            leaves,
            macros=macros,
            macro_area=macro_area,
            min_area=2500,
            max_area=10000,
            aspect_ratio_max=3.0,
            min_edge_px=50,
            max_width_px=int(0.45 * width),
            max_height_px=int(0.45 * height),
        )

        # Fallback: if still not enough (extremely rare), relax constraints minimally
        if len(boxes) < macros:
            leaves = _random_space_partition(operate_x0, operate_y0, operate_x1, operate_y1,
                                             min_leaf_area=100,
                                             target_leaf_count=max(macros * 3, macros + 2),
                                             margin_px=5)
            boxes = _allocate_rectangles_in_leaves(
                leaves,
                macros=macros,
                macro_area=macro_area,
                min_area=2500,
                max_area=12000,
                aspect_ratio_max=3.0,
                min_edge_px=50,
                max_width_px=int(0.45 * width),
                max_height_px=int(0.45 * height),
            )

        # Normalize to [0,1]
        bbox_list = []
        for i in range(min(macros, len(boxes))):
            x0, y0, x1, y1 = boxes[i]
            bbox_list.append([x0 / width, y0 / height, x1 / width, y1 / height])
        return bbox_list


    def sample_pins(
        self,
        pins: list,
    ):
        pin_list = []
        top_list = []
        left_list = []
        bottom_list = []
        right_list = []
        
        for i in range(4):
            pin_amount = pins[i]
            match i:
                case 0: # top
                    for _ in range(pin_amount):
                        x = np.clip(np.random.normal(0.445, 0.15), 0.0004, 0.89)
                        top_list.append([pin_amount, x, 0.89])
                    if pin_amount == 0:
                        top_list = [[0.0, 0.0, 0.0]]
                case 1: # left
                    for _ in range(pin_amount):
                        y = np.clip(np.random.normal(0.445, 0.15), 0.0004, 0.89)
                        left_list.append([pin_amount, 0.0004, y])
                    if pin_amount == 0:
                        left_list = [[0.0, 0.0, 0.0]]
                case 2: # bottom
                    for _ in range(pin_amount):
                        x = np.clip(np.random.normal(0.445, 0.15), 0.0004, 0.89)
                        bottom_list.append([pin_amount, x, 0.0004])
                    if pin_amount == 0:
                        bottom_list = [[0.0, 0.0, 0.0]]
                case 3: # right
                    for _ in range(pin_amount):
                        y = np.clip(np.random.normal(0.445, 0.15), 0.0004, 0.89)
                        right_list.append([pin_amount, 0.89, y])
                    if pin_amount == 0:
                        right_list = [[0.0, 0.0, 0.0]]
        top_list = np.array(top_list)[None, ...]
        left_list = np.array(left_list)[None, ...]
        bottom_list = np.array(bottom_list)[None, ...]
        right_list = np.array(right_list)[None, ...]
        pin_list.append(top_list)
        pin_list.append(left_list)
        pin_list.append(bottom_list)
        pin_list.append(right_list)
        return pin_list











class StableDiffusionPipeline_TestUNet(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`VAE`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: VAE,
        unet: Union[UNet2DConditionModel, EDAUNet, EMAModel],
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        """
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        """
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler
        )
        block_out_channels = [
            128,
            256,
            512,
            512
        ]
        self.vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        clock_period,
        utilization,
        macros,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
        """

        model_dtype = self.unet.shadow_params[0].dtype if isinstance(self.unet, EMAModel) else self.unet.dtype
        prompt = []
        for i in range(len(macros)):
            prompt.append([utilization, clock_period, macros[i][0], macros[i][1], macros[i][2], macros[i][3]])
        if len(prompt) < 100:
            for _ in range(100 - len(prompt)):
                prompt.append([utilization, clock_period, 0, 0, 0, 0])
        prompt = np.array(prompt)
        bbox_embeds = torch.from_numpy(prompt[:, 2:]).to(dtype=torch.float32, device=device)
        para_embeds = torch.from_numpy(prompt[:, :2]).to(dtype=torch.float32, device=device)
        bbox_embeds = bbox_embeds[None, ...]
        para_embeds = para_embeds[None, ...]

        bs_embed, seq_len, _ = bbox_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bbox_embeds = bbox_embeds.repeat(1, num_images_per_prompt, 1)
        bbox_embeds = bbox_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        para_embeds = para_embeds.repeat(1, num_images_per_prompt, 1)
        para_embeds = para_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_bbox = []
            negative_para = []
            for _ in range(100):
                negative_bbox.append([0, 0, 0, 0])
                negative_para.append([0.0, 0.0])
            negative_bbox_embeds = torch.tensor(
                negative_bbox,
                dtype=model_dtype,
                device=device
            )[None, ...]
            negative_para_embeds = torch.tensor(
                negative_para,
                dtype=model_dtype,
                device=device
            )[None, ...]
            negative_bbox_embeds = negative_bbox_embeds.repeat(1, num_images_per_prompt, 1)
            negative_bbox_embeds = negative_bbox_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            negative_para_embeds = negative_para_embeds.repeat(1, num_images_per_prompt, 1)
            negative_para_embeds = negative_para_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            bbox_embeds = torch.cat([negative_bbox_embeds, bbox_embeds], dim=0)
            para_embeds = torch.cat([negative_para_embeds, para_embeds], dim=0)

        return bbox_embeds, para_embeds

    def decode_latents(self, latents):
        image = self.vae.decoder(latents)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        utilization,
        clock_period,
        macros,
        callback_steps,
    ):
        if utilization <= 0 or utilization > 1:
            raise ValueError(f"`utilization` has to be greater than 0 and less than or equal to 1 but is {utilization}.")
        if clock_period <= 0:
            raise ValueError(f"`clock_period (ns)` has to be greater than 0 but is {clock_period}.")
        
        if isinstance(macros, list):
            for macro in macros:
                if len(macro) != 4:
                    raise ValueError(f"`macros` has to be a list of lists with 4 floating numbers but is {macro}.")
                else:
                    for i in range(4):
                        if not isinstance(macro[i], float) and not isinstance(macro[i], np.float64):
                            raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers but is {macro}. with type {type(macro[i])}")
        elif isinstance(macros, int):
            if macros <= 0:
                raise ValueError(f"`macros` has to be greater than or equal to 0 but is {macros}.")
        elif isinstance(macros, np.ndarray):
            for macro in macros:
                if len(macro) != 4:
                    raise ValueError(f"`macros` has to be a list of lists with 4 floating numbers but is {macro}.")
                else:
                    for i in range(4):
                        if not isinstance(macro[i], float) and not isinstance(macro[i], np.float64):
                            raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers but is {macro}. with type {type(macro[i])}")
        else:
            raise ValueError(f"`macros` has to be a list of tuples with 4 floating numbers or an integer but is {macros}. with type {type(macros)}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        height_ = height
        width_ = width
        if height_ % 8 != 0:
            height_ = height_ + height_ % 8
        if width_ % 8 != 0:
            width_ = width_ + width_ % 8

        latent_height = int(height_ // self.vae_scale_factor)
        latent_width = int(width_ // self.vae_scale_factor)
            
        shape = (int(batch_size), int(num_channels_latents), int(latent_height), int(latent_width))

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            #torch.manual_seed(100)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        clock_period: Optional[float] = None,
        utilization: Optional[float] = None,
        macros: Optional[int] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "numpy",
        return_split_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_scale: Optional[float] = 1.0,
        index: Optional[int] = 0,
        restriction: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_split_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a dictionary with the generated images split into their respective components.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        do_classifier_free_guidance = True if guidance_scale > 1.0 else False
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            utilization, clock_period, macros, callback_steps
        )

        if isinstance(macros, int):
            macros = self.sample_macro_bounding_boxes(macros, height, width, utilization)
        # 2. Define call parameters
        batch_size = 1

        device = self.unet.shadow_params[0].device if isinstance(self.unet, EMAModel) else self.unet.device
        print(f"Device: {device}")

        # 3. Encode input prompt
        bbox_embeds, para_embeds = self._encode_prompt(
            clock_period = clock_period,
            utilization = utilization,
            macros = macros,
            device = device,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = do_classifier_free_guidance
        )

        not_meet_break_condition = True
        iterations = 0
        while not_meet_break_condition:
            latents = None
            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            in_channels = 4
            num_channels_latents = in_channels
            
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                bbox_embeds.dtype,
                device,
                generator,
                latents,
            )
            
            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        bbox_tensor=bbox_embeds,
                        para_tensor=para_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
            iterations += 1
            noisefig = latents.clone().detach().cpu().numpy()[0]

            if output_type == "numpy":
                # 8. Post-processing
                image = self.decode_latents(latents)
            else:
                image = latents

            if not return_split_dict:
                return image

            height = int(height)
            width = int(width)
            return_dict = {
                "cell_density": [np.array(image[i][0][:height, :width]) for i in range(len(image))],
                "macro_region": [np.array(image[i][1][:height, :width]) for i in range(len(image))],
                "RUDY": [np.array(image[i][2][:height, :width]) for i in range(len(image))],
                "IR_drop": [np.array(image[i][3][:height, :width]) for i in range(len(image))],
                "power_all": [np.array(image[i][4][:height, :width]) for i in range(len(image))],
                "power_sca": [np.array(image[i][5][:height, :width]) for i in range(len(image))]
            }

            for image_key, image_value in return_dict.items():
                for i in range(len(image_value)): # batch size 
                    return_dict[image_key][i] = self.shift_value_to_zero(image_value[i], rgb=False)

            # Post-processing
            for i in range(len(return_dict["macro_region"])): # batch size 
                return_dict["macro_region"][i], return_dict["cell_density"][i] = self.macro_cell_post_processing(return_dict["macro_region"][i],return_dict["cell_density"][i])
                return_dict["IR_drop"][i], return_dict["power_all"][i], return_dict["power_sca"][i] = self.power_IR_post_processing(
                    return_dict["IR_drop"][i], return_dict["power_all"][i], return_dict["power_sca"][i], return_dict["macro_region"][i])
                return_dict["cell_density"][i], return_dict["power_all"][i], return_dict["power_sca"][i] = self.power_calibration(
                    return_dict["cell_density"][i], return_dict["power_all"][i], return_dict["power_sca"][i], return_dict["macro_region"][i])
                        
            not_meet_break_condition = self.checker(return_dict, utilization, height, width)             

            if restriction:
                if iterations > 25:
                    not_meet_break_condition = False

        return return_dict, iterations

    def checker(self, return_dict, utilization, height, width):
        for i in range(len(return_dict["cell_density"])): # batch size 
            if np.sum(np.where((return_dict["cell_density"][i] > 0) & (return_dict["macro_region"][i] == 0), 1, 0)) >= \
                utilization * 0.8 * ((height - 30) * (width - 30) - np.sum(np.where(return_dict["macro_region"][i] == 1, 1, 0))):
                if np.sum(np.where((return_dict["cell_density"][i] > 0) & (return_dict["macro_region"][i] == 0), 1, 0)) <= \
                utilization * 1.25 * ((height - 30) * (width - 30) - np.sum(np.where(return_dict["macro_region"][i] == 1, 1, 0))):
                    if np.sum(np.where(return_dict["power_all"][i] > 0, 1, 0)) <= np.sum(np.where(return_dict["IR_drop"][i] > 0, 1, 0)):
                        if np.sum(np.where(return_dict["power_all"][i] > 0, 1, 0)) * 1.25 >= np.sum(np.where(return_dict["IR_drop"][i] > 0, 1, 0)):
                            if np.sum(return_dict["macro_region"][i]) >= 0.1 * (height - 30) * (width - 30):
                                return False
        return True

    def find_best_cut(
        self,
        labels,
        index,
        x0,
        y0,
        x1,
        y1,
        util_based_split: bool = True
    ):
        """Helper function for finding the best cut point, optimized with Numba."""
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        max_area_diff = -1.0 # Use float for Numba compatibility if needed
        cut_x, cut_y = -1, -1
        direction = 0 # 0: none, 1: vertical, 2: horizontal

        if util_based_split:
            component_mask = (labels == index)

            for x in range(x0 + 1, x1):
                # Vertical cut simulation
                first_part_mask_v = component_mask.copy()
                first_part_mask_v[:, x:] = False
                second_part_mask_v = component_mask.copy()
                second_part_mask_v[:, :x] = False

                first_coords_v = np.where(first_part_mask_v)
                second_coords_v = np.where(second_part_mask_v)

                if first_coords_v[0].size > 0 and second_coords_v[0].size > 0:
                    first_part_area_x0 = np.min(first_coords_v[1])
                    first_part_area_x1 = np.max(first_coords_v[1])
                    first_part_area_y0 = np.min(first_coords_v[0])
                    first_part_area_y1 = np.max(first_coords_v[0])
                    second_part_area_x0 = np.min(second_coords_v[1])
                    second_part_area_x1 = np.max(second_coords_v[1])
                    second_part_area_y0 = np.min(second_coords_v[0])
                    second_part_area_y1 = np.max(second_coords_v[0])

                    # Ensure positive dimensions before calculating area
                    fp_h = first_part_area_y1 - first_part_area_y0 + 1
                    fp_w = first_part_area_x1 - first_part_area_x0 + 1
                    sp_h = second_part_area_y1 - second_part_area_y0 + 1
                    sp_w = second_part_area_x1 - second_part_area_x0 + 1

                    if fp_h > 0 and fp_w > 0 and sp_h > 0 and sp_w > 0:
                        first_part_area_bbox = fp_w * fp_h
                        second_part_area_bbox = sp_w * sp_h
                        area_diff = abs(first_part_area_bbox + second_part_area_bbox - bbox_area)
                        if area_diff > max_area_diff:
                            max_area_diff = area_diff
                            cut_x = x
                            cut_y = -1 # Mark y as unused for vertical
                            direction = 1 # vertical

            for y in range(y0 + 1, y1):
                # Horizontal cut simulation
                first_part_mask_h = component_mask.copy()
                first_part_mask_h[y:, :] = False
                second_part_mask_h = component_mask.copy()
                second_part_mask_h[:y, :] = False

                first_coords_h = np.where(first_part_mask_h)
                second_coords_h = np.where(second_part_mask_h)

                if first_coords_h[0].size > 0 and second_coords_h[0].size > 0:
                    first_part_area_x0 = np.min(first_coords_h[1])
                    first_part_area_x1 = np.max(first_coords_h[1])
                    first_part_area_y0 = np.min(first_coords_h[0])
                    first_part_area_y1 = np.max(first_coords_h[0])
                    second_part_area_x0 = np.min(second_coords_h[1])
                    second_part_area_x1 = np.max(second_coords_h[1])
                    second_part_area_y0 = np.min(second_coords_h[0])
                    second_part_area_y1 = np.max(second_coords_h[0])

                    # Ensure positive dimensions
                    fp_h = first_part_area_y1 - first_part_area_y0 + 1
                    fp_w = first_part_area_x1 - first_part_area_x0 + 1
                    sp_h = second_part_area_y1 - second_part_area_y0 + 1
                    sp_w = second_part_area_x1 - second_part_area_x0 + 1

                    if fp_h > 0 and fp_w > 0 and sp_h > 0 and sp_w > 0:
                        first_part_area_bbox = fp_w * fp_h
                        second_part_area_bbox = sp_w * sp_h
                        area_diff = abs(first_part_area_bbox + second_part_area_bbox - bbox_area)
                        if area_diff > max_area_diff:
                            max_area_diff = area_diff
                            cut_x = -1 # Mark x as unused for horizontal
                            cut_y = y
                            direction = 2 # horizontal

        else:
            cut_x = (x0 + x1) // 2
            cut_y = (y0 + y1) // 2
            direction = 0
            if abs(abs(x1 - cut_x + 1) - abs(y1 - y0 + 1)) <= abs(abs(x1 - x0 + 1) - abs(y1 - cut_y + 1)):
                cut_y = -1
                direction = 1
            else:
                cut_x = -1
                direction = 2

        return cut_x, cut_y, direction

    def macro_cell_post_processing(
        self,
        macro_region: np.ndarray,
        cell_density: np.ndarray,
    ):
        if np.sum(macro_region) == 0:
            return macro_region, cell_density
        macro_img = macro_region * 255
        macro_img = macro_img.astype(np.uint8)
        macro_img = cv2.resize(macro_img, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)


        erosion_kernel_1 = np.ones((3, 3), np.uint8)
        erosion_kernel_2 = np.ones((5, 5), np.uint8)
        dilation_kernel_1 = np.ones((5, 5), np.uint8)
        dilation_kernel_2 = np.ones((3, 3), np.uint8)

        macro_img = cv2.erode(macro_img, erosion_kernel_1, iterations=1)
        macro_img = cv2.dilate(macro_img, dilation_kernel_1, iterations=1)
        macro_img = cv2.erode(macro_img, erosion_kernel_2, iterations=1)
        macro_img = cv2.dilate(macro_img, dilation_kernel_2, iterations=1)

        macro_img = cv2.resize(
            macro_img,
            (macro_region.shape[1], macro_region.shape[0]),
            interpolation=cv2.INTER_AREA
        )

        erosion_kernel = np.ones((3, 3), np.uint8)
        dilation_kernel_1 = np.ones((5, 5), np.uint8)
        macro_img = cv2.erode(macro_img, erosion_kernel, iterations=1)

        macro_img = macro_img / 255
        macro_img[macro_img < 0.9] = 0.0
        macro_img[macro_img >= 0.9] = 1.0
        macro_img = macro_img * 255
        macro_img = macro_img.astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(macro_img)
        index = 1
        constraint_area = min(labels.shape[0], labels.shape[1]) ** 2
        while index < num_labels:
            
            coords = np.where(labels == index)
            if coords[0].size > 0:
                x0 = np.min(coords[1])
                y0 = np.min(coords[0])
                x1 = np.max(coords[1])
                y1 = np.max(coords[0])
                bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                if (np.sum(np.where(labels == index, 1, 0)) / bbox_area) > 0.9:
                    macro_img[y0:y1+1, x0:x1+1] = 255
                    labels[y0:y1+1, x0:x1+1] = index
                    index += 1
                elif (np.sum(np.where(labels == index, 1, 0)) / constraint_area) < 0.04:
                    macro_img[y0:y1+1, x0:x1+1] = 255
                    labels[y0:y1+1, x0:x1+1] = index
                    index += 1
                else:
                    cut_x, cut_y, direction = self.find_best_cut(labels, index, x0, y0, x1, y1, True)
                    if direction == 0:
                        cut_x, cut_y, direction = self.find_best_cut(labels, index, x0, y0, x1, y1, False)
                    new_label_id = num_labels
                    component_mask_coords = np.where(labels == index)
                    if direction == 1:
                        mask_to_change = (component_mask_coords[1] >= cut_x)
                        coords_to_change = (component_mask_coords[0][mask_to_change], component_mask_coords[1][mask_to_change])
                        labels[coords_to_change] = new_label_id
                        num_labels += 1
                    elif direction == 2:
                        mask_to_change = (component_mask_coords[0] >= cut_y)
                        coords_to_change = (component_mask_coords[0][mask_to_change], component_mask_coords[1][mask_to_change])
                        labels[coords_to_change] = new_label_id
                        num_labels += 1

            else:
                index += 1
        macro_img = cv2.dilate(macro_img, dilation_kernel_1, iterations=1)
        macro_img = macro_img/255
        macro_img[macro_img < 0.9] = 0.0
        macro_img[macro_img >= 0.9] = 1.0
        
        cell_img = deepcopy(cell_density)
        cell_img = cell_img * 255
        cell_img = cell_img * macro_img
        if np.sum(cell_img) == 0:
            return macro_img, cell_density
        cell_img = cell_img.astype(np.int64)
        cell_img_temp = deepcopy(cell_img)
        macro_img_most_frequent_value = np.bincount(cell_img_temp.flatten())[1:].argmax() + 1
        # from int to float
        macro_img_filter = np.where(cell_img == macro_img_most_frequent_value, 1, 0)
        macro_pixel_amount = np.sum(macro_img_filter)
        macro_pixel_value = np.sum(cell_img[macro_img_filter == 1]) / macro_pixel_amount
        macro_pixel_value = macro_pixel_value / 255

        cell_img = deepcopy(cell_density)
        cell_img[macro_img == 1.0] = 0.08333333333333331 # macro_pixel_value
        return macro_img, cell_img

    def power_IR_post_processing(
        self,
        IR_drop: np.ndarray,
        power_all: np.ndarray,
        power_sca: np.ndarray,
        macro_region: np.ndarray
    ):
        # get smaller macro filter
        if np.sum(macro_region) == 0:
            return IR_drop, power_all, power_sca
        macro_img_small = deepcopy(macro_region) 
        macro_img_small = macro_img_small * 255
        macro_img_small = macro_img_small.astype(np.uint8)
        dilation_kernel = np.ones((3, 3), np.uint8)
        macro_img_small = cv2.erode(macro_img_small, dilation_kernel, iterations=1)
        macro_img_small = macro_img_small / 255
        macro_img_small[macro_img_small < 0.9] = 0.0
        macro_img_small[macro_img_small >= 0.9] = 1.0
        
        IR_drop_img = deepcopy(IR_drop)
        IR_drop_img1 = IR_drop_img * macro_img_small
        max_value = np.max(IR_drop_img1)
        filter = np.where(IR_drop_img <= max_value, 1, 0)
        IR_drop_img[filter == 1] = 0
        # power_all
        power_all_img = deepcopy(power_all)
        power_all_img1 = power_all_img * macro_img_small
        max_value = np.max(power_all_img1)
        filter = np.where(power_all_img <= max_value, 1, 0)
        power_all_img[filter == 1] = 0
        # power_sca
        power_sca_img = deepcopy(power_sca)
        power_sca_img1 = power_sca_img * macro_img_small
        max_value = np.max(power_sca_img1)
        filter = np.where(power_sca_img <= max_value, 1, 0)
        power_sca_img[filter == 1] = 0
            
        return IR_drop_img, power_all_img, power_sca_img

    def power_calibration(
        self,
        cell_density: np.ndarray,
        power_all: np.ndarray,
        power_sca: np.ndarray,
        macro_region: np.ndarray
    ):

        intersection = np.where((power_all > 0) & (power_sca > 0) & (cell_density > 0), 1, 0)

        power_all = power_all * intersection
        power_sca = power_sca * intersection

        union = np.where((intersection == 1) | (macro_region == 1), 1, 0)

        cell_density = cell_density * union
        return cell_density, power_all, power_sca

    def shift_value_to_zero(
        self,
        image: np.ndarray,
        rgb: bool = False,
    ):
        # check the min value of the 2d np array, if it is not zero, then get the delta of it, and shift all the values in the array by the delta so the min is zero
        min_value = np.min(image)
        if min_value != 0:
            delta = -min_value
            image = image + delta
        
        # normalize values between 0 and 1 if max value > 1
        max_value = np.max(image)
        if rgb:
            if max_value > 255:
                image = image / max_value
            else:
                image = image / 255.0
        else:
            if max_value > 1:
                image = image / max_value
            
        return image

    def sample_macro_bounding_boxes(
        self,
        macros: int,
        height: int,
        width: int,
        utilization: float,
    ):
        # Operating region with 15px inset
        operate_x0, operate_y0 = 15, 15
        operate_x1, operate_y1 = width - 15, height - 15

        # Target total macro area
        macro_area = float(utilization) * float((height - 30) * (width - 30)) * 0.9

        # Partition space into sufficient leaves (no overlap by construction)
        min_leaf_area = 160
        leaves = _random_space_partition(operate_x0, operate_y0, operate_x1, operate_y1,
                                         min_leaf_area=min_leaf_area,
                                         target_leaf_count=max(macros * 2, macros + 1),
                                         margin_px=5)

        # Allocate rectangles within leaves (non-overlapping)
        boxes = _allocate_rectangles_in_leaves(
            leaves,
            macros=macros,
            macro_area=macro_area,
            min_area=2500,
            max_area=10000,
            aspect_ratio_max=3.0,
            min_edge_px=50,
            max_width_px=int(0.45 * width),
            max_height_px=int(0.45 * height),
        )

        # Fallback
        if len(boxes) < macros:
            leaves = _random_space_partition(operate_x0, operate_y0, operate_x1, operate_y1,
                                             min_leaf_area=100,
                                             target_leaf_count=max(macros * 3, macros + 2),
                                             margin_px=5)
            boxes = _allocate_rectangles_in_leaves(
                leaves,
                macros=macros,
                macro_area=macro_area,
                min_area=2500,
                max_area=12000,
                aspect_ratio_max=3.0,
                min_edge_px=50,
                max_width_px=int(0.45 * width),
                max_height_px=int(0.45 * height),
            )

        # Normalize to [0,1]
        bbox_list = []
        for i in range(min(macros, len(boxes))):
            x0, y0, x1, y1 = boxes[i]
            bbox_list.append([x0 / width, y0 / height, x1 / width, y1 / height])
        return bbox_list

