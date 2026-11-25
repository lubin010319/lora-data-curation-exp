"""Train SDXL UNet LoRA from TOML, or blind-render a matched A/B evaluation."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import tomllib

from PIL import ImageOps
import numpy as np

from curate_lora_dataset import bucket_for, load_image
from demo_data import ROOT

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict

PROMPTS = [
    "sorami, 1girl, purple short hair, yellow hairclip, blue jacket, front-facing portrait",
    "sorami in a garden, short purple hair, yellow hairclip, blue jacket, profile illustration",
    "sorami, short purple hair, yellow hairclip, blue jacket, watercolor bust portrait",
]
SEEDS = [107, 311]
DEVICE = "cuda"


def read_toml(path):
    with (ROOT / path).open("rb") as file:
        return tomllib.load(file)


def dtype_for(settings):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[settings["precision"]]


def load_pipe(settings):
    kwargs = {key: settings[key] for key in ("variant", "revision") if key in settings}
    return StableDiffusionXLPipeline.from_pretrained(settings["base_model"],
                                                     torch_dtype=dtype_for(settings), **kwargs)


def prepare_examples(pipe, settings, data):
    manifest = ROOT / data["manifest"]
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"Dataset {manifest} is empty")
    vae = pipe.vae.to(DEVICE, dtype=torch.float32).eval()
    pipe.text_encoder.to(DEVICE).eval()
    pipe.text_encoder_2.to(DEVICE).eval()
    precision = dtype_for(settings)
    examples = []
    # Cache the frozen VAE/text features on CPU before loading the UNet to GPU.
    with torch.no_grad():
        for row in rows:
            path = manifest.parent / row["file"]
            image = load_image(path)
            target = (settings["resolution"],) * 2
            if data["enable_bucket"]:
                target = bucket_for(image.size, settings["resolution"])
            input_image = ImageOps.fit(image, target)
            pixels = torch.tensor(np.array(input_image), device=DEVICE).permute(2, 0, 1)
            pixels = pixels[None].float() / 127.5 - 1
            latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
            caption = path.with_suffix(data["caption_extension"]).read_text(encoding="utf-8")
            prompt, _, pooled, _ = pipe.encode_prompt(caption, device=DEVICE, do_classifier_free_guidance=False)
            width, height = target
            # Conditioning describes the actual normalized input frame used for
            # both groups. No source coordinates are inferred from staging files.
            time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=precision)
            examples.append((latents[0].to("cpu", dtype=precision), prompt.cpu(), pooled.cpu(), time_ids))
    vae.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    return examples


def train(settings):
    out = ROOT / settings["output"]
    out.mkdir(parents=True, exist_ok=True)
    seed = settings["seed"]
    torch.manual_seed(seed)
    noise_generator = torch.Generator(device=DEVICE).manual_seed(seed)
    example_rng = random.Random(seed)
    pipe = load_pipe(settings)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    examples = prepare_examples(pipe, settings, read_toml(settings["dataset"]))
    unet = pipe.unet.to(DEVICE)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=settings["rank"], lora_alpha=settings["rank"],
                               target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    parameters = [param for param in unet.parameters() if param.requires_grad]
    for param in parameters:
        param.data = param.data.float()
    unet.enable_gradient_checkpointing()
    unet.train()
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
    losses = []
    for step in range(settings["steps"]):
        latents, prompt, pooled, time_ids = (value.to(DEVICE) for value in example_rng.choice(examples))
        latents = latents[None]
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=DEVICE, generator=noise_generator)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=DEVICE,
                                 generator=noise_generator)
        noisy = scheduler.add_noise(latents, noise, timestep)
        predicted = unet(noisy, timestep, prompt,
                         added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
        target = scheduler.get_velocity(latents, noise, timestep) if scheduler.config.prediction_type == "v_prediction" else noise
        loss = torch.nn.functional.mse_loss(predicted.float(), target.float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if settings["steps"] <= 4 or (step + 1) % 10 == 0:
            print(f"step {step + 1:03d}  loss {losses[-1]:.6f}", flush=True)
    adapter = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(out, unet_lora_layers=adapter)
    report = {"base": settings["base_model"], "dataset": settings["dataset"],
              "gpu": torch.cuda.get_device_name(), "examples": len(examples), "steps": len(losses),
              "trainable_parameters": sum(param.numel() for param in parameters), "loss": losses,
              "seed": seed, "precision": settings["precision"]}
    (out / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate(settings):
    shared = [{k: v for k, v in item.items() if k not in ("dataset", "output")} for item in settings]
    if shared[0] != shared[1]:
        raise ValueError("Use the same base, seed and training/inference settings in both evaluation arms")
    out = ROOT / "work" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipe(settings[0])
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    rng = random.Random(settings[0]["seed"])
    schedule = [(group, prompt, seed) for prompt in PROMPTS for seed in SEEDS for group in range(2)]
    rng.shuffle(schedule)
    scorecard, key = [], []
    for group, item in enumerate(settings):
        pipe.load_lora_weights(ROOT / item["output"], adapter_name=f"group{group}")
    for i, (group, prompt, seed) in enumerate(schedule, 1):
        name = f"trial_{i:02d}.png"
        pipe.set_adapters([f"group{group}"])
        sample = pipe(prompt, width=settings[0]["resolution"], height=settings[0]["resolution"],
                      num_inference_steps=settings[0]["inference_steps"], guidance_scale=5.0,
                      generator=torch.Generator(device=DEVICE).manual_seed(seed)).images[0]
        sample.save(out / name)
        scorecard.append({"image": name, "prompt": prompt, "character_fidelity_0_to_2": None,
                          "style_contamination_0_to_2": None, "visible_artifact_0_to_2": None})
        key.append({"image": name, "dataset": settings[group]["dataset"], "seed": seed})
    for name, rows in (("scorecard", scorecard), ("key", key)):
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"{len(schedule)} generated trials in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="+", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Training and generation require a CUDA-enabled PyTorch environment.")
    settings = [read_toml(path) for path in args.config]
    if args.evaluate:
        if len(settings) != 2:
            parser.error("--evaluate requires the two group config files")
        evaluate(settings)
        return
    for item in settings:
        print(json.dumps(train(item), indent=2), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
