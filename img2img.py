import argparse
import os
import random

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def flatten_model_name(model_name):
    tokens = model_name.split("/")
    return "-".join(tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--image_path", type=str, default="assets/input/high_sketch2.jpg")
    parser.add_argument("--prompt", type=str, default="A photograph")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--guidance", type=float, default=3)
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
    )
    pipe = pipe.to(args.device)
    init_image = Image.open(args.image_path).convert("RGB")
    images = pipe(
        prompt=args.prompt,
        image=init_image,
        strength=args.strength,
        guidance_scale=args.guidance,
    ).images
    image_title = os.path.split(args.image_path)[-1].split(".")[0]
    model_name = flatten_model_name(args.model)
    os.makedirs(os.path.join("assets", "output", model_name, args.prompt), exist_ok=True)
    images[0].save(
        os.path.join(
            "assets",
            "output",
            model_name,
            args.prompt,
            f"{image_title}_{args.strength}_{args.guidance}.png",
        )
    )


if __name__ == "__main__":
    main(parse_args())
