import argparse
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--image_path", type=str, default="assets/hillhouse.jpg")
    parser.add_argument("--prompt", type=str, default="A photograph")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=3)
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
    )
    pipe = pipe.to(args.device)
    init_image = Image.open(args.image_path).convert("RGB")
    images = pipe(
        prompt=args.prompt,
        image=init_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
    ).images
    image_title = os.path.split(args.image_path)[-2]
    images[0].save(
        os.path.join(
            "assets",
            "outputs",
            f"{image_title}_{args.prompt}_{args.strength}_{args.guidance_scale}.png",
        )
    )


if __name__ == "__main__":
    main(parse_args())
