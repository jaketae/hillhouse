import argparse
import os

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
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
    image_title = os.path.split(args.image_path)[-1]
    os.makedirs(os.path.join("assets", "output"), exist_ok=True)
    images[0].save(
        os.path.join(
            "assets",
            "output",
            f"{image_title}_{args.prompt}_{args.strength}_{args.guidance}.png",
        )
    )


if __name__ == "__main__":
    main(parse_args())
