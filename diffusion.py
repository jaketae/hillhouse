import argparse
import os
from typing import List, Optional, Union

import torch
from diffusers import DDPMPipeline, ImagePipelineOutput
from PIL import Image
from torchvision import transforms

from utils import set_seed


class ImageConditionedDiffusionPipeline(DDPMPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        image,
        num_forward_steps: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # forward process
        # image = image.unsqueeze(0)
        noise = torch.randn_like(image)
        image = self.scheduler.add_noise(image, noise, torch.tensor([num_forward_steps]))

        for t in self.progress_bar(self.scheduler.timesteps):
            if t > num_forward_steps:
                continue
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="google/ddpm-cifar10-32")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--image_path", type=str, default="assets/input/low_sketch.jpg")
    parser.add_argument("--forwards", type=int, default=100)
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    pipe = ImageConditionedDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
    )
    pipe = pipe.to(args.device)
    pipe.enable_attention_slicing()
    init_image = Image.open(args.image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize((128, 128)),
            transforms.ConvertImageDtype(getattr(torch, args.dtype)),
        ]
    )
    images = pipe(
        image=transform(init_image).to(args.device).unsqueeze(0), num_forward_steps=args.forwards
    ).images
    image_title = os.path.split(args.image_path)[-1]
    os.makedirs(os.path.join("assets", "output"), exist_ok=True)
    images[0].save(
        os.path.join(
            "assets",
            "output",
            f"{image_title}_{args.forwards}.png",
        )
    )


if __name__ == "__main__":
    main(parse_args())
