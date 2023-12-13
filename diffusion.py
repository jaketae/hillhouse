import argparse
from typing import List, Optional, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput


class ImageConditionedDiffusionPipeline(DiffusionPipeline):
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
        image = image.unsqueeze(0)
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
    args = parser.parse_args("--model", type=str, default="")
    args = parser.parse_args("--forwards", type=int, default=100)
    return args


def main(args):
    pass


if __name__ == "__main__":
    main(parse_args())
