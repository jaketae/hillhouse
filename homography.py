import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.transform
from PIL import Image


def config_plot():
    plt.box(False)
    plt.axis("off")


def load_image(image_path: str):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.asarray(image)
    return image


def main():
    # load images
    image0 = load_image(os.path.join("assets", "input", "hillhouse.jpg"))
    image1 = load_image(os.path.join("assets", "input", "music_building.jpg"))

    # keypoints and matches
    keypoints0 = [
        [1022, 150],
        [1022, 166],
        [1022, 287],
        [1022, 438],
        [1022, 454],
        [974, 159],
        [974, 410],
    ]
    keypoints0 = [keypoint[::-1] for keypoint in keypoints0]
    keypoints0 = np.array(
        [[image0.shape[0] - keypoint[0], keypoint[1]] for keypoint in keypoints0]
    )
    keypoints1 = [[44, 46], [44, 64], [44, 180], [44, 329], [44, 347], [0, 49], [0, 293]]
    keypoints1 = np.array([keypoint[::-1] for keypoint in keypoints1])
    keypoints1 = np.array(
        [[image1.shape[0] - keypoint[0], keypoint[1]] for keypoint in keypoints1]
    )
    matches_images01 = np.array([[i, i] for i in range(len(keypoints1))])
    keypoints0_matched_indices = matches_images01[:, 0]
    keypoints1_matched_indices = matches_images01[:, 1]

    # extract matched keypoints from image 0 and 1
    keypoints0_matched = keypoints0[keypoints0_matched_indices, :]
    keypoints1_matched = keypoints1[keypoints1_matched_indices, :]
    keypoints0_matched = keypoints0_matched[:, ::-1]
    keypoints1_matched = keypoints1_matched[:, ::-1]

    # fit transform
    data = (keypoints0_matched, keypoints1_matched)
    homography1to0 = skimage.transform.ProjectiveTransform()
    homography1to0.estimate(*data)

    # prepare stitching
    n_height0, n_width0 = image0.shape[:2]
    _, n_width1 = image1.shape[:2]
    output_shape = (n_height0, n_width0 + n_width1)
    image1to0 = skimage.transform.warp(
        image1,
        homography1to0,
        output_shape=output_shape,
        order=1,
        cval=-1,
    )
    image1to0_valid_pixels = np.where(image1to0 != -1, 1, 0)
    image1to0 = image1to0 * image1to0_valid_pixels
    image0_extended = np.pad(
        image0.astype(float) / 255, ((0, 0), (0, n_width1), (0, 0)), constant_values=-1
    )
    image0_extended_valid_pixels = np.where(image0_extended != -1, 1, 0)
    image0_extended = image0_extended * image0_extended_valid_pixels

    # combine images as linear combination
    sum_valid_pixels = image1to0_valid_pixels + image0_extended_valid_pixels
    image1to0_weights = np.where(
        sum_valid_pixels > 0,
        image1to0_valid_pixels / sum_valid_pixels,
        0.0,
    )
    image0_extended_weights = np.where(
        sum_valid_pixels > 0,
        image0_extended_valid_pixels / sum_valid_pixels,
        0.0,
    )
    image_combined = image1to0_weights * image1to0 + image0_extended_weights * image0_extended

    # save result
    os.makedirs(os.path.join("assets", "output"), exist_ok=True)
    config_plot()
    plt.imshow(image_combined)
    plt.tight_layout()
    plt.savefig(os.path.join("assets", "output", "stitched.png"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
