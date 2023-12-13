# Hillhouse

This repository contains code for Reimagining Hillhouse. This work was completed as part of CPSC 480: Introduction to Computer Vision.

## Abstract

> Lorem ipsum dolor sit amet. Et porro dolor a modi necessitatibus vel ipsa facere cum error quasi et rerum soluta et esse deserunt et suscipit odit. Aut voluptates fuga aut cupiditate suscipit eum tenetur architecto sed blanditiis commodi et velit magni eum culpa saepe. Sit doloribus sint et enim voluptatum vel consequuntur tempora sed fugit consequatur est sint iure. In assumenda beatae ut minima recusandae vel earum rerum est dolorum accusamus qui praesentium ratione qui excepturi temporibus sed dicta quod? Ex corrupti autem et quibusdam quod ex error odit est molestiae galisum. Et quia dolorum est eveniet voluptatem aut neque sapiente. Eos unde voluptates eos facere repudiandae et dicta explicabo est magnam voluptatem qui dolorum voluptatibus. Sed enim officia vel aperiam ratione hic aperiam aliquid id nobis ducimus.

## Quickstart

1. Clone this repository.

```
$ git clone https://github.com/nicoleytlam/hillhouse.git
```

2. Create a Python virtual environment and install package requirements.

```
$ cd hillhouse
$ python -m venv venv
$ pip install -U pip wheel # update pip
$ pip install -r requirements.txt
```

3. Run image-to-image stylization on the combined Hillhouse image.

```
CUDA_VISIBLE_DEVICES=0 python img2img.py
```

## Homography

TODO.

## Stylization

To run image stylization and generation, run [`img2img.py`](img2img.py). The full list of supported arguments are shown below.

```
$ python img2img.py -h
usage: img2img.py [-h] [--seed SEED] [--device DEVICE] [--model MODEL] [--dtype DTYPE]
                  [--image_path IMAGE_PATH] [--prompt PROMPT] [--strength STRENGTH] [--guidance GUIDANCE]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  --device DEVICE
  --model MODEL
  --dtype DTYPE
  --image_path IMAGE_PATH
  --prompt PROMPT
  --strength STRENGTH
  --guidance GUIDANCE
```

## License

Released under the [MIT License](LICENSE).
