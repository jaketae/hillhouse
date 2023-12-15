#!/bin/sh
for item in 'low_sketch' 'high_sketch1' 'high_sketch2'; do
   for strength in 0.4 0.8; do
       for guidance in 2 4; do
           python img2img.py --image_path "assets/input/$item.jpg" --strength $strength --guidance $guidance --prompt "a newly built college building next to a brick library"
        done
    done
done
