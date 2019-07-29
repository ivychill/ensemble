#!/bin/bash

export PYTHONPATH=$(pwd)/src
#python triplet/represent.py \
#    --data_dir $(pwd)/dataset/helmet/helmet_160 \
#    --output_dir $(pwd)/emb/helmet/triplet \
#    --trained_model_dir $(pwd)/model/20190220-111313/
#    --gpu 3
#
#python arc/represent.py \
#    --data_dir $(pwd)/dataset/helmet/helmet_112 \
#    --output_dir $(pwd)/emb/helmet/arc \
#    --model /data/fengchen/ensemble/model/model-r100-ii/model,0 \
#    --gpu 3

python triplet/represent.py \
    --data_dir $(pwd)/dataset/camera/camera_160 \
    --output_dir $(pwd)/emb/camera/triplet \
    --trained_model_dir $(pwd)/model/20190220-111313/ \
    --gpu 3

python arc/represent.py \
    --data_dir $(pwd)/dataset/camera/camera_112 \
    --output_dir $(pwd)/emb/camera/arc \
    --model /data/fengchen/ensemble/model/model-r100-ii/model,0 \
    --gpu 3