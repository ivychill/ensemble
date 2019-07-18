#!/bin/bash

python triplet_represent.py \
    --data_dir /data/fengchen/datasets/lfw/lfw_mtcnnpy_160 \
    --output_dir $(pwd)/emb/triplet \
    --trained_model_dir $(pwd)/model/20190220-111313/

python arc_represent.py \
    --data_dir /data/fengchen/datasets/lfw/lfw_mtcnnpy_112 \
    --output_dir $(pwd)/emb/triplet \
    --trained_model_dir $(pwd)/model/model-r100-ii/