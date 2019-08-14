#!/usr/bin/env bash

#export PYTHONPATH=$(pwd)/src
#python align/align_dataset_mtcnn.py \
#    $(pwd)/dataset/MegaFace/raw/ \
#    $(pwd)/dataset/MegaFace/MegaFace_160/ \
#    --image_size 160 \
#    --margin 32
#python align/align_dataset_mtcnn.py \
#    $(pwd)/dataset/MegaFace/raw/ \
#    $(pwd)/dataset/MegaFace/MegaFace_112/ \
#    --image_size 112 \
#    --margin 20
#python align/align_dataset_mtcnn.py \
#    $(pwd)/dataset/ytf/raw/ \
#    $(pwd)/dataset/ytf/ytf_160/ \
#    --image_size 160 \
#    --margin 32
#python align/align_dataset_mtcnn.py \
#    $(pwd)/dataset/ytf/raw/ \
#    $(pwd)/dataset/ytf/ytf_112/ \
#    --image_size 112 \
#    --margin 20
#python arc/align.py \
#    --data_dir $(pwd)/dataset/MegaFace/raw/ \
#    --output_dir $(pwd)/dataset/MegaFace/MegaFace_112/ \
#    --gpu 1
python arc/align_mt.py \
    --data_dir $(pwd)/dataset/ytf/raw/ \
    --output_dir $(pwd)/dataset/ytf/ytf_112/ \
    --gpu 3