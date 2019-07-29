#!/usr/bin/env bash

#python src/align/align_dataset_mtcnn.py \
#    /nfs/kc/fengchen/ensemble/dataset/xiongma_160/ \
#    /nfs/kc/fengchen/ensemble/dataset/xiongma_112/ \
#    --image_size 112 \
#    --margin 20

python align/align_dataset_mtcnn.py \
    $(pwd)/dataset/camera/camera_160/ \
    $(pwd)/dataset/camera/camera_112/ \
    --image_size 112 \
    --margin 20