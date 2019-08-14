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

#python triplet/represent.py \
#    --data_dir $(pwd)/dataset/MegaFace/MegaFace_160 \
#    --output_dir $(pwd)/emb/MegaFace/triplet \
#    --trained_model_dir $(pwd)/model/20190220-111313/ \
#    --gpu 1

#python arc/represent.py \
#    --data_dir $(pwd)/dataset/MegaFace/MegaFace_112 \
#    --output_dir $(pwd)/emb/MegaFace/arc \
#    --model /data/fengchen/ensemble/model/model-r100-ii/model,0 \
#    --gpu 3

#python arc/represent.py \
#    --data_dir $(pwd)/dataset/MegaFace/MegaFace_112 \
#    --output_dir $(pwd)/emb/MegaFace/combined \
#    --model /data/fengchen/ensemble/model/combined/r100_combined_emore,100 \
#    --gpu 3
#
#python arc/represent.py \
#    --data_dir $(pwd)/dataset/MegaFace/MegaFace_112 \
#    --output_dir $(pwd)/emb/MegaFace/cos \
#    --model /data/fengchen/ensemble/model/cosface/r100_cosface_emore,111 \
#    --gpu 3

#python arc/represent.py \
#    --data_dir $(pwd)/dataset/MegaFace/MegaFace_112 \
#    --output_dir $(pwd)/emb/MegaFace/nsoftmax \
#    --model /data/fengchen/ensemble/model/nsoftmax/r100_nsoftmax_emore,99 \
#    --gpu 3

python arc/represent.py \
    --data_dir $(pwd)/dataset/ytf/ytf_112 \
    --output_dir $(pwd)/emb/ytf/arc \
    --model /data/fengchen/ensemble/model/model-r100-ii/model,0 \
    --gpu 3

python arc/represent.py \
    --data_dir $(pwd)/dataset/ytf/ytf_112 \
    --output_dir $(pwd)/emb/ytf/combined \
    --model /data/fengchen/ensemble/model/combined/r100_combined_emore,100 \
    --gpu 3

python arc/represent.py \
    --data_dir $(pwd)/dataset/ytf/ytf_112 \
    --output_dir $(pwd)/emb/ytf/cos \
    --model /data/fengchen/ensemble/model/cosface/r100_cosface_emore,111 \
    --gpu 3

python arc/represent.py \
    --data_dir $(pwd)/dataset/ytf/ytf_112 \
    --output_dir $(pwd)/emb/ytf/nsoftmax \
    --model /data/fengchen/ensemble/model/nsoftmax/r100_nsoftmax_emore,99 \
    --gpu 3