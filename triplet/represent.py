#!/usr/bin/env python
# coding=utf-8

import os
import sys
import argparse
import time
from scipy import misc

sys.path.insert(1, "../src")
import facenet
import numpy as np
from sklearn.datasets import load_files
import tensorflow as tf
from six.moves import xrange

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_output_path(output_dir, dir, file):
    prefix, suffix = os.path.splitext(file)
    target_dir = os.path.join(output_dir, dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_path = os.path.join(target_dir, prefix+".npy")
    return target_path

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # create output directory if it doesn't exist
            output_dir = os.path.expanduser(args.output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            # load the model
            print("Loading trained model...\n")
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.trained_model_dir))
            facenet.load_model(args.trained_model_dir)

            # grab all image paths and labels
            print("Finding image paths and targets...\n")
            data = load_files(args.data_dir, load_content=False, shuffle=False)
            # labels_array = data['target']
            # paths = data['filenames']
            # print(data)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input_ID:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings_ID:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Generating embeddings from images...\n')
            # emb = np.zeros(embedding_size)
            dirs = os.listdir(args.data_dir)
            dirs.sort()
            for dir in dirs:
                path = os.path.join(args.data_dir, dir)
                if os.path.isdir(path):
                    print("path: ", path)
                    files = os.listdir(path)
                    files.sort()
                    for file in files:
                        output_path = get_output_path(output_dir, dir, file)
                        image_path = os.path.join(path, file)
                        images = facenet.load_image(image_path, do_random_crop=False, do_random_flip=False,
                                                   image_size=image_size, do_prewhiten=True)
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        np.save(output_path, emb[0])

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Batch-represent face embeddings from a given data directory")
    parser.add_argument('-d', '--data_dir', type=str,
                        help='directory of images with structure as seen at the top of this file.')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='directory containing aligned face patches with file structure as seen at the top of this file.')
    parser.add_argument('--trained_model_dir', type=str,
                        help='Load a trained model before training starts.')
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=50)
    parser.add_argument('--gpu', default=0, type=str, help='gpu id')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))