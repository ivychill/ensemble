import face_model
import argparse
import cv2
import sys
import numpy as np
import os
from scipy import misc


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--data_dir', type=str, default='/data/dataset/MegaFace/raw/', help='data path')
    parser.add_argument('--output_dir', type=str, default='/data/dataset/MegaFace/MegaFace_160/', help='data path')
    parser.add_argument('--image_size', default='112,112', help='')
    parser.add_argument('--model', default='/data/fengchen/ensemble/model/model-r100-ii/model,0', help='path to load model.')
    parser.add_argument('--ga_model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='mtcnn opttion,ver dist threshold')
    return parser.parse_args(argv)

def get_output_path(output_dir, dir, file):
    prefix, suffix = os.path.splitext(file)
    target_dir = os.path.join(output_dir, dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_path = os.path.join(target_dir, prefix+".npy")
    return target_path

def main(args):
    # loading model
    model = face_model.FaceModel(args)
    # # get feature
    # feature_label_id = {"feature":[],"label":[],"id":[]}
    output_dir = args.output_dir
    if not  os.path.exists (output_dir):
        os.makedirs(output_dir)

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
                # img_rgb = cv2.imread(image_path)
                # img_input = model.get_input(img_rgb)
                img_rgb = misc.imread(image_path)
                img_input = np.transpose(img_rgb, (2, 0, 1))

                if img_input is None:
                    print("get feature fail ", image_path)
                    continue
                else:
                    emb = model.get_feature(img_input)
                    np.save(output_path, emb)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))