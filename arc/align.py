import mxnet as mx
import argparse
import cv2
import sys
import numpy as np
import os
from scipy import misc
from mtcnn_detector import MtcnnDetector
import face_image
import face_preprocess


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--data_dir', type=str,default='/data/fengchen/ensemble/dataset/ytf/raw/', help='data path')
    parser.add_argument('--output_dir', type=str,default='/data/fengchen/ensemble/dataset/ytf_112', help='output path')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    return parser.parse_args(argv)

def get_output_path(output_dir, dir, file):
    target_dir = os.path.join(output_dir, dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_path = os.path.join(target_dir, file)
    return target_path

def get_input(face_img, args):
    ctx = mx.gpu(args.gpu)
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.6,0.7,0.8])
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    ret = detector.detect_face(face_img, det_type = args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    return nimg
    # aligned = np.transpose(nimg, (2,0,1))
    # return aligned

def main(args):
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
                img_rgb = cv2.imread(image_path)
                img_input = get_input(img_rgb, args)
                if img_input is None:
                    print("align fail: ", image_path)
                    continue
                else:
                    misc.imsave(output_path, img_input)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))