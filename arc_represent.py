import face_model
import argparse
import cv2
import sys
import numpy as np
import os

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--data_path', type=str,default='/data/liukang/face_data/xiongma_size112/cam_ID/', help='data path')
    parser.add_argument('--output_dir', type=str,default='/data/liukang/face_data/20190708_test/xiongma/80w_size160/', help='data path')
    parser.add_argument('--image_size', default='112,112', help='')
    parser.add_argument('--model', default='/data/liukang/Project_facenet/insightface-master/recognition/models/80w_arcface_new/r100_arcface_emore,156', help='path to load model.')
    parser.add_argument('--ga_model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='mtcnn opttion,ver dist threshold')
    return parser.parse_args(argv)

def gen_feature(args):
    #loading model
    model = face_model.FaceModel(args)
    #get feature
    feature_lebal_id={"feature":[],"label":[],"id":[]}
    output_dir=args.output_dir
    if not  os.path.exists (output_dir):
        os.makedirs(output_dir)
    lebal_list=os.listdir(args.data_path)
    print("data_path",len(lebal_list))
    for i_id in range(len(lebal_list)):
        lebal_path=os.path.join(args.data_path,lebal_list[i_id])
        if os.path.isdir(lebal_path):
            print("procesing:{}/{}".format(i_id ,len(lebal_list)))
            for img in os.listdir(lebal_path):
                img_path=os.path.join(lebal_path,img)
                img_RGB = cv2.imread(img_path)
                img_input = model.get_input(img_RGB)
                if img_input is None:
                    print("get feature fail",img_path)
                    continue
                else:
                    feature= model.get_feature(img_input)
                    feature_lebal_id['feature'].append(feature)
                    feature_lebal_id['label'].append(lebal_list[i_id])
                    feature_lebal_id['id'].append(i_id)

    print(len(feature_lebal_id['feature']))
    # print(feature_lebal_id['id'])
    print(len(feature_lebal_id['label']))
    np.save(os.path.join(output_dir, "labels_name.npy"), feature_lebal_id['label'])
    np.save(os.path.join(output_dir, "gallery.npy"), feature_lebal_id['id'])
    np.save(os.path.join(output_dir, "signatures.npy"), feature_lebal_id['feature'])

if __name__ == '__main__':
    gen_feature(parse_arguments(sys.argv[1:]))
