from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import sys
import numpy as np
import os
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from scipy import misc
import glob
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_preprocess

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    # _vec = args.image_size.split(',')
    # assert len(_vec)==2
    # image_size = (int(_vec[0]), int(_vec[1]))
    # self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = args.image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector

  def get_input(self, face_img,output_filename_n):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is not None:
        bbox, points = ret
        if bbox.shape[0] !=0:
            bbox = bbox[0,0:4]
            points = points[0,:].reshape((2,5)).T
            # print('bbox',bbox)
            # print('points',points)
            nimg = face_preprocess.preprocess(face_img, bbox, points, image_size=self.image_size)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            misc.imsave(output_filename_n, nimg)
            return 1
        else:
            return 0
    else:
        return 0
    # aligned = np.transpose(nimg, (2,0,1))
    # return aligned

def class_thread(inpath, maxconnections):
    list_class_per_thread = []
    all_class = os.listdir(inpath)
    all_class_num=len(all_class)
    class_per_thread = int(all_class_num/maxconnections)
    print(all_class_num, class_per_thread)   #(5762, 1152)
    for i in range(maxconnections-1):
        list_class_per_thread.append(all_class[(i*class_per_thread):((i+1)*class_per_thread)])
    list_class_per_thread.append(all_class[((maxconnections-1)*class_per_thread):])
    assert len(list_class_per_thread) == maxconnections
    print('split class  to thread ....')
    return list_class_per_thread

def multi_thread(args):
    # #loading model
    # model = FaceModel(args)
    maxconnections = args.thread
    class_thread_list = class_thread(args.data_dir, maxconnections)
    semlock = threading.BoundedSemaphore(maxconnections)
    for i in range(len(class_thread_list)):
        semlock.acquire()
        t=threading.Thread(target=align, args=(args,class_thread_list[i],i,))
        t.start()

def align(args,class_thread,thread_num):
    #loading model
    model = FaceModel(args)
    fail_align = 0
    output_dir=args.output_dir
    print("all class number ", len(class_thread))
    for i_id  in  range(len(class_thread)):
        lebal_path=os.path.join(args.data_dir,class_thread[i_id])
        out_class_path=os.path.join(output_dir,class_thread[i_id])
        if not  os.path.exists (out_class_path):
            os.makedirs(out_class_path)
        if os.path.isdir(lebal_path):
            print("thread_num{}\t procesing:{}/{}".format(thread_num,i_id ,len(class_thread)))
            for img in os.listdir(lebal_path):
                fname,fename=os.path.splitext(img)
                img_path=os.path.join(lebal_path,img)
                output_filename = os.path.join(out_class_path,fname+'.png')
                if os.path.exists(output_filename):
                    continue
                else:
                    try :
                        img_BGR = cv2.imread(img_path)
                        img_input = model.get_input(img_BGR,output_filename)
                        if img_input == 0:
                            print('thread_num',thread_num,"align fail 1",img_path)
                            fail_align = fail_align + 1
                    except BaseException:
                        print('thread_num',thread_num,"align fail 2",img_path)
                        fail_align = fail_align + 1
                        continue
                    
    print ('fail to align img num:',fail_align)
    semlock.release()

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--data_dir', type=str,default='/data/fengchen/ensemble/dataset/ytf/raw/', help='data path')
    parser.add_argument('--output_dir', type=str,default='/data/fengchen/ensemble/dataset/ytf_112', help='output path')
    parser.add_argument('--image_size', default='112,112', help='')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--thread', default=12, type=int, help='thread number')
    return parser.parse_args(argv)


if __name__ == '__main__':
    multi_thread(parse_arguments(sys.argv[1:]))
