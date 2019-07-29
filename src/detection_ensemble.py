#!/usr/bin/env python

import os, sys, argparse
import numpy as np
import _pickle as cPickle
#import cPickle
import GPy
import GPyOpt
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

def parse_rec(filename):
    '''
    parse the xml file of the annotation files
    :param filename: the filename of the xml file
    :return:
    '''
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        obj_struct['bbox'] =[int(xmin),
                             int(ymin),
                             int(xmax),
                             int(ymax)]
        objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    '''
    calculate the ap
    :param rec: recall
    :param prec: precision
    :param use_07_metric: the format of the calculated way
    :return:
    '''
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_voc_results_file_template(image_set):
    '''
    get the results name of the detected result file
    :param image_set:
    :return:
    '''
    filename = 'comp4_det_' + image_set + '_{:s}.txt'
    #print("the filename is: ",filename)
    path = os.path.join(filename)
    return path

def get_annoinfo(annopath,imagesetfile,classname):
    '''
    get the annotations of the testdata
    :param annopath: the path of the annotation of the testdata
    :param imagesetfile: the txt path included the path of the testdata
    :param classname: the class name of the object
    :return: class_recs
              the format is
              dict:{"car":{"imgName":{'bbox': bbox, 'difficult': difficult,'det': det}},"person":{"imgName":{'bbox': bbox, 'difficult': difficult,'det': det}}}
    '''
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()   # the path of the test file  including the image name
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        imagename1 = imagename.split("/")[-1].split(".")[0]
        recs[imagename1] = parse_rec(annopath.format(imagename1))
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        imagename2 = imagename.split("/")[-1].split(".")[0]
        R = [obj for obj in recs[imagename2] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename2] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    return class_recs,npos

def get_detinfo(detpath,classname):
    '''
    get the detect result of the detect
    :param detpath: the detect result path
    :param classname: the class name of the object
    :return: image_ids,confidence,BB format is [],[],[]
    '''
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    return image_ids,confidence,BB

def get_testsetinfo(devkit_path,voc_dir_test,year, image_set, classes, output_dir_list = ['results1','results2']):
    '''
    get the testsetinfo,including class_res,which is the annotations,det_res,which is the detect results
    :param devkit_path: the test data path
    :param voc_dir_test: the test data path
    :param year: decide the way to get the ap
    :param image_set: the kind of the data set
    :param classes: the category of the object
    :param output_dir_list: the list of the txt of the result
    :return: class_res,which is the annotations;det_res,which is the detect results
    '''
    annopath = os.path.join(
        devkit_path,
        year,
        'Annotations',
        '{}.xml')    # get the testset annotation
    imagesetfile = os.path.join(
        voc_dir_test,
        image_set + '.txt') # get the testset file list
    # dict:{"car":{"imgName":{'bbox': bbox, 'difficult': difficult,'det': det}},"person":{"imgName":{'bbox': bbox, 'difficult': difficult,'det': det}}}
    class_res = {}
    # dict:{"yolo":{"car":[image_ids, confidence, BB],"person":[image_ids, confidence, BB]},"yolt":{"car":[image_ids, confidence, BB],"person":[image_ids, confidence, BB]}}
    det_res = {"yolo":{},"yolt":{}}
    # get the annoInfo
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        class_subres,npos = get_annoinfo(annopath, imagesetfile, cls)
        class_res[cls] = {"anno":class_subres,"npos":npos}
    # get the det result info
    for i_path in range(len(output_dir_list)):
        output_dir_cur = output_dir_list[i_path]
        #print("the output_dir_cur is:",output_dir_cur)
        for j,category in enumerate(classes):
            if category == '__background__':
                continue
            filename = output_dir_cur + '/' + get_voc_results_file_template(image_set).format(category)
            image_ids, confidence, BB = get_detinfo(filename,category)
            if output_dir_cur.split("/")[-1] == "results_yolo":
                det_res["yolo"][category] = [image_ids, confidence, BB]
            else:
                det_res["yolt"][category] = [image_ids, confidence, BB]
    return class_res,det_res

def get_bounding_box_oneimage(det_res_yolo,det_res_yolt,classes):
    '''
     get the bounding box of the two models detect and put in one dict,which key is image_id
    :param det_res_yolo: the detect results of the yolo model
    :param det_res_yolt: the detect results of the yolt model
    :param classes: the list of the classes name
    :return:
    '''

    det_res_imageid_allcls = {}
    for i, cls in enumerate(classes):
        det_res_imageid = {"yolo": {}, "yolt": {}}
        if cls == '__background__':
            continue
        det_res_yolo_list = det_res_yolo[cls]  # [image_ids, confidence, BB]
        det_res_yolt_list = det_res_yolt[cls]  # [image_ids, confidence, BB]
        cur_image_id_yolo = " "
        cur_image_id_yolt = " "
        for j1 ,image_id_yolo in enumerate(det_res_yolo_list[0]):
            if image_id_yolo != cur_image_id_yolo:
                cur_image_id_yolo = image_id_yolo
                cur_confidence_yolo = det_res_yolo_list[1][j1]
                cur_BB_yolo = det_res_yolo_list[2][j1]
                if image_id_yolo in det_res_imageid["yolo"].keys():
                    det_res_imageid["yolo"][image_id_yolo].append([cur_confidence_yolo,cur_BB_yolo])
                else:
                    det_res_imageid["yolo"][image_id_yolo] = [[cur_confidence_yolo, cur_BB_yolo]]
            else:
                det_res_imageid["yolo"][image_id_yolo].append([det_res_yolo_list[1][j1], det_res_yolo_list[2][j1]])
        for j2 ,image_id_yolt in enumerate(det_res_yolt_list[0]):
            if image_id_yolt != cur_image_id_yolt:
                cur_image_id_yolt = image_id_yolt
                cur_confidence_yolt = det_res_yolt_list[1][j2]
                cur_BB_yolt = det_res_yolt_list[2][j2]
                if image_id_yolt in det_res_imageid["yolt"].keys():
                    det_res_imageid["yolt"][image_id_yolt].append([cur_confidence_yolt,cur_BB_yolt])
                else:
                    det_res_imageid["yolt"][image_id_yolt] = [[cur_confidence_yolt, cur_BB_yolt]]
            else:
                det_res_imageid["yolt"][image_id_yolt].append([det_res_yolt_list[1][j2], det_res_yolt_list[2][j2]])
        det_res_imageid_allcls[cls] = det_res_imageid
    return det_res_imageid_allcls

def get_final_boxes(det_res_imageid_allcls,classes,x):
    '''
    get the final boxes info
    :param det_res_imageid_allcls: the dict of the detresults
    :param classes: the list of the classname of the object
    :param r: the weight
    :return:det_final format{"cls":[imgid,bb,conf]}
    '''

    det_final = {}
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        det_res_imageid_yolo = det_res_imageid_allcls[cls]["yolo"]
        det_res_imageid_yolt = det_res_imageid_allcls[cls]["yolt"]
        bb_final = np.empty([0, 4])
        confidence_final = np.empty([0])
        imgids_final = []
        for key in (det_res_imageid_yolo.keys() & det_res_imageid_yolt.keys()):
            #imgids_samekey = []
            bb_samekey = np.empty([0, 4])
            bb_yolo_new = np.empty([0,4])
            confidence_samekey = np.empty([0])
            confidence_yolo_new = np.empty([0])
            res_imageid_yolo = det_res_imageid_yolo[key]  # format:[[confidence,bb],[confidence,bb],...]
            res_imageid_yolt = det_res_imageid_yolt[key]
            n_box_yolo = len(res_imageid_yolo)
            n_box_yolt = len(res_imageid_yolt)
            n_boxes = n_box_yolo + n_box_yolt
            confidence_yolo = np.array([float(x[0]) for x in res_imageid_yolo])
            bb_yolo = np.array([x[1] for x in res_imageid_yolo])
            #bb_yolo_new = np.append(bb_yolo_new, bb_yolo, axis=0)
            for n2, box2 in enumerate(res_imageid_yolt):
                bb = box2[1]
                confidence_tem = np.array([box2[0]])
                ixmin = np.maximum(bb_yolo[:, 0], bb[0])
                iymin = np.maximum(bb_yolo[:, 1], bb[1])
                ixmax = np.minimum(bb_yolo[:, 2], bb[2])
                iymax = np.minimum(bb_yolo[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bb_yolo[:, 2] - bb_yolo[:, 0] + 1.) *
                       (bb_yolo[:, 3] - bb_yolo[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                if ovmax > 0.5:
                    #bb_yolo[jmax, :] = x[0][0] * bb_yolo[jmax, :] + x[0][1] * bb
                    bb_yolo[jmax, :] = x[0][0] * bb_yolo[jmax, :] + (1- x[0][0]) * bb
                    if confidence_yolo[jmax] > confidence_tem:
                        confidence_yolo[jmax] = confidence_yolo[jmax]
                    else:
                        confidence_yolo[jmax] = confidence_tem
                    n_boxes = n_boxes - 1
                    #imgids_final.append(key)
                else:
                    bb = np.reshape(bb, (1, 4))
                    bb_yolo_new = np.append(bb_yolo_new, bb, axis=0)
                    confidence_yolo_new = np.append(confidence_yolo_new, confidence_tem, axis=0)
                    #imgids_final.append(key)
            bb_samekey = np.append(bb_samekey, bb_yolo, axis=0)
            bb_samekey = np.append(bb_samekey, bb_yolo_new, axis=0)
            confidence_samekey = np.append(confidence_samekey, confidence_yolo, axis=0)
            confidence_samekey = np.append(confidence_samekey, confidence_yolo_new, axis=0)
            n_boxes_samekey = bb_samekey.shape[0]
            #print("the number of the boxes of the %s id %d" % (key,n_boxes))
            for n_box in range(n_boxes_samekey):
                imgids_final.append(key)
            bb_final = np.append(bb_final,bb_samekey,axis=0)
            confidence_final = np.append(confidence_final,confidence_samekey,axis=0)
        #print("the same key is: ", det_res_imageid_yolo.keys() & det_res_imageid_yolt.keys())
        #print("the different key is: ", det_res_imageid_yolo.keys() ^ det_res_imageid_yolt.keys())
        for key2 in (det_res_imageid_yolo.keys() ^ det_res_imageid_yolt.keys()):
            bb_diffkey = np.empty([0, 4])
            confidence_diffkey = np.empty([0])
            if key2 in det_res_imageid_yolo.keys():
                res_imageid_yolo_diffkey = det_res_imageid_yolo[key2]  # format:[[confidence,bb],[confidence,bb],...]
                confidence_yolo_diffkey = np.array([float(x[0]) for x in res_imageid_yolo_diffkey])
                bb_yolo_diffkey = np.array([x[1] for x in res_imageid_yolo_diffkey])
                n_boxes_diffkey = bb_yolo_diffkey.shape[0]
                for n_box1 in range(n_boxes_diffkey):
                    imgids_final.append(key2)
                bb_final = np.append(bb_final, bb_yolo_diffkey, axis=0)
                confidence_final = np.append(confidence_final, confidence_yolo_diffkey, axis=0)
            else:
                res_imageid_yolt_diffkey = det_res_imageid_yolt[key2]  # format:[[confidence,bb],[confidence,bb],...]
                confidence_yolt_diffkey = np.array([float(x[0]) for x in res_imageid_yolt_diffkey])
                bb_yolt_diffkey = np.array([x[1] for x in res_imageid_yolt_diffkey])
                n_boxes_diffkey = bb_yolt_diffkey.shape[0]
                for n_box2 in range(n_boxes_diffkey):
                    imgids_final.append(key2)
                bb_final = np.append(bb_final, bb_yolt_diffkey, axis=0)
                confidence_final = np.append(confidence_final, confidence_yolt_diffkey, axis=0)
        det_final[cls] = [imgids_final,confidence_final,bb_final]
    return det_final

def train_gpyopt(x):
    # first: get the detect results and the annotations (true label) of the testset
    print("the x is: ",x)
    #x = np.atleast_2d(np.exp(x))
    #print("the x is: ", x)
    #print("the shape of r.shape:",r.shape)
    #print("the shape of r.shape[0]:", r.shape[0])
    det_res_yolo = {}
    det_res_yolt = {}
    class_res_anno,det_res = get_testsetinfo(voc_dir,voc_dir_test,year, image_set, classes, output_dir_list)
    # second: process the data
    for key1 in det_res.keys():
        if key1 == "yolo":
           det_res_yolo = det_res[key1] # format {"car":[image_ids, confidence, BB],"person":[image_ids, confidence, BB]}
        elif key1 == "yolt":
          det_res_yolt = det_res[key1] #{"car":[image_ids, confidence, BB],"person":[image_ids, confidence, BB]}
        else:
          continue
    # format:{"car":{"yolo":{"imageid":[[confidence,bb],...]},"yolt":{"imgid":[[],...]}},"perosn":}
    det_res_imageid_allcls = get_bounding_box_oneimage(det_res_yolo,det_res_yolt,classes)
    det_final = get_final_boxes(det_res_imageid_allcls,classes,x)
    aps = []
    recs=[]
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        det_info = det_final[cls]  # the detected results of the two models
        class_recs = class_res_anno[cls]["anno"] # the annotation of the test data
        npos = class_res_anno[cls]["npos"]
        image_ids = det_info[0]
        confidence = det_info[1]
        BB = det_info[2]
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        confidence_sorted = [confidence[x] for x in sorted_ind]
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > 0.8:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print("the recall of %s is " % cls)
        print('Mean Recall={:.3f}'.format(np.mean(rec)))
        aps += [ap]
        recs += [rec]
        print('AP for {} = {:.4f}'.format(cls, ap))
    map = np.array([[np.mean(aps)]])
    map_min = -map
   # map = np.mean(aps)
    #map_min= np.empty([0,1])
    #map_min = np.append(map_min,(-map))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    return map_min

if __name__ == '__main__':
    #args = parse_args()
    output_dir_list = ["/home/asus/mixGroup/lkp/yolo/GPyOptTest/results_yolo",
                       "/home/asus/mixGroup/lkp/yolo/GPyOptTest/results_yolt"]  # the detected reults txt of two models
    class_file = '/home/asus/mixGroup/lkp/yolo/darknet/data/voc_Chengdu_car_person_cutmodel.names'
    voc_dir = '/home/asus/mixGroup/DataSet/Chengdu/VOC_data_only_car_person/VOC_data_test/VOCdevkit/'
    voc_dir_test = '/home/asus/mixGroup/DataSet/Chengdu/VOC_data_only_car_person/VOC_data_test'
    year = 'VOC2007'
    year1 = '2017'
    image_set = 'test'

    with open(class_file, 'r') as f:
        lines = f.readlines()

    classes = [t.strip('\n') for t in lines]

    use_07_metric = True if int(year1) < 2010 else False
    print('Evaluating detections')
   # det_res_yolo = {}
   # det_res_yolt = {}
   # class_res_anno, det_res = get_testsetinfo(voc_dir, voc_dir_test, year, image_set, classes, output_dir_list)
   # for key1 in det_res.keys():
       # if key1 == "yolo":
        #    det_res_yolo = det_res[key1] # format {"car":[image_ids, confidence, BB],"person":[image_ids, confidence, BB]}
        #elif key1 == "yolt":
         #   det_res_yolt = det_res[key1] #{"car":[image_ids, confidence, BB],"person":[image_ids, confidence, BB]}
       # else:
          #  continue
    #det_res_imageid_allcls = get_bounding_box_oneimage(det_res_yolo, det_res_yolt, classes)
    domain = [{ 'name':'x1','type':'continuous','domain':(0,1)}]
    #{'name': 'x2', 'type': 'continuous', 'domain': (0, 1)}

    X_init = np.array([[0.6]])
    #print("the X_init is:",X_init.shape)
    Y_init = train_gpyopt(X_init)
    #print("the Y_init is:", Y_init.shape)

    y_min = 0
    iter = 0
    iter_count = 10
    current_iter = 0
    X_step = X_init
    Y_step = Y_init

    while current_iter < iter_count:
        print("this is the %d-th iter" % current_iter)
        bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=X_step, Y=Y_step)
        x_next = bo_step.suggest_next_locations()
        print("the x_next is:",x_next)
        y_next = train_gpyopt(x_next)
        print("the y_next is:",y_next)
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_next))
        current_iter += 1
        if y_next[0][0] < y_min:
            y_min = y_next[0][0]
            iter = current_iter-1
        else:
            continue

    print("the min y is: ",y_min)
    print("the iter is: ",iter)
    print("the X_step is:",X_step)
    print(np.where(X_step == np.min(X_step)))
    print("the Y_step is:", Y_step)
    print(np.where(Y_step == np.min(Y_step)))

    x = np.arange(0.0, 1.0, 0.01)
    y = np.empty([0,1])
    for i in range(x.shape[0]):
        y1 = train_gpyopt(np.array([[x[i]]]))
        y = np.append(y,y1)

    plt.figure()
    plt.plot(x, y)
    for i, (xs, ys) in enumerate(zip(X_step, Y_step)):
        #plt.plot(xs, ys, 'rD', markersize=10 + 20 * (i + 1) / len(X_step))
        plt.plot(xs, ys, 'rD')
    plt.savefig('./test.jpg')
    plt.show()


    #constraints = [{'name': 'constr_1','constraint': '(x[:,0] + x[:,1]) - 1 - 0.01'},
                 #{'name': 'constr_2','constraint': '1 - (x[:,0] + x[:,1]) - 0.01'}
                   # ]

    #opt = GPyOpt.methods.BayesianOptimization(f = train_gpyopt,
                                            #  domain=domain,
                                              #constraints=constraints,
                                             # acquisition_type='LCB',
                                             # acquisition_weight=0.1)
    #opt.run_optimization(max_iter=10)
    #print(opt.x_opt)
    #print(opt.fx_opt)
    #opt.plot_convergence()
