# coding=utf-8
import numpy as np
import os
# import pandas as pd
import warnings
from itertools import combinations
# from getRep import getReptf
from functools import partial
from datetime import datetime
# import matplotlib.pyplot as plt
from sys import argv
# import face
import xlwt
import sys
import argparse

warnings.filterwarnings('ignore')
cwd = os.getcwd()


def getDistance(x1, x2):
    # print 'get distance',x1.shape,type(x1)
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        d = x1.astype(np.float32)-x2.astype(np.float32)
    else:
        x1 = np.array(x1,dtype=np.float32)
        x2 = np.array(x2,dtype=np.float32)
        d = x1-x2
    return np.round(np.dot(d,d.T),3)


def baseImageRep_tf(feature_path):
    dataSet=  np.load(os.path.join(feature_path, "signatures.npy"))
    labels = np.load(os.path.join(feature_path, "labels_name.npy"))
    assert len(dataSet) == len(labels)  # 样本数和标签数要相等
    dataSet = np.array(dataSet)
    labels = np.array(labels).reshape(-1,1)
    baseData = np.hstack((labels,dataSet))
    return baseData


def combinations_self(indata_ID,indata_camera):   # 输入numpy数组，两两组合各种可能,返回一个包含各种可能组合之间的欧式距离的list
    pairs_list_all=[]
    for ID in indata_ID:
        for camera in indata_camera:
            pairs_list_all.append((ID,camera))
    return  pairs_list_all


def combineCompare(indata):   # 输入numpy数组，两两组合各种可能,返回一个包含各种可能组合之间的欧式距离的list
    result = []
    # print indata.shape
    if indata.shape[0] >1:
        res = combinations(indata,2)
        for pair in res:
            x1, x2 = pair
            distance = getDistance(x1,x2)
            print ('distance:',distance)
            result.append(distance)
    return result


def zipCompare(data1,data2):
    result = []
    pairlist = []
    if data1.shape>0 and data2.shape>0:
        for index in data1:
            pairlist.extend(zip([index]*data2.shape[0],data2))
        for pair in pairlist:
            x1,x2 = pair
            dist = getDistance(x1, x2)
            result.append(dist)
            print ('dist:',dist)
    return result


def calculate_accuracy(threshold, distList, actual_issameList):
    predict_issame = np.less(distList, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issameList))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issameList)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issameList)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issameList))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    recall = tpr
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(distList)
    precision =  0 if (tp + fp == 0) else float(tp) / float(tp + fp)

    tnr =  0 if (tn + fp == 0) else float(tn) / float(tn + fp)
    fnr =  0 if (tp + fn == 0) else float(fn) / float(tp + fn)
    return recall, precision, acc, tnr, fpr,fnr

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--feature_path', type=str,default=sys.path.append(os.path.join(os.path.dirname(__file__),'..' 'Saveout')), help='feature path')
    parser.add_argument('--output_dir', type=str,default=os.path.join(os.path.dirname(__file__), '..','Saveout'), help='data path')
    parser.add_argument('--P_valid_force', type=int,default=100000)
    parser.add_argument('--N_valid_force', type=int,default=1000000)
    return parser.parse_args(argv)


def main(argv):
    now = datetime.strftime(datetime.now(), '%Y-%m-%d-%H_%M_%S')
    thresholds = np.arange(0.01, 3.2, 0.01)
    P_valid_force = argv.P_valid_force
    N_valid_force = argv.N_valid_force
    best_precison = 0
    best_accuracy = 0
    best_threshold_precison=None
    best_threshold_acc=None
    myRound = partial(round,ndigits=4)     #偏函数,2018.08.06，by xjxf
    feature_path =argv.feature_path

    savedir = os.path.join(argv.output_dir,now)
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))
    # 创建excel文件指针,2018.08.07,by xjxf  _start
    path_excel=os.path.join(savedir,'oush_distance%s_data.xls'%now)
    xls_3=xlwt.Workbook()
    sheet_1=xls_3.add_sheet('sheet_1',cell_overwrite_ok=True)
    sheel_title=['threshold','tnr', 'fpr','fnr','recall', 'precision', 'acc']
    for i_sheet in range(len(sheel_title)):
        sheet_1.write(0,i_sheet,sheel_title[i_sheet])
    row_count=1
    # 创建excel文件指针,2018.08.07,by xjxf  _end
    print ('-----测试日期:{},  测试数据:{},距离指标:欧式距离------\n'.format(now,feature_path))

    baseData = baseImageRep_tf(feature_path)  # 提取基础集的特征点和标签
    pairs_list_all = list(combinations(baseData, 2))

    np.random.shuffle(pairs_list_all)
    P_num = 0
    N_num = 0
    pair_list =[]
    for pair in pairs_list_all:
        if pair[0][0]==pair[1][0] and P_num<P_valid_force:
            pair_list.append(pair)
            P_num+=1
        elif pair[0][0]!=pair[1][0] and N_num<N_valid_force:
            pair_list.append(pair)
            N_num+=1
        elif P_num==P_valid_force and N_num== N_valid_force:
            break
    s_ = ('**正测试单元数量:{},  负测试单元数量:{}**\n'.format(P_num,N_num))
    print (s_)

    distList = []
    actual_issameList = []
    for pair in pair_list:
        actual_issame = True
        if pair[0][0] != pair[1][0]:
            actual_issame = False
        dist = getDistance(pair[0][1:], pair[1][1:])
        distList.append(dist)
        actual_issameList.append(actual_issame)

    for threshold in thresholds:
        print ('---------threshold:',round(threshold,2),'---------------\n')

        recall, precision, acc, tnr, fpr,fnr = map(myRound,calculate_accuracy(threshold, distList, actual_issameList))

        if precision>best_precison:
            best_precison = precision
            best_threshold_precison = threshold

        if acc>best_accuracy:
            best_accuracy=acc
            best_threshold_acc = threshold

        ss = 'recall召回率:'+str(recall)+'\t'+'precison精准率:'+str(precision)+'\t'+' accuracy正确率:'+str(acc)+'\n'
        sss = 'TNR真负率:'+str(tnr)+'\t'+'FPR假正率:'+str(fpr)+'\t'+'FNR假负率:'+str(fnr)+'\t\n\n'
        print (ss,sss)
        #写excel文件,2018.08.07, by xjxf  _start
        list_data=[threshold,tnr, fpr,fnr,recall, precision, acc]
        for i_1 in range(len(list_data)):
            sheet_1.write(row_count,i_1,list_data[i_1])
        row_count=row_count+1
    xls_3.save(path_excel)

        # 写excel文件,2018.08.07, by xjxf  _end

    foo =  '\n\n*********conclusion********\n'+'best_precison:{}   best_threshold_precision:{}\nbest_accuracy:{}   best_threshold_acc:{} '.format(best_precison, best_threshold_precison,best_accuracy,best_threshold_acc)
    print ( foo)
    print("data path:",feature_path)
    print("excel path:",path_excel)



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))