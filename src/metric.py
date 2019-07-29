
import numpy as np
from log import logger


def get_distance(x1, x2):
    # print 'get distance',x1.shape,type(x1)
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        d = x1.astype(np.float32) - x2.astype(np.float32)
    else:
        x1 = np.array(x1, dtype=np.float32)
        x2 = np.array(x2, dtype=np.float32)
        d = x1 - x2
    return np.round(np.dot(d, d.T), 3)

def calculate_accuracy(threshold, distList, actual_issameList):
    predict_issame = np.less(distList, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issameList))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issameList)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issameList)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issameList))
    # logger.debug('th: %d, same: %d, tp: %d, fp: %d, tn: %d, fn: %d' % (len(distList), len(actual_issameList), tp, fp, tn, fn))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    recall = tpr
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(distList)
    precision =  0 if (tp + fp == 0) else float(tp) / float(tp + fp)

    tnr =  0 if (tn + fp == 0) else float(tn) / float(tn + fp)
    fnr =  0 if (tp + fn == 0) else float(fn) / float(tp + fn)
    return recall, precision, acc, tnr, fpr, fnr

# def evaluate(embeddings, actual_issame, nrof_folds=10):
#     # Calculate evaluation metrics
#     thresholds = np.arange(0, 4, 0.01)
#     embeddings1 = embeddings[0::2]
#     embeddings2 = embeddings[1::2]
#     tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
#         np.asarray(actual_issame), nrof_folds=nrof_folds)
#     thresholds = np.arange(0, 4, 0.001)
#     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
#         np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
#     return tpr, fpr, accuracy, val, val_std, far
#
# def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
#     assert (embeddings1.shape[0] == embeddings2.shape[0])
#     assert (embeddings1.shape[1] == embeddings2.shape[1])
#     nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
#     nrof_thresholds = len(thresholds)
#     k_fold = KFold(n_splits=nrof_folds, shuffle=False)
#
#     tprs = np.zeros((nrof_folds, nrof_thresholds))
#     fprs = np.zeros((nrof_folds, nrof_thresholds))
#     accuracy = np.zeros((nrof_folds))
#
#     diff = np.subtract(embeddings1, embeddings2)
#     dist = np.sum(np.square(diff), 1)
#     indices = np.arange(nrof_pairs)
#
#     for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
#
#         # Find the best threshold for the fold
#         acc_train = np.zeros((nrof_thresholds))
#         for threshold_idx, threshold in enumerate(thresholds):
#             _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
#         best_threshold_index = np.argmax(acc_train)
#         for threshold_idx, threshold in enumerate(thresholds):
#             tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
#                                                                                                  dist[test_set],
#                                                                                                  actual_issame[
#                                                                                                      test_set])
#         _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
#                                                       actual_issame[test_set])
#
#     tpr = np.mean(tprs, 0)
#     fpr = np.mean(fprs, 0)
#     return tpr, fpr, accuracy
#
#
# def calculate_accuracy(threshold, dist, actual_issame):
#     predict_issame = np.less(dist, threshold)
#     tp = np.sum(np.logical_and(predict_issame, actual_issame))
#     fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#     tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
#     fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
#
#     tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
#     fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
#     acc = float(tp + tn) / dist.size
#     return tpr, fpr, acc
#
#
# def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
#     assert (embeddings1.shape[0] == embeddings2.shape[0])
#     assert (embeddings1.shape[1] == embeddings2.shape[1])
#     nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
#     nrof_thresholds = len(thresholds)
#     k_fold = KFold(n_splits=nrof_folds, shuffle=False)
#
#     val = np.zeros(nrof_folds)
#     far = np.zeros(nrof_folds)
#
#     diff = np.subtract(embeddings1, embeddings2)
#     dist = np.sum(np.square(diff), 1)
#     indices = np.arange(nrof_pairs)
#
#     for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
#
#         # Find the threshold that gives FAR = far_target
#         far_train = np.zeros(nrof_thresholds)
#         for threshold_idx, threshold in enumerate(thresholds):
#             _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
#         if np.max(far_train) >= far_target:
#             f = interpolate.interp1d(far_train, thresholds, kind='slinear')
#             threshold = f(far_target)
#         else:
#             threshold = 0.0
#
#         val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
#
#     val_mean = np.mean(val)
#     far_mean = np.mean(far)
#     val_std = np.std(val)
#     return val_mean, val_std, far_mean
#
#
# def calculate_val_far(threshold, dist, actual_issame):
#     predict_issame = np.less(dist, threshold)
#     true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
#     false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#     n_same = np.sum(actual_issame)
#     n_diff = np.sum(np.logical_not(actual_issame))
#     val = float(true_accept) / float(n_same)
#     far = float(false_accept) / float(n_diff)
#     return val, far