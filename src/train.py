
import argparse
import os.path
import sys
from datetime import datetime
from numpy.random import seed
import GPy
import GPyOpt
import xlwt
from dataset import Dataset
from metric import *
from log import *


# objective function
def accuracy(parameters):
    global row
    weight = parameters[0][0]
    threshold = parameters[0][1]
    logger.debug("parameters: %s, type: %s" % (parameters, type(parameters)))
    logger.debug("weight: %f, threshold: %f" % (weight, threshold))

    distance_list = []
    issame_list = []
    for distances_and_issame in dataset.distances_and_issame_list:
        distance = distances_and_issame[0] * weight + distances_and_issame[1] * (1 - weight)
        issame = distances_and_issame[2]
        distance_list.append(distance)
        issame_list.append(issame)

    # logger.debug("construction of distances and issame finish...")
    recall, precision, acc, tnr, fpr, fnr = calculate_accuracy(threshold, np.asarray(distance_list), np.asarray(issame_list))
    logger.debug('accuracy: %f' % (acc))
    with open(os.path.join(log_dir, 'train_result.txt'),'at') as f:
        f.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n' % (weight, threshold, acc, precision, fpr, precision, tnr, fnr))

    list_data = [weight, threshold, acc, precision, fpr, precision, tnr, fnr]
    for i_1 in range(len(list_data)):
        sheet_1.write(row, i_1, list_data[i_1])
    xls_file.save(path_excel)
    row += 1

    return acc

def main():
    global row
    with open(os.path.join(log_dir, 'train_result.txt'),'at') as f:
        f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('weight', 'threshold', 'acc', 'recall', 'fpr', 'precision', 'tnr', 'fnr'))

    sheet_title = ['weight', 'threshold', 'acc', 'recall', 'fpr', 'precision', 'tnr', 'fnr']
    for i_sheet in range(len(sheet_title)):
        sheet_1.write(row, i_sheet, sheet_title[i_sheet])
    xls_file.save(path_excel)
    row += 1

    bounds = [{'name': 'weight', 'type': 'continuous', 'domain': (0,1)},
              {'name': 'threshold', 'type': 'continuous', 'domain': (0,4)}]

    optimizer = GPyOpt.methods.BayesianOptimization(f=accuracy,
                                     domain=bounds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     exact_feval=True,
                                     maximize=True)

    optimizer.run_optimization(max_iter=1024)
    print('plot_acquisition')
    optimizer.plot_acquisition()
    print('plot_convergence')
    optimizer.plot_convergence()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_base_dir', type=str,
                        help='Directory where to write event log.', default='./log')
    parser.add_argument('--emb_dir', type=str,
                        help='Directory of embedding.', default='./emb')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.log_base_dir), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    set_logger(logger, log_dir)

    MAX_POSITIVE_NUM = int(1e5)
    MAX_NEGATIVE_NUM = int(1e6)
    dataset = Dataset(args.emb_dir, MAX_POSITIVE_NUM, MAX_NEGATIVE_NUM)

    path_excel=os.path.join(log_dir, 'train_result.xls')
    xls_file = xlwt.Workbook()
    sheet_1 = xls_file.add_sheet('sheet_1', cell_overwrite_ok=True)
    row = 0
    seed(123)
    main()