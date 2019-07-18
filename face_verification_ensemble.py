
import argparse
import os.path
import sys
from datetime import datetime
from itertools import combinations
import numpy as np
from numpy.random import seed
import matplotlib
from sklearn import metrics
import GPy
import GPyOpt
import dataset
from dataset import Dataset
from metric import *
from log import *


# objective function
def accuracy(parameters):
    weight = parameters[:, 0]
    threshold = parameters[:, 1]
    logger.debug("weight: %f, threshold: %f" % (weight, threshold))
    # emb_dir_triplet = os.path.join(os.path.expanduser(args.emb_dir), 'triplet')
    # emb_dir_arc = os.path.join(os.path.expanduser(args.emb_dir), 'arc')
    # dataset_triplet = Dataset(emb_dir_triplet)
    # dataset_arc = Dataset(emb_dir_arc)
    dataset = Dataset(args.emb_dir)
    distances_and_issames = dataset.distances_and_issame_list
    logger('distances_and_issames: %s' % (distances_and_issames))
    distance_list = []
    issame_list = []
    for distances_and_issame in distances_and_issames:
        distance = distances_and_issame[0] * weight + distances_and_issame[1] * (1 - weight)
        issame = distances_and_issame[2]
        distance_list.append(distance)
        issame_list.append(issame)

    accuracy = calculate_accuracy(threshold, distance_list, issame_list)
    logger.debug('accuracy: %f' % (accuracy))
    return accuracy


def main():
    bounds = [{'name': 'weight', 'type': 'continuous', 'domain': (0,1)},
              {'name': 'threshold', 'type': 'continuous', 'domain': (0,4)}]

    optimizer = GPyOpt.methods.BayesianOptimization(f=accuracy,
                                     domain=bounds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     exact_feval=True,
                                     maximize=True)

    optimizer.run_optimization(max_iter=100)
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
    main()