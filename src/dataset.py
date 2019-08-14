
from metric import *
import os
from itertools import combinations, product
import numpy as np


# MAX_POSITIVE_NUM = 10e5
# MAX_NEGATIVE_NUM = 10e6

class Dataset(object):
    # def __init__(self, emb_dir):
    #     self.emb_dir = emb_dir
    #     self.embeddings_and_labels()
    #     self.validation_dataset()

    def __init__(self, emb_dir, max_positive_sample=None, max_negative_sample=None):
        self.validation_dataset_from_path(emb_dir, max_positive_sample, max_negative_sample)
        self.distances_and_issame()

    # return [((model1_emb1, model2_emb2), (model2_emb1, model2_emb2), issame)...]
    def validation_dataset_from_path(self, emb_dir, max_positive_sample, max_negative_sample):
        logger.info("construct pairs...")
        models = os.listdir(emb_dir)
        models.sort()
        models = map(lambda model: os.path.join(emb_dir, model), models)
        models = filter(lambda model: os.path.isdir(model), models)
        persons = os.listdir(models[0])
        # persons.sort()

        # construct positive pairs
        self.positive_pairs = []
        positive_pairs_merge = []
        for person in persons:
            person_paths = map(lambda model: os.path.join(model, person), models)
            images = os.listdir(person_paths[0])
            images.sort()
            positive_pairs_merge = []
            for index, person_path in enumerate(person_paths):
                # logger.debug('index: %d' % (index))
                embeddings = map(lambda image: np.load(os.path.join(person_path, image)), images)
                positive_pairs = list(combinations(embeddings, 2))
                if index == 0:
                    positive_pairs_merge = positive_pairs
                else:
                    positive_pairs_merge = map(lambda x, y: x + (y,), positive_pairs_merge, positive_pairs)

            self.positive_pairs.extend(positive_pairs_merge)
            positive_size =  len(self.positive_pairs)
            if max_positive_sample is not None and positive_size > max_positive_sample:
                logger.debug('positive sample size: %d' % (positive_size))
                break

        # construct negative pairs
        self.negative_pairs = []
        negative_pairs_merge = []
        person_pairs = list(combinations(persons, 2))
        for person_pair in person_pairs:
            for index, model in enumerate(models):
                # logger.debug('index: %d' % (index))
                person_0 = os.path.join(model, person_pair[0])
                files_0 = os.listdir(person_0)
                files_0.sort()
                embeddings_0 = map(lambda x: np.load(os.path.join(person_0, x)), files_0)

                person_1 = os.path.join(model, person_pair[1])
                files_1 = os.listdir(person_1)
                files_1.sort()
                embeddings_1 = map(lambda x: np.load(os.path.join(person_1, x)), files_1)

                negative_pairs = list(product(embeddings_0, embeddings_1))
                if index == 0:
                    negative_pairs_merge = negative_pairs
                else:
                    negative_pairs_merge = map(lambda x, y: x + (y,), negative_pairs_merge, negative_pairs)

            self.negative_pairs.extend(negative_pairs_merge)
            negative_size =  len(self.negative_pairs)
            if max_negative_sample is not None and negative_size > max_negative_sample:
                logger.debug('negative sample size: %d' % (negative_size))
                break

    # return [(model1_distance, model2_distance, issame)...]
    def distances_and_issame(self):
        logger.info("construct distances...")
        self.distances_and_issame_list = []
        for pair in self.positive_pairs:
            distance_triplet = get_distance(pair[0][0], pair[0][1])
            distance_arc = get_distance(pair[1][0], pair[1][1])
            issame = True
            distances_and_issame = (distance_triplet, distance_arc, issame)
            self.distances_and_issame_list.append(distances_and_issame)

        for pair in self.negative_pairs:
            distance_triplet = get_distance(pair[0][0], pair[0][1])
            distance_arc = get_distance(pair[1][0], pair[1][1])
            issame = False
            distances_and_issame = (distance_triplet, distance_arc, issame)
            self.distances_and_issame_list.append(distances_and_issame)

    # # return [((model1_emb1, model2_emb2), (model2_emb1, model2_emb2), issame)...]
    # def validation_dataset_from_path(self, emb_dir, max_positive_sample, max_negative_sample):
    #     logger.info("construct pairs...")
    #     emb_dir_triplet = os.path.join(emb_dir, 'triplet')
    #     emb_dir_arc = os.path.join(emb_dir, 'arc')
    #     dirs = os.listdir(emb_dir_triplet)
    #     # dirs.sort()
    #
    #     # construct positive pairs
    #     self.positive_pairs = []
    #     for dir in dirs:
    #         path_triplet = os.path.join(emb_dir_triplet, dir)
    #         path_arc = os.path.join(emb_dir_arc, dir)
    #         files = os.listdir(path_triplet)
    #         files.sort()
    #         embeddings_triplet = map(lambda x: np.load(os.path.join(path_triplet, x)), files)
    #         embeddings_arc = map(lambda x: np.load(os.path.join(path_arc, x)), files)
    #         positive_pairs_triplet = list(combinations(embeddings_triplet, 2))
    #         positive_pairs_arc = list(combinations(embeddings_arc, 2))
    #         positive_pairs_merge = map(lambda x, y: (x, y, True), positive_pairs_triplet, positive_pairs_arc)
    #         self.positive_pairs.extend(positive_pairs_merge)
    #         positive_size =  len(self.positive_pairs)
    #         if max_positive_sample is not None and positive_size > max_positive_sample:
    #             logger.debug('positive sample size: %d' % (positive_size))
    #             break
    #
    #     # construct negative pairs
    #     self.negative_pairs = []
    #     dir_pairs = list(combinations(dirs, 2))
    #     for dir_pair in dir_pairs:
    #         path_triplet_0 = os.path.join(emb_dir_triplet, dir_pair[0])
    #         path_arc_0 = os.path.join(emb_dir_arc, dir_pair[0])
    #         files_0 = os.listdir(path_triplet_0)
    #         files_0.sort()
    #         embeddings_triplet_0 = map(lambda x: np.load(os.path.join(path_triplet_0, x)), files_0)
    #         embeddings_arc_0 = map(lambda x: np.load(os.path.join(path_arc_0, x)), files_0)
    #
    #         path_triplet_1 = os.path.join(emb_dir_triplet, dir_pair[1])
    #         path_arc_1 = os.path.join(emb_dir_arc, dir_pair[1])
    #         files_1 = os.listdir(path_triplet_1)
    #         files_1.sort()
    #         embeddings_triplet_1 = map(lambda x: np.load(os.path.join(path_triplet_1, x)), files_1)
    #         embeddings_arc_1 = map(lambda x: np.load(os.path.join(path_arc_1, x)), files_1)
    #
    #         negative_pairs_triplet = list(product(embeddings_triplet_0, embeddings_triplet_1))
    #         negative_pairs_arc = list(product(embeddings_arc_0, embeddings_arc_1))
    #         negative_pairs_merge = map(lambda x, y: (x, y, False), negative_pairs_triplet, negative_pairs_arc)
    #         self.negative_pairs.extend(negative_pairs_merge)
    #         negative_size =  len(self.negative_pairs)
    #         if max_negative_sample is not None and negative_size > max_negative_sample:
    #             logger.debug('negative sample size: %d' % (negative_size))
    #             break
    #
    #     self.all_pairs = []
    #     self.all_pairs.extend(self.positive_pairs)
    #     self.all_pairs.extend(self.negative_pairs)

    # # return [(model1_distance, model2_distance, issame)...]
    # def distances_and_issame(self):
    #     logger.info("construct distances...")
    #     self.distances_and_issame_list = []
    #     for pair in self.all_pairs:
    #         distance_triplet = get_distance(pair[0][0], pair[0][1])
    #         distance_arc = get_distance(pair[1][0], pair[1][1])
    #         issame = pair[2]
    #         distances_and_issame = (distance_triplet, distance_arc, issame)
    #         self.distances_and_issame_list.append(distances_and_issame)

    # def validation_dataset_from_config(self, args):
    #     pass
    #
    # def embeddings_and_labels(self):
    #     embeddings =  np.load(os.path.join(self.emb_dir, "signatures.npy"))
    #     labels = np.load(os.path.join(self.emb_dir, "labels_name.npy"))
    #     assert len(embeddings) == len(labels)
    #     embeddings = np.array(embeddings)
    #     labels = np.array(labels).reshape(-1,1)
    #     embeddings_and_labels = np.hstack((labels, embeddings))
    #     self.all_pairs = list(combinations(embeddings_and_labels, 2))
    #     np.random.shuffle(self.all_pairs)
    #
    # def validation_dataset(self):
    #     p_num = 0
    #     n_num = 0
    #     validation_pairs = []
    #     for pair in self.all_pairs:
    #         if pair[0][0] == pair[1][0] and p_num < MAX_POSITIVE_NUM:
    #             validation_pairs.append(pair)
    #             p_num += 1
    #         elif pair[0][0] != pair[1][0] and n_num < MAX_NEGATIVE_NUM:
    #             validation_pairs.append(pair)
    #             n_num += 1
    #         elif p_num == MAX_POSITIVE_NUM and n_num == MAX_NEGATIVE_NUM:
    #             break
    #     logger.debug ('positive: %d, negative: %d' % (p_num, n_num))
    #
    #     self.distance_list = []
    #     self.issame_list = []
    #     for pair in validation_pairs:
    #         ground_truth_issame = True
    #         if pair[0][0] != pair[1][0]:
    #             actual_issame = False
    #         dist = get_distance(pair[0][1:], pair[1][1:])
    #         self.distance_list.append(dist)
    #         self.issame_list.append(ground_truth_issame)

    # def batch_gen(self, data, batch_size):
    #     data_size = len(data)
    #     idx = 0
    #     while idx < data_size:
    #         start = idx
    #         idx += batch_size
    #         yield data[start:start + batch_size]
    #
    # def get_batch(self):
    #     BATCH_SIZE = 64
    #     gen = self.batch_gen(self.distance_list, BATCH_SIZE)
    #     n_batches = int(np.ceil(len(self.distance_list) / BATCH_SIZE))
    #     for index in range(n_batches):
    #         bat = next(gen)
    #         # TODO

# dataset: [('bob',['bob_01.png',...]),...]
# def get_supervised_dataset(path):
#     path_exp = os.path.expanduser(path)
#
#     dataset = []
#     path_dir_exp = os.path.join(path_exp)
#     classes = [path for path in os.listdir(path_dir_exp) \
#                if os.path.isdir(os.path.join(path_dir_exp, path))]
#     classes.sort()
#     nrof_classes = len(classes)
#     for i in range(nrof_classes):
#         class_name = classes[i]
#         facedir = os.path.join(path_dir_exp, class_name)
#         image_paths = facenet.get_image_paths(facedir)
#         dataset.append(facenet.ImageClass(class_name, image_paths))
#
#     # logger.debug(dataset)
#     return dataset
#
# def get_image_paths(facedir):
#     image_paths = []
#     if os.path.isdir(facedir):
#         images = os.listdir(facedir)
#         for img in images:
#             image_path = os.path.join(facedir, img)
#             if os.path.isdir(image_path) == False:
#                 image_paths.append(image_path)
#
#     # logger.debug("image_paths: %s" % (image_paths))
#     return image_paths
#
#
# def get_paths(lfw_dir, pairs, file_ext):
#     nrof_skipped_pairs = 0
#     path_list = []
#     issame_list = []
#     for pair in pairs:
#         if len(pair) == 3:
#             path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
#             path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
#             issame = True
#         elif len(pair) == 4:
#             path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
#             path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
#             issame = False
#         if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
#             path_list += (path0, path1)
#             issame_list.append(issame)
#         else:
#             nrof_skipped_pairs += 1
#     if nrof_skipped_pairs > 0:
#         print('Skipped %d image pairs' % nrof_skipped_pairs)
#
#     return path_list, issame_list
#
#
# def read_pairs(pairs_filename):
#     pairs = []
#     with open(pairs_filename, 'r') as f:
#         for line in f.readlines()[1:]:
#             pair = line.strip().split()
#             pairs.append(pair)
#     return np.array(pairs)