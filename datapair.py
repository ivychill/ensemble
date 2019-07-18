import argparse
import sys
import os, shutil
import numpy as np


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def gen_pairs(path, cycle, num):
    dataset = get_dataset(path)
    pairspath = os.path.join(path,'pairs.txt')
    file = open(pairspath, 'w')
    file.write(str(cycle) + '    ' + str(num) + '\n')

    for i in range(cycle):
        oo = 0
        while oo < num:
            if len(dataset) > 1:
                num_cls = np.random.randint(0, len(dataset))
                cls = dataset[num_cls]
                if len(cls.image_paths)>0:
                    # split source and target, source: *_00xx.png, target: *_01xx.png
                    id_images = [s for s in cls.image_paths if '_00' in s]
                    im_no1 = np.random.randint(0, len(id_images))
                    im_no2 = np.random.randint(len(id_images), len(cls.image_paths))
                    sort_paths = np.sort(cls.image_paths)
                    print(sort_paths[im_no1].split('.')[-2])
                    no1 = int(sort_paths[im_no1].split('.')[-2][-4:])
                    no2 = int(sort_paths[im_no2].split('.')[-2][-4:])
                    if im_no2 != im_no1:
                        file.write(cls.name + '    ' + str(no1) + '    ' + str(no2) + '\n')
                        oo = oo + 1
        nn = 0
        while nn < num:
            cls_no1 = np.random.randint(0, len(dataset))
            cls_no2 = np.random.randint(0, len(dataset))
            if cls_no1 != cls_no2:
                cls1 = dataset[cls_no1]
                cls2 = dataset[cls_no2]
                if len(cls1.image_paths) > 0 and len(cls2.image_paths) > 0:
                    im_no1 = np.random.randint(0, len(cls1.image_paths))
                    sort_paths = np.sort(cls1.image_paths)
                    no1 = int(sort_paths[im_no1].split('.')[-2][-4:])
                    sort_paths = np.sort(cls2.image_paths)
                    im_no2 = np.random.randint(0, len(cls2.image_paths))
                    no2 = int(sort_paths[im_no2].split('.')[-2][-4:])
                    file.write(cls1.name + '    ' + str(no1) + '    ' + cls2.name + '    ' + str(no2) + '\n')
                    nn = nn + 1

    file.flush()
    file.close()

def main_txt(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gen_pairs(args.input_dir, 10, 300)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', default='/opt/yanhong.jia/datasets/lfw_align_160', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('--output_dir', default='.', type=str,
                        help='Directory with aligned face thumbnails.')
    parser.add_argument('--mode', default='pairs', type=str,
                        help='gen_pairs.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main_txt(parse_arguments(sys.argv[1:]))