import os
import argparse
import random
import shutil
from shutil import copyfile
from utils import printProgressBar

############      创建文件夹     ############

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)

############ 按比例随机划分数据集 ############

def main(config):

    rm_mkdir(config.train_valid_path)
    rm_mkdir(config.train_valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]
            data_list.append('*_' + filename + '.jpg')            # *代表文件名。例如在ISIC_0000.jpg中，*代表ISIC
            GT_list.append('*_' + filename + '_segmentation.pnp')

    num_total = len(data_list)
    num_train_valid = int((config.train_valid_ratio / (config.train_valid_ratio + config.test_ratio)) * num_total)
    num_test = num_total - num_train_valid

    print('\nNum of train set : ', num_train_valid)
    print('\nNum of test set : ', num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train_valid):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_valid_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_valid_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_train_valid, prefix='Producing train set:', suffix='Complete', length=50)

    for i in range(num_test):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.test_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.test_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_test, prefix='Producing test set:', suffix='Complete', length=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_valid_ratio', type=float, default=5 / 6)
    parser.add_argument('--test_ratio', type=float, default=1 / 6)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='.\data_DS/DS/')
    parser.add_argument('--origin_GT_path', type=str, default='.\data_DS/DS_labels/')

    parser.add_argument('--train_valid_path', type=str, default='./data_DS/train_valid/')
    parser.add_argument('--train_valid_GT_path', type=str, default='./data_DS/train_valid_GT/')
    parser.add_argument('-test_path', type=str, default='./data_DS/test/')
    parser.add_argument('-test_GT_path', type=str, default='./data_DS/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)