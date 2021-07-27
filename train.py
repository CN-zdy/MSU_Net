import os
import glob
import warnings
import argparse
from utils import *
from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader


warnings.filterwarnings("ignore")

def read_image_file(str):
    files_image = glob.glob(str + '/*.jpg')
    return files_image

def read_label_file(str):
    files_label = glob.glob(str + '/*_segmentation.png')
    return files_label

def main(config):

    cudnn.benchmark = True
    if config.model_type not in ['U_Net','MSU_Net']:
        print(
             'ERROR!! model_type should be selected in U_Net/MSU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    config.train_result_path = os.path.join(config.train_result_path, config.model_type)
    if not os.path.exists(config.train_result_path):
        os.makedirs(config.train_result_path)

    for i in range(config.k):

        tv_image = read_image_file('./data/train_valid/')
        tv_label = read_label_file('./data/train_valid_GT/')

        X_train, y_train, X_valid, y_valid = k_fold_data(config.k, i, tv_image, tv_label)

        print('*' * 25, '第', i + 1, '折', '*' * 25)

        config.model_path = ('./model/%s/第-%1d-折' % (config.model_type, i + 1))
        os.makedirs(config.model_path)
        print('Create path - %s' % config.model_path)

        config.train_result_path = ('./results/%s/train_result/第-%1d-折' % (config.model_type, i + 1))
        os.makedirs(config.train_result_path)
        print('Create path - %s' % config.train_result_path)

        config.val_result_path = ('./results/%s/val_result/第-%1d-折' % (config.model_type, i + 1))
        os.makedirs(config.val_result_path)
        print('Create path - %s \n' % config.val_result_path)

        lr = 0.01
        epoch = 100

        config.num_epochs = epoch
        config.lr = lr

        print(config)

        train_loader = get_loader(imList=X_train,
                                  labelList=y_train,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  drop_last=True)
        valid_loader = get_loader(imList=X_valid,
                                  labelList=y_valid,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='valid',
                                  drop_last=True)
        test_loader = []

        solver = Solver(config, train_loader, valid_loader, test_loader)

        # Train and sample the images
        if config.mode == 'train':
            solver.train()
        elif config.mode == 'test':
            solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='train')

    # model_type and path
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/MSU_Net')
    parser.add_argument('--model_path', type=str, default='./model/')

    parser.add_argument('--train_result_path', type=str, default='./results/model/train_result/')
    parser.add_argument('--val_result_path', type=str, default='./results/model/val_result/')
    parser.add_argument('--test_result_path', type=str, default='./results/model/test_result/')

    parser.add_argument('--train_valid_path', type=str, default='./data/train_valid/')
    parser.add_argument('--test_path', type=str, default='./data/test/')

    # hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs_test', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4) # 4
    parser.add_argument('--k', type=int, default=5)  # 5
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    config = parser.parse_args()
    # print(config)
    main(config)
