# MSU_Net
MSU-Net: Multi-scale U-Net for 2D Medical Image Segmentation \
Article：https://www.frontiersin.org/articles/10.3389/fgene.2021.639930/full

![Figure_6_Segmentation_Result](https://user-images.githubusercontent.com/50656765/125736033-501fa324-f6bf-4622-b124-41d4ee497811.jpg)
****

# Quick Start Examples

## Install
### Python >= 3.6.0 required with all requirements.txt dependencies installed

## Package
### Pytorch >= 1.0 or >= 1.2

## Data
### Datasets can be linked to in the paper

Note:different datasets have different epochs. It's about 20,30,50.
****

# Train and Test

      -data\
      ---EM\
      ---EM_labels\
      -data_* \
      ---trian_valid\
      ---train_valid_GT\
      ---test\
      ---test_GT\

**1. Configure your environment**

**2. The dataset directory is shown above. Datasets can be divided randomly by dataset.py**

      """
      # model hyper-parameters
      parser.add_argument('--train_valid_ratio', type=float, default= *)
      parser.add_argument('--test_ratio', type=float, default= *)

      # data path
      parser.add_argument('--origin_data_path', type=str, default='.\data_EM/EM/')
      parser.add_argument('--origin_GT_path', type=str, default='.\data_EM/EM_labels/')

      parser.add_argument('--train_valid_path', type=str, default='./data_EM/train_valid/')
      parser.add_argument('--train_valid_GT_path', type=str, default='./data_EM/train_valid_GT/')
      parser.add_argument('-test_path', type=str, default='./data_EM/test/')
      parser.add_argument('-test_GT_path', type=str, default='./data_EM/test_GT/')

**3. Set the path and hpyer-parameters in train.py and test.py**
      """
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

**4. Run train.py and test.py in the terminal's current directory**

Note：The segmentation results of the training, validation and testing can be obtained by modifying slover.py

****
