from setup import TrainModel
import datetime
import argparse
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(params):

    # basic parameters
    parser = argparse.ArgumentParser()
    
    # mode
    parser.add_argument('--mode', type=str, default="DEL", help='Select what model train: only delineation (DEL), only severity (GRA) or both (multitask)')

    # dataloader and sampler
    parser.add_argument('--encoder_name', type=str, default="resnet50", help='Models encoder that we are using, resnet34, resnet50')
    parser.add_argument('--segmentation_network', type=str, default="unet", help='Type of segmentation network to train')
    parser.add_argument('--crop_size', type=int, default=512, help='Cropped size of input image to network')
    parser.add_argument('--stride', type=int, default=256, help='Stride used in gridGeoSampler')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
    parser.add_argument('--sample_per_epoch', type=int, default=10, help='Number of images sampled in each epoch')
    parser.add_argument('--burn_prop_batch', type=float, default=0.9, help='Proportion of images in each batch with above threshold burned area')
    parser.add_argument('--burn_area_prop', type=float, default=0.4, help='Proportion of area burned in the image')
    
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    
    # loss, optimzer and scheduler
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate used for train')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--lr_policy', type=str, default='Constant', help='scheduler learning rate policy')

    # epochs
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation (epochs)')

    # paths
    parser.add_argument('--dataset_CSV_folder', type=str, default=None, help='path to dataset csv')
    parser.add_argument('--save_model_path', type=str, default="models/", help='path to save model')
    parser.add_argument('--tensorboard_folder', type=str, default="runs/CEMS_trainer", help='path to save tensorboard')

    # checkpoints
    parser.add_argument('--model_checkpoint', type=str, default=None, help='path to checkpoint saved model')
    parser.add_argument('--checkpoint_folder_path', type=str, default="checkpoints/", help='path to pretrained model')

    # date 
    parser.add_argument('--date', type=str, default="20230401_12", help='date execution model')

    # training
    parser.add_argument('--seed', type=int, default=42, help='seed needed to replicate the training')
    parser.add_argument('--validation_folds', nargs='+', type=int, default=1, help='folds used as test set')

    args = parser.parse_args(params)
    # args, unknown = parser.parse_known_args(params)
    print(params)
    # print(f"Unkwown parameters: {unknown}")

    trainModel = TrainModel(args)
    trainModel.start()


if __name__ == '__main__': 

    params = [

        # mode
        '--mode', 'GRA', # DEL, GRA, multitask

        # dataloader and sampler
        '--encoder_name', "resnet50", # resnet50, resnet34, resnet101
        '--segmentation_network', 'unet', # unet, deeplab, unetplus, deeplabplus, pspnet, unetAtt, unetplusAtt
        '--crop_size', '512', #512
        '--stride', '256', #256
        '--batch_size', '12', #12 -> deeplab:8   -> unet:12  -> unetplus:5   -> deeplabplus:12  -> pspnet:20
        '--sample_per_epoch', '240', #120 #240 #480
        '--burn_prop_batch', '0.6', # 0.7
        '--burn_area_prop', '0.3', # 0.4
        
        # loss, optimzer and scheduler
        '--learning_rate', '0.0003', # 0.0003
        '--optimizer', 'adamw', # sgd, adamw
        '--loss', 'dice', # dice, bce, focal
        '--lr_policy', 'MyCosineAnnealing', # Costant, Step_LR, ReduceLROnPlateau, Lambda_rule1, CosineAnnealing, MyCosineAnnealing, CosineAnnealingExponential
        
        # epochs
        '--num_epochs', '100', #100
        '--epoch_start_i', '0',
        '--validation_step', '5', #5 
        '--checkpoint_step', '500', #50

        # hardware
        '--num_workers', '0', 
        '--use_gpu', 'True',

        # model path
        '--dataset_CSV_folder', 'dataset/', # 'dataset/cross_validation/' # 'dataset/'
        '--save_model_path', 'models/',
        '--tensorboard_folder', "_tb_runs/",

        # checkpoints
        # '--model_checkpoint', 'CP8_F6_0405_20.pth',
        '--checkpoint_folder_path', 'checkpoints/',

        # date
        '--date', datetime.datetime.now().strftime('%Y%m%d_%H%M'), #datetime.datetime.today().strftime('%Y%m%d_%H%M'),

        # training
        '--seed', '42',
        '--validation_folds', '5',
    ]
    
    # only if text_folds is passed by command line
    # params.extend(sys.argv[1:])
    main(params)



    #---------------------------------------------- COMMANDS ---------------------------------------------------------------

    # launch this code
    # CUDA_VISIBLE_DEVICES=3 python multitask-segmentation/Semantic\ Segmentation/launch.py
    # CUDA_VISIBLE_DEVICES=2 python launch.py
    # ./launch_CV.sh

    # TENSORBOARD
    # tensorboard --logdir=multitask-segmentation/Semantic\ Segmentation/_tb_runs/ --port=7367
    # tensorboard --logdir=_tb_runs/CV_6F_unet_240000_00030_0406_11
    # tensorboard --logdir=_tb_runs/CV_6folds_GRA_unet_600_0512_09 --port=7367
    # tensorboard --logdir=_tb_runs/CV_6F_unet_57600_00030_0423_20 --port=7367
    # tensorboard --logdir=_tb_runs/ --port=7367

    # launch program
    # CUDA_VISIBLE_DEVICES=3 python launch.py
    # CUDA_VISIBLE_DEVICES=2 python launch.py
    # CUDA_VISIBLE_DEVICES=1 python launch.py
    # CUDA_VISIBLE_DEVICES=0 python launch.py

    # usage GPU
    # nvidia-smi

    # usage CPU/RAM
    # htop