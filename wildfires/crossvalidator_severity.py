from pathlib import Path

# from kornia import augmentation as K
import kornia as K
from kornia.augmentation import PatchSequential, ImageSequential
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import collections
from torch import Tensor
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast, overload

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss, FocalLoss# Reference https://smp.readthedocs.io/en/latest/losses.html
import torch

import torchvision
import numpy as np
np.set_printoptions(precision=5)
torch.set_printoptions(precision=5)
import os
import csv
import time
from datetime import datetime

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, MulticlassJaccardIndex, MulticlassF1Score

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from wildfires.dataset import DatasetCEMS_Severity
from wildfires.settings import Settings
from wildfires.geoSampler import ConstrainedRandomBatchGeoSampler 
from wildfires.scheduler import MyScheduler
from wildfires.utils import save_log2df, save_model_configuration, set_seed, save_configuration_tb, compute_confusion_matrix, \
                            compute_prec_recall_f1_acc, weight_init,  remap_severity_labels, to_RGB_Mask ,compute_squared_errors
from wildfires.model_multitask import Unet_Segmentation_multitask, UnetPlusPlus_Segmentation_multitask, DeepLabV3_Segmentation_multitask, DeepLabV3Plus_Segmentation_multitask, PSPNet_Segmentation_multitask
from wildfires.losses import My_MSELoss

from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import stack_samples


class CrossValidator_Severity:
    
    def __init__(self, args):

        self.args = args
        self.line_csv = ""

        self.cfg = Settings()
        set_seed(self.args.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.args.model_checkpoint is not None:
            self.args.date = '_'.join(self.args.model_checkpoint.replace('.pth','').split('_')[-2:])
        else:
            self.args.date = datetime.strptime(self.args.date, '%Y%m%d_%H%M')
    

    def start_cross_validation(self):

        for fold in self.args.validation_folds:
            fold = int(fold)

            if self.args.segmentation_network == 'unet':
                self.model = smp.Unet(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.args.segmentation_network == 'unetplus':
                self.model = smp.UnetPlusPlus(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.args.segmentation_network == 'deeplab':
                self.model = smp.DeepLabV3(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.args.segmentation_network == 'deeplabplus':
                self.model = smp.DeepLabV3Plus(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.args.segmentation_network == 'pspnet':
                self.model = smp.PSPNet(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            else:
                print("Wrong segmentation_network name")
                return
            
            self.model.apply(weight_init)
            self.model.to(self.device)
            
            # Define Kornia transformation
            # self.train_transform = K.augmentation.AugmentationSequential(
            #             K.augmentation.RandomVerticalFlip(p=0.5),
            #             K.augmentation.RandomHorizontalFlip(p=0.5),
            #             K.augmentation.RandomRotation(degrees=90, p=0.5),
            #             K.augmentation.RandomRotation(degrees=270, p=0.5),
            #             data_keys=['input', 'mask', 'mask'],
            #             )
            self.train_transform = A.Compose([
                                        A.Flip(p=0.5),
                                        A.Affine(
                                            translate_percent=0.2,
                                            scale=(0.8, 1.2),
                                            rotate=360,
                                            shear=(-20,20),
                                            mode=cv2.BORDER_REFLECT_101,
                                            p=0.5,
                                        ),
                                        A.RandomBrightnessContrast(
                                            brightness_limit=0.1,
                                            contrast_limit=0.1,
                                            p=0.5,
                                        ),
                                        ToTensorV2(transpose_mask = True),
                                    ],

                                    additional_targets={"grading": "mask"},
                                )


            # Create datasets and dataloaders
            self.train_dataset = DatasetCEMS_Severity(root=Path(self.cfg.rootData), annotation_type = self.cfg.annotation_type_severity,  csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.dataset_path_crossValidation_severity), transforms=self.train_transform, cache=False, fold_test=fold, test_set=False, severity_convention=self.cfg.severity_convention) # transforms=self.train_transform, 
            self.val_dataset = DatasetCEMS_Severity(root=Path(self.cfg.rootData), annotation_type = self.cfg.annotation_type_severity, csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.dataset_path_crossValidation_severity), cache=False, fold_test=fold, test_set=True, severity_convention=self.cfg.severity_convention)
            self.test_dataset = DatasetCEMS_Severity(root=Path(self.cfg.rootData), annotation_type = self.cfg.annotation_type_severity, csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.dataset_path_crossValidation_severity), cache=False, fold_test=0, test_set=True, severity_convention=self.cfg.severity_convention)
            print(f"\nTraining length: \t{len(self.train_dataset)}")
            print(f"Validation length: \t{len(self.val_dataset)}")
            print(f"Test length: \t\t{len(self.test_dataset)}\n")

            # Create geosampler
            self.train_sampler = ConstrainedRandomBatchGeoSampler(  dataset=self.train_dataset, 
                                                                    batch_size=self.args.batch_size, 
                                                                    size=self.args.crop_size, 
                                                                    length=self.args.sample_per_epoch, 
                                                                    burn_prop_batch=self.args.burn_prop_batch ,
                                                                    burn_area_prop=self.args.burn_area_prop,
                                                                    mask = "gra_mask")
            self.val_sampler = GridGeoSampler(self.val_dataset, size=self.args.crop_size, stride=self.args.stride)
            self.test_sampler = GridGeoSampler(self.test_dataset, size=self.args.crop_size, stride=self.args.stride)
            print(f"Sampler Training: \t{len(self.train_sampler)}")
            print(f"Sampler Validation: \t{len(self.val_sampler)}")
            print(f"Sampler Test: \t\t{len(self.test_sampler)}\n")

            # Create a PyTorch DataLoader for the training set with the sampler
            self.train_dataloader = DataLoader(self.train_dataset, batch_sampler = self.train_sampler, num_workers = self.args.num_workers, collate_fn=stack_samples)
            self.val_dataloader = DataLoader(self.val_dataset, sampler = self.val_sampler, batch_size = self.args.batch_size, collate_fn=stack_samples)
            self.test_dataloader = DataLoader(self.test_dataset, sampler = self.test_sampler, batch_size = self.args.batch_size, collate_fn=stack_samples)
            print(f"Dataloader Training batches: \t{len(self.train_dataloader)}")
            print(f"Dataloader Validation batches: \t{len(self.val_dataloader)}")
            print(f"Dataloader Test batches: \t{len(self.test_dataloader)}\n")

            # Define the loss function and optimizer
            if self.args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
            elif self.args.optimizer == "adamw":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay = 0.001)

            if self.args.loss == "bce":
                self.loss_criterion = SoftBCEWithLogitsLoss(pos_weight=torch.tensor(1.5), ignore_index=255)
            elif self.args.loss == "dice":
                self.loss_criterion = DiceLoss(from_logits = False, ignore_index= 255, mode='binary', eps=0.000001)
            elif self.args.loss == "focal":
                self.loss_criterion = FocalLoss(ignore_index= 255, mode='binary', alpha=0.5, gamma=2.0)

            self.regressor_criterion = My_MSELoss(reduction='mean', ignore_index = 255 )

            # Scheduler https://github.com/mpyrozhok/adamwr
            self.scheduler = MyScheduler(self.optimizer, lr_policy = self.args.lr_policy, initial_step = 30, step_size=15, gamma=0.2, stability=50)

            self.iou_delineation = BinaryJaccardIndex(ignore_index=255).to(self.device)
            self.f1_delineation = BinaryF1Score(ignore_index=255).to(self.device)
            self.iou_severity = MulticlassJaccardIndex(num_classes=self.cfg.severity_convention+1, ignore_index=255, average = 'none').to(self.device)
            self.f1_severity = MulticlassF1Score(num_classes=self.cfg.severity_convention+1, ignore_index=255, average = 'none').to(self.device)

            # Paths
            self.model_name_folder = 'CV_{}_GRA_{}_{}_{}/'.format( self.cfg.dataset_path_crossValidation_severity.split("_")[1].replace("folds","F"), self.args.segmentation_network, self.args.num_epochs * self.args.sample_per_epoch, str(self.args.date.strftime( '%m%d_%H' )))
            self.model_name = 'CV_foldTest_{}.pth'.format(fold)
            self.model_save_path = self.args.save_model_path + self.model_name_folder + self.model_name
            self.model_runs_path = self.cfg.tb_runs_configuration_folder + self.model_name_folder + self.model_name.replace(".pth", "/") # where save tensorboard in format csv
            self.tensorboard_path = self.args.tensorboard_folder + self.model_name_folder + self.model_name.replace(".pth", "")

            if not os.path.exists(os.path.dirname(self.model_runs_path)):
                os.makedirs(os.path.dirname(self.model_runs_path))

            if not os.path.exists(os.path.dirname(self.args.save_model_path + self.model_name_folder)):
                os.makedirs(os.path.dirname(self.args.save_model_path + self.model_name_folder))
            # print(f'model path: {model_save_path}')
            # print(f'tb csv folder {model_runs_path}')
            # print(f'tb path: {tensorboard_path}')

            self.start_train(fold)


    def start_train(self, fold: int):

        print("\n", "-" * 100, sep="")
        print(f"VALIDATION fold: {fold}")
        print("-" * 100, "\n", sep="")

        epoch_number = 1

        # ----- CHECKPOINT LOADING -----
        if self.args.checkpoint_folder_path is not None and self.args.model_checkpoint is not None:

            if not os.path.exists(self.args.checkpoint_folder_path + self.args.model_checkpoint):
                print("\n", "`" * 100, sep="")
                print(f"Wrong configuration: checkpoint {self.args.checkpoint_folder_path + self.args.model_checkpoint} does not exists")
                print("," * 100, "\n", sep="")
                return
        
            print("\n", "*" * 100, sep="")
            print("Loading checkpoint...")
            checkpoint = torch.load(os.path.join(self.args.checkpoint_folder_path, self.args.model_checkpoint))

            # Load the model and optimizer states from the checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_number = checkpoint['epoch'] + 1 # start from next epoch 
            self.tensorboard_path = checkpoint['log_dir']
            self.scheduler = checkpoint['scheduler']

            print("Done!")
            print("*" * 100, "\n", sep="")

        self.writer = SummaryWriter(self.tensorboard_path)

        # ----- EPOCH CYCLE -----
        for _ in range(epoch_number, self.args.num_epochs+1):
            print(' --- EPOCH {} - Fold {}'.format( epoch_number, fold ), end=" ")

            # ----- TRAIN -----
            start_train_epoch = time.time()

            self.model.train()
            avg_training_loss, proportion_burned_pixels, avg_train_loss_delineation, avg_train_rmse_severity = self.train_one_epoch(epoch_number, self.writer)

            end_train_epoch = time.time()
            print("--- TIME ELAPSED: {:.3f} sec --- Proportion burned area: {:.3f}%".format((end_train_epoch - start_train_epoch), proportion_burned_pixels*100), end=" ")
            print(f" - Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            # ----- VALIDATION -----
            if epoch_number % self.args.validation_step == 0 and epoch_number != 0:
                
                print(f"Validation: --- epoch: { epoch_number } - fold: {fold}", end = ' ')

                avg_val_loss, avg_val_loss_delineation, avg_rmse_severity = self.validate_model(epoch_number) # validation model

                if self.args.lr_policy == "ReduceLROnPlateau":
                    self.scheduler.step(epoch_number, avg_val_loss)

                print(f" - LR: {self.optimizer.state_dict()['param_groups'][0]['lr']}", end = ' ')
                print('--- LOSSES: train {:.6f} - validation {:.6f}'.format(avg_training_loss, avg_val_loss))

                # Log the running loss averaged per batch for both training and validation
                self.writer.add_scalar('LossTrain/Training_loss',  avg_training_loss, epoch_number )
                self.writer.add_scalar('LossTrain/Training_rmse_loss_severity',  avg_train_rmse_severity, epoch_number )
                self.writer.add_scalar('LossTrain/Training_loss_delineation',  avg_train_loss_delineation, epoch_number )
                self.writer.add_scalar('LossVal/Validation_loss', avg_val_loss, epoch_number )
                self.writer.add_scalar('LossVal/Validation_rmse_loss_severity', avg_rmse_severity, epoch_number )
                self.writer.add_scalar('LossVal/Validation_loss_delineation', avg_val_loss_delineation, epoch_number )
                self.writer.add_scalar('optimizer/Learning_rate', self.optimizer.state_dict()["param_groups"][0]["lr"], epoch_number )
                self.writer.add_scalar('metrics/Burned_pixels_epoch', proportion_burned_pixels, epoch_number )
                self.writer.add_scalar('metrics_Delineation/IOU',  self.iou_delineation.compute() , epoch_number )
                self.writer.add_scalar('metrics_Delineation/F1', self.f1_delineation.compute() , epoch_number )
                self.writer.add_scalar('metrics_Grading/F1', np.mean(self.f1_severity.compute().cpu().numpy()) , epoch_number )
                self.writer.add_scalar('metrics_Grading/IOU', np.mean(self.iou_severity.compute().cpu().numpy()) , epoch_number )
                self.writer.flush()

                self.iou_delineation.reset()
                self.f1_delineation.reset()
                self.iou_severity.reset()
                self.f1_severity.reset() 
            
            # ----- CHECKPOINT SAVING -----
            if epoch_number % self.args.checkpoint_step == 0 and epoch_number != 0:
                print("\n", "*" * 100, sep="")
                print(f"Epoch {epoch_number} - Saving checkpoint ...")
                if not os.path.isdir(self.args.checkpoint_folder_path):
                    os.mkdir(self.args.checkpoint_folder_path)

                checkpoint = {
                    'epoch': epoch_number,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'log_dir': self.tensorboard_path,
                    'scheduler': self.scheduler
                }

                model_name_checkpoint = 'CP{}_F{}_GRA_{}.pth'.format( epoch_number, fold, str(self.args.date.strftime( '%m%d_%H' )) )

                torch.save( checkpoint, os.path.join(self.args.checkpoint_folder_path, model_name_checkpoint) )
                print("Done!")
                print("*" * 100, "\n", sep="")

            if self.args.lr_policy != "ReduceLROnPlateau":
                self.scheduler.step(epoch_number)

            epoch_number+=1
            
        # ----- TEST -----
        print(f"\n--- Final TESTING --- epoch: {epoch_number} --- date:{datetime.today().strftime('%Y/%m/%d_%H:%M:%S')}\n")
        avg_test_loss, avg_test_rmse_loss, test_accuracy_delineation, test_rmse, conf_matr_delineation, conf_matr_severity = self.test_model()
        
        test_f1_delineation = self.f1_delineation.compute()
        test_IoU_delineation = self.iou_delineation.compute()
        test_f1_severity = self.f1_severity.compute()
        test_IoU_severity = self.iou_severity.compute()

        precision_delineation, recall_delineation, f1score_delineation, accuracy_delineation = compute_prec_recall_f1_acc(conf_matr_delineation)
        precision_severity, recall_severity, f1score_severity, accuracy_severity = compute_prec_recall_f1_acc(conf_matr_severity)

        # save model and its configuration, also convert tensorboard to dataframe and save it as csv
        torch.save(self.model.state_dict(), self.model_save_path)

        tensorboard_csv = self.model_runs_path + self.model_name.replace(".pth", ".csv")
        json_parameters_path = self.model_runs_path + "configuration.json"
        
        # save hyperparameters and configuration in tensorboard
        self.writer = save_configuration_tb(self.writer, self.args, test_IoU_delineation=test_IoU_delineation, test_f1_delineation=test_f1_delineation, test_rmse=test_rmse, test_IoU_severity=test_IoU_severity, test_f1_severity=test_f1_severity, conf_matr_delineation=conf_matr_delineation, conf_matr_severity=conf_matr_severity, cfg=self.cfg)
        save_log2df(self.tensorboard_path, tensorboard_csv)
        save_model_configuration(json_parameters_path, self.args, test_IoU_delineation=test_IoU_delineation, test_f1_delineation=test_f1_delineation, test_rmse=test_rmse, test_IoU_severity=test_IoU_severity, test_f1_severity=test_f1_severity, conf_matr_delineation=conf_matr_delineation, conf_matr_severity=conf_matr_severity, cfg=self.cfg)

        # Print final results
        print(' - Test Loss: {:.6f}\n - Test Acc delineation: {:.6f}%'.format(avg_test_loss, test_accuracy_delineation))
        print("\n***** DELINEATION *****\n")
        print(' - Test IoU delineation: {:.6f}%\n - Test F1 delineation: {:.6f}%'.format(test_IoU_delineation *100, test_f1_delineation *100 ))
        print(f'Confusion matrix delineation:  \n{conf_matr_delineation}\n')
        print(f'Delineation test precision: {precision_delineation *100}\nDelineation test recall: {recall_delineation *100}\nDelineation test F1 delineation: {f1score_delineation *100}\nDelineation test accuracy: {accuracy_delineation *100}')
        print("\n***** SEVERITY *****\n")
        print(f' - Test rmse_loss: {avg_test_rmse_loss}')
        print(f" - Test RMSE severity: {test_rmse[:self.cfg.severity_convention+1]} - mean: {np.mean(test_rmse[:self.cfg.severity_convention+1])} - total: {test_rmse[self.cfg.severity_convention+1]}")
        print(f' - Test IoU severity: {test_IoU_severity.cpu().numpy() *100}% - mean: {np.mean(test_IoU_severity.cpu().numpy() *100)}% \n - Test F1 severity: {test_f1_severity.cpu().numpy() *100}% - mean: {np.mean(test_f1_severity.cpu().numpy() *100)}%')
        print(f'Confusion matrix severity:  \n{conf_matr_severity}\n')
        print(f'Severity test precision: {precision_severity *100}\nSeverity test recall: {recall_severity *100}\nSeverity test F1: {f1score_severity *100}\nSeverity test accuracy: {accuracy_severity *100}')

        self.iou_delineation.reset()
        self.f1_delineation.reset()
        self.iou_severity.reset()
        self.f1_severity.reset()

        # Close SummaryWriter
        self.writer.close()


    def train_one_epoch(self, epoch_index, tb_writer):

        running_loss = 0.
        running_loss_delineation = 0.
        running_rmse_severity = 0.
        num_batches = 0
        count_pixel_burned = 0
        count_void_images = 0

        burned_pixels_batch = 0
        burned_pixels_total = 0
        pixels_total = 0
        
        for i, batch in enumerate(self.train_dataloader):

            images = batch["image"].to(self.device)
            labels_severity = batch["gra_mask"].to(self.device)
 
            # count number pixel burned per batch
            for label_mask in labels_severity:
                count_pixel_burned = ((label_mask > 0) & (label_mask < 120)).sum() 
                if count_pixel_burned == 0:
                    count_void_images+=1
                burned_pixels_batch += count_pixel_burned
            
            # print(f"Batch: {i+1}, burnedPixels: {burned_pixels_batch}, percentage: {burned_pixels_batch/labels.numel()}, void images: {count_void_images}, number_pixel_batch: {labels.numel()}")
            burned_pixels_total += burned_pixels_batch
            pixels_total += labels_severity.numel()
            burned_pixels_batch = 0
        
            logits_severity = self.model(images)

            pred_severity = torch.sigmoid(logits_severity)
            pred_severity_remapped = remap_severity_labels(pred_severity, num_labels = self.cfg.severity_convention)
            labels_severity_remapped = remap_severity_labels(labels_severity, num_labels = self.cfg.severity_convention)

            labels_delineation =  (torch.sigmoid(labels_severity_remapped) >= 1).int()
            pred_delineation = (torch.sigmoid(pred_severity_remapped) >= 1).int()
            loss_delineation = self.loss_criterion(pred_delineation, labels_delineation)

            loss_severity = self.regressor_criterion(pred_severity, labels_severity)
            loss = loss_delineation + loss_severity

            loss.backward()                 # Back Propagation
            self.optimizer.step()           # Gardient Descent
            self.optimizer.zero_grad()
            
            running_loss += loss.item()
            running_loss_delineation += loss_delineation.item()
            running_rmse_severity += torch.sqrt(loss_severity).item()
            num_batches += 1

        avg_train_loss = running_loss / (num_batches - 1) # average loss per batch
        avg_train_loss_delineation = running_loss_delineation / (num_batches - 1)
        avg_train_rmse_severity = running_rmse_severity / (num_batches - 1) # average rmse per batch

        return avg_train_loss, burned_pixels_total/pixels_total, avg_train_loss_delineation, avg_train_rmse_severity
    

    def validate_model(self, epoch_number):

        with torch.no_grad():
            self.model.eval()

            num_batches = 0
            running_val_loss = 0.0
            running_loss_delineation = 0.
            running_rmse_severity = 0.
            saved = False

            sq_err = np.zeros(self.cfg.severity_convention + 2)
            counters = np.zeros(self.cfg.severity_convention + 2)

            for i, batch in enumerate(self.val_dataloader):

                images = batch["image"].to(self.device)
                labels_severity = batch["gra_mask"].to(self.device)
                
                # -- Tensorboard part 1 --
                index = 0
                setted = False
                for i, _ in enumerate(labels_severity):
                    if labels_severity.shape[0]>2 and len(torch.unique(labels_severity[i]).numpy(force = True))>3 and (255 not in list(torch.unique(labels_severity[i]))):
                        index = i
                        setted = True
                        break

                if not saved and setted:
                    first_image = images[index][1:4]
                    first_image = first_image.flip(dims=(0,)) # RGB ordered

                    image_grid = torchvision.utils.make_grid(first_image, normalize=True, scale_each=True)
                    self.writer.add_image('validation/Image', image_grid, global_step=epoch_number)

                    first_label = labels_severity[index]
                    image_grid = torchvision.utils.make_grid(first_label.unsqueeze(0).float(), normalize=True, scale_each=True)
                    self.writer.add_image('validation/Label', image_grid, global_step=epoch_number)

                logits_severity = self.model(images)

                pred_severity = torch.sigmoid(logits_severity)
                pred_severity_remapped = remap_severity_labels(pred_severity, num_labels = self.cfg.severity_convention)
                labels_severity_remapped = remap_severity_labels(labels_severity, num_labels = self.cfg.severity_convention)

                labels_delineation =  (torch.sigmoid(labels_severity_remapped) >= 1).int()
                pred_delineation = (torch.sigmoid(pred_severity_remapped) >= 1).int()
                loss_delineation = self.loss_criterion(pred_delineation, labels_delineation)

                loss_severity = self.regressor_criterion(pred_severity, labels_severity)

                running_val_loss += (loss_delineation.item() + loss_severity.item())
                running_loss_delineation += loss_delineation.item()
                running_rmse_severity += torch.sqrt(loss_severity).item()

                # Metrics
                sq_err_res_image, count_res_image = compute_squared_errors(pred_severity_remapped, labels_severity_remapped, self.cfg.severity_convention+1)
                sq_err += sq_err_res_image
                counters += count_res_image
                
                self.iou_delineation(pred_delineation, labels_delineation)
                self.f1_delineation(pred_delineation, labels_delineation)

                self.iou_severity(pred_severity_remapped, labels_severity_remapped)
                self.f1_severity(pred_severity_remapped, labels_severity_remapped)

                # -- Tensorboard part 2 --
                if not saved and setted:
                    first_logit = pred_severity[index]
                    first_output = pred_severity_remapped[index]
                    image_grid = torchvision.utils.make_grid(first_output.unsqueeze(0), normalize=True, scale_each=True)
                    self.writer.add_image('validation/Prediction', image_grid, global_step=epoch_number)
                    image_grid = torchvision.utils.make_grid(first_logit.unsqueeze(0), normalize=True, scale_each=True)
                    self.writer.add_image('validation/Logits', image_grid, global_step=epoch_number)
                    saved = True
                
                num_batches += 1

        avg_val_loss = running_val_loss / (num_batches - 1)
        avg_val_loss_delineation = running_loss_delineation / (num_batches - 1)
        avg_val_rmse_severity = running_rmse_severity / (num_batches - 1)
        mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
        print(f" --- MSE: {mse[:self.cfg.severity_convention]} - total: {mse[self.cfg.severity_convention+1]}, RMSE: {np.sqrt(mse[:self.cfg.severity_convention])} - total: {np.sqrt(mse[self.cfg.severity_convention+1])}", end=" ")

        return avg_val_loss, avg_val_loss_delineation, avg_val_rmse_severity 
    

    def test_model(self):

        test_loss = 0
        rmse_loss = 0
        correct_pred = 0
        total = 0
        conf_matr_delineation = 0
        conf_matr_severity = 0
        num_batches = 0
        sq_err = np.zeros(self.cfg.severity_convention + 2)
        counters = np.zeros(self.cfg.severity_convention + 2)

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.test_dataloader):
                
                images = batch["image"].to(self.device)
                labels_severity = batch["gra_mask"].to(self.device)

                logits_severity = self.model(images)

                pred_severity = torch.sigmoid(logits_severity)
                pred_severity_remapped = remap_severity_labels(pred_severity, num_labels = self.cfg.severity_convention)
                labels_severity_remapped = remap_severity_labels(labels_severity, num_labels = self.cfg.severity_convention)

                labels_delineation =  (torch.sigmoid(labels_severity_remapped) >= 1).int()
                pred_delineation = (torch.sigmoid(pred_severity_remapped) >= 1).int()
                loss_delineation = self.loss_criterion(pred_delineation, labels_delineation)

                correct_pred += (pred_delineation == labels_delineation).sum().item()
                total += labels_delineation.numel()

                loss_severity = self.regressor_criterion(pred_severity, labels_severity)

                test_loss += (loss_delineation.item() + loss_severity.item())
                rmse_loss += torch.sqrt(loss_severity).item()
                
                # Metrics
                pred_severity_remapped = remap_severity_labels(pred_severity, num_labels = self.cfg.severity_convention)
                labels_severity_remapped = remap_severity_labels(labels_severity, num_labels = self.cfg.severity_convention)
                sq_err_res_image, count_res_image = compute_squared_errors(pred_severity_remapped, labels_severity_remapped, self.cfg.severity_convention+1)
                sq_err += sq_err_res_image
                counters += count_res_image
                # self.DEBUG_save_image(pred_severity_remapped, labels_severity_remapped)

                self.iou_delineation(pred_delineation, labels_delineation)
                self.f1_delineation(pred_delineation, labels_delineation)

                self.iou_severity(pred_severity_remapped, labels_severity_remapped)
                self.f1_severity(pred_severity_remapped, labels_severity_remapped)

                conf_matr_delineation += compute_confusion_matrix(pred_delineation, labels_delineation, labels_list=[0,1])
                conf_matr_severity += compute_confusion_matrix(pred_severity_remapped, labels_severity_remapped, labels_list=list(range(0, self.cfg.severity_convention + 1)))

                num_batches += 1

        avg_test_loss = test_loss / (num_batches-1)
        avg_rmse_loss = rmse_loss / (num_batches-1)
        test_accuracy_delineation = 100. * correct_pred / total
        mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
        rmse = np.sqrt(mse)

        return avg_test_loss, avg_rmse_loss, test_accuracy_delineation, rmse, conf_matr_delineation, conf_matr_severity
    

    def DEBUG_save_image(self, pred_mask, gt_mask):
        
        int_rnd = np.random.randint(1, 20)
        path = "assets/images/" + str(int_rnd)

        # print(path)
        # print(f"------------------------------------------------------{int_rnd}-------------------------------------")
        # print(f"G_T_ labels MASK: \t {torch.unique(gt_mask[0].int()).numpy(force = True)}")        
        # print(f"PRED labels MASK: \t {torch.unique(pred_mask[0].int()).numpy(force = True)}")

        to_RGB_Mask(pred_mask[0], path + "PRED.png")
        to_RGB_Mask(gt_mask[0], path + "GT.png")









