from pathlib import Path

# from kornia import augmentation as K
import kornia as K
from kornia.augmentation import PatchSequential, ImageSequential
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss, FocalLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
import torchvision
import numpy as np
import os
import csv
import time
from datetime import datetime

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torchmetrics import Dice

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from wildfires.dataset import DatasetCEMS_Delineation
from wildfires.settings import Settings
from wildfires.geoSampler import ConstrainedRandomBatchGeoSampler 
from wildfires.scheduler import MyScheduler
from wildfires.utils import save_log2df, save_model_configuration, set_seed, save_configuration_tb, compute_confusion_matrix, compute_prec_recall_f1_acc, weight_init

from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import stack_samples


class CrossValidator_Delineation:
    
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
            #             data_keys=['input', 'mask'],
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

                                    additional_targets={"delineation": "mask"},
                                )

            # Create datasets and dataloaders
            self.train_dataset = DatasetCEMS_Delineation(root=Path(self.cfg.rootData), annotation_type = self.cfg.annotation_type_delineation,  csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.dataset_path_crossValidation_severity), transforms=self.train_transform, cache=False, fold_test=fold, test_set=False) # transforms=self.train_transform, 
            self.val_dataset = DatasetCEMS_Delineation(root=Path(self.cfg.rootData), annotation_type = self.cfg.annotation_type_delineation, csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.dataset_path_crossValidation_severity), cache=False, fold_test=fold, test_set=True)
            self.test_dataset = DatasetCEMS_Delineation(root=Path(self.cfg.rootData), annotation_type = self.cfg.annotation_type_delineation, csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.dataset_path_crossValidation_severity), cache=False, fold_test=0, test_set=True)
            print(f"\nTraining length: \t{len(self.train_dataset)}")
            print(f"Validation length: \t{len(self.val_dataset)}")
            print(f"Test length: \t\t{len(self.test_dataset)}\n")

            # Create geosampler
            self.train_sampler = ConstrainedRandomBatchGeoSampler(  dataset=self.train_dataset,
                                                                    batch_size=self.args.batch_size,
                                                                    size=self.args.crop_size,
                                                                    length=self.args.sample_per_epoch,
                                                                    burn_prop_batch=self.args.burn_prop_batch ,
                                                                    burn_area_prop=self.args.burn_area_prop)
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
                self.criterion = SoftBCEWithLogitsLoss(pos_weight=torch.tensor(1.5), ignore_index=255)
            elif self.args.loss == "dice":
                self.criterion = DiceLoss(from_logits = True, ignore_index= 255, mode='binary', eps=0.000001)
            elif self.args.loss == "focal":
                self.loss_criterion = FocalLoss(ignore_index= 255, mode='binary', alpha=0.5, gamma=2.0)

            # Scheduler https://github.com/mpyrozhok/adamwr
            self.scheduler = MyScheduler(self.optimizer, self.args.num_epochs, lr_policy = self.args.lr_policy, initial_step = 30, step_size=15, gamma=0.2, stability=50)

            self.iou = BinaryJaccardIndex(ignore_index=255).to(self.device)
            self.f1 = BinaryF1Score(ignore_index=255).to(self.device)

            # Paths
            self.model_name_folder = 'CV_{}_DEL_{}_{}_{}/'.format( self.cfg.dataset_path_crossValidation.split("_")[1].replace("folds.csv","F"), self.args.segmentation_network, self.args.num_epochs * self.args.sample_per_epoch, str(self.args.date.strftime( '%m%d_%H' )))
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

            if not os.path.exists(self.args.checkpoint_folder_path + self.model_name_folder + self.args.model_checkpoint):
                print("\n", "`" * 100, sep="")
                print(f"Wrong configuration: checkpoint {self.args.checkpoint_folder_path + self.model_name_folder + self.args.model_checkpoint} does not exists")
                print("," * 100, "\n", sep="")
                return
        
            print("\n", "*" * 100, sep="")
            print("Loading checkpoint...")

            checkpoint = torch.load(os.path.join(self.args.checkpoint_folder_path, self.model_name_folder, self.args.model_checkpoint))

            # Load the model and optimizer states from the checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_number = checkpoint['epoch'] + 1 # start from next epoch 
            self.tensorboard_path = checkpoint['log_dir']
            self.scheduler = checkpoint['scheduler']

            # self.scheduler.set_epoch(epoch_number)

            print("Done!")
            print("*" * 100, "\n", sep="")

        self.writer = SummaryWriter(self.tensorboard_path)

        # ----- EPOCH CYCLE -----
        for _ in range(epoch_number, self.args.num_epochs+1):
            print(' --- EPOCH {} - Fold {}'.format( epoch_number, fold ), end=" ")

            # ----- TRAIN -----
            start_train_epoch = time.time()

            self.model.train()
            avg_training_loss, proportion_burned_pixels = self.train_one_epoch(epoch_number, self.writer)

            end_train_epoch = time.time()
            print("--- TIME ELAPSED: {:.3f} sec --- Proportion burned area: {:.3f}%".format((end_train_epoch - start_train_epoch), proportion_burned_pixels*100), end = ' ')
            print(f" - Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            # ----- VALIDATION -----
            if epoch_number % self.args.validation_step == 0 and epoch_number != 0:
                
                print(f"Validation: --- epoch: { epoch_number } - fold: {fold}", end = ' ')

                avg_val_loss = self.validate_model(epoch_number) # validation model

                if self.args.lr_policy == "ReduceLROnPlateau":
                    self.scheduler.step(epoch_number, avg_val_loss)

                print(f" - Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}", end = ' ')
                print('--- LOSSES: train {:.6f} --- validation {:.6f}'.format(avg_training_loss, avg_val_loss))  

                # Log the running loss averaged per batch for both training and validation
                self.writer.add_scalar('Loss/Training_loss',  avg_training_loss, epoch_number )
                self.writer.add_scalar('Loss/Validation_loss', avg_val_loss, epoch_number )
                self.writer.add_scalar('optimizer/Learning_rate', self.optimizer.state_dict()["param_groups"][0]["lr"], epoch_number )
                self.writer.add_scalar('metrics/Burned_pixels_epoch', proportion_burned_pixels, epoch_number )
                self.writer.add_scalar('metrics/IOU',  self.iou.compute() , epoch_number )
                self.writer.add_scalar('metrics/F1', self.f1.compute() , epoch_number )
                self.writer.flush()

                self.iou.reset()
                self.f1.reset() 

            # ----- CHECKPOINT SAVING -----
            if epoch_number % self.args.checkpoint_step == 0 and epoch_number != 0:
                print("\n", "*" * 100, sep="")
                print(f"Epoch {epoch_number} - Saving checkpoint ...")
                if not os.path.isdir(self.args.checkpoint_folder_path + self.model_name_folder):
                    os.mkdir(self.args.checkpoint_folder_path + self.model_name_folder)

                checkpoint = {
                    'epoch': epoch_number,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'log_dir': self.tensorboard_path,
                    'scheduler': self.scheduler
                }

                model_name_checkpoint = 'CP{}_F{}_DEL_{}.pth'.format( epoch_number, fold, str(self.args.date.strftime( '%m%d_%H' )) )

                torch.save( checkpoint, os.path.join(self.args.checkpoint_folder_path, self.model_name_folder, model_name_checkpoint) )
                print("Done!")
                print("*" * 100, "\n", sep="")

            if self.args.lr_policy != "ReduceLROnPlateau":
                self.scheduler.step(epoch_number-1)           

            epoch_number+=1

        # ----- TEST -----
        print(f"--- Final TESTING --- epoch: {epoch_number-1} --- date:{datetime.today().strftime('%Y/%m/%d_%H:%M:%S')}")
        avg_test_loss, test_accuracy, conf_matr = self.test_model()
        
        test_f1 = self.f1.compute()
        test_IOU = self.iou.compute()

        prec, recall, f1, acc = compute_prec_recall_f1_acc(conf_matr)

        # save model and its configuration, also convert tensorboard to dataframe and save it as csv
        torch.save(self.model.state_dict(), self.model_save_path)

        tensorboard_csv = self.model_runs_path + self.model_name.replace(".pth", ".csv")
        json_parameters_path = self.model_runs_path + "configuration.json"
        
        # save hyperparameters and configuration in tensorboard
        self.writer = save_configuration_tb(writer=self.writer, args=self.args, test_IoU_delineation=test_IOU, test_f1_delineation=test_f1, conf_matr_delineation=conf_matr)
        save_log2df(self.tensorboard_path, tensorboard_csv)
        save_model_configuration(json_save_path=json_parameters_path, args=self.args, test_IoU_delineation=test_IOU, test_f1_delineation=test_f1, conf_matr_delineation=conf_matr)

        # Print final results
        print('Test Loss: {:.6f} - Test Acc: {:.6f}%'.format(avg_test_loss, test_accuracy))
        print('Test IOU: {:.6f}% - Test F1: {:.6f}%'.format(test_IOU *100, test_f1 *100 ))
        print(f'Confusion matrix: {conf_matr}\n')
        print(f'Test prec: {prec}\nTest recall: {recall}\nTest f1: {f1}\nTest acc: {acc*100}')
        
        self.iou.reset()
        self.f1.reset()

        # Close SummaryWriter
        self.writer.close()


    def train_one_epoch(self):

        running_loss = 0.
        num_batches = 0
        count_pixel_burned = 0
        count_void_images = 0

        burned_pixels_batch = 0
        burned_pixels_total = 0
        pixels_total = 0
        
        for i, batch in enumerate(self.train_dataloader):
            images = batch["image"].to(self.device)
            labels = batch["del_mask"].to(self.device)
 
            # count number pixel burned per batch
            for label_mask in labels:
                count_pixel_burned = ((label_mask > 0) & (label_mask < 120)).sum()
                if count_pixel_burned == 0:
                    count_void_images+=1
                burned_pixels_batch += count_pixel_burned
            
            # print(f"Batch: {i+1}, burnedPixels: {burned_pixels_batch}, percentage: {burned_pixels_batch/labels.numel()}, void images: {count_void_images}, number_pixel_batch: {labels.numel()}")
            burned_pixels_total += burned_pixels_batch
            pixels_total += labels.numel()
            burned_pixels_batch = 0
        
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()                 # Back Propagation
            self.optimizer.step()           # Gardient Descent
            self.optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / (num_batches - 1) # loss per batch

        return avg_train_loss, burned_pixels_total/pixels_total
    

    def validate_model(self, epoch_number):

        with torch.no_grad():
            self.model.eval()

            running_val_loss = 0.0
            saved = False

            for i, batch in enumerate(self.val_dataloader):

                images = batch["image"].to(self.device)
                labels = batch["del_mask"].to(self.device)
                
                index = 0
                setted = False
                if labels.shape[0]>2 and torch.max(labels[0])==1:
                    index = 0
                    setted = True
                if labels.shape[0]>2 and torch.max(labels[1])==1:
                    index = 1
                    setted = True
                if labels.shape[0]>2 and torch.max(labels[2])==1:
                    index = 2
                    setted = True
                
                if not saved and setted:
                    first_image = images[index][1:4]
                    first_image = first_image.flip(dims=(0,)) # RGB ordered

                    image_grid = torchvision.utils.make_grid(first_image, normalize=True, scale_each=True)
                    self.writer.add_image('validation/Image', image_grid, global_step=epoch_number)

                    first_label = labels[index]
                    image_grid = torchvision.utils.make_grid(first_label.unsqueeze(0).float(), normalize=True, scale_each=True)
                    self.writer.add_image('validation/Label', image_grid, global_step=epoch_number)
                
                logits = self.model(images)
                val_loss = self.criterion(logits, labels)
                running_val_loss += val_loss

                self.iou(logits, labels)
                self.f1(logits, labels)

                if not saved and setted:
                    first_logit = logits[index]
                    first_output = torch.sigmoid(first_logit)
                    first_output = (first_output > 0.5).float()
                    image_grid = torchvision.utils.make_grid(first_output.unsqueeze(0), normalize=True, scale_each=True)
                    self.writer.add_image('validation/Prediction', image_grid, global_step=epoch_number)
                    image_grid = torchvision.utils.make_grid(first_logit.unsqueeze(0), normalize=True, scale_each=True)
                    self.writer.add_image('validation/Logits', image_grid, global_step=epoch_number)
                    saved = True
        
        avg_val_loss = running_val_loss / (i + 1)

        return avg_val_loss
        

    def test_model(self):
        
        test_loss = 0
        correct_pred = 0
        total = 0
        num_batches = 0
        conf_matr = 0

        with torch.no_grad():
            self.model.eval()
            for batch in self.test_dataloader:
                
                images = batch["image"].to(self.device)
                labels = batch["del_mask"].to(self.device)

                logits = self.model(images)
                test_loss += self.criterion(logits, labels).item()

                output = torch.sigmoid(logits)
                predicted = (output > 0.5).int()  # apply mask
                correct_pred += (predicted == labels).sum().item()
                total += labels.numel()
                
                num_batches += 1

                # Metrics
                self.iou(logits, labels)
                self.f1(logits, labels)
                conf_matr += compute_confusion_matrix(predicted, labels, labels_list=[0,1])

        avg_test_loss = test_loss / (num_batches + 1)
        test_accuracy = 100. * correct_pred / total

        return avg_test_loss, test_accuracy, conf_matr

