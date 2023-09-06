from pathlib import Path

# from kornia import augmentation as K
import onnxruntime # pip install onnxruntime-gpu
import onnx
import segmentation_models_pytorch as smp
import torch
import torchvision
import numpy as np
import pandas as pd
import os
import glob
import json
import csv
import time
import shutil
import rasterio

import argparse
from datetime import datetime

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, MulticlassJaccardIndex, MulticlassF1Score

from wildfires.settings import Settings
from wildfires.smooth_tiling import predict_smooth_windowing

from wildfires.losses import My_MSELoss
from wildfires.model_multitask import Unet_Segmentation_multitask, UnetPlusPlus_Segmentation_multitask, DeepLabV3_Segmentation_multitask, DeepLabV3Plus_Segmentation_multitask, PSPNet_Segmentation_multitask

from wildfires.utils import compute_confusion_matrix, remap_severity_labels, to_RGB_Mask, compute_squared_errors

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


###########################################################################################################################################################################################
#
#                                                                        MULTITASK
#
###########################################################################################################################################################################################

class Test_Multitask:

    def __init__(self, args):

        self.args = args
        self.line_csv = ""

        self.cfg = Settings()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # General metrics on whole dataset selected
        self.iou_delineation = BinaryJaccardIndex(ignore_index=255) #.to(self.device)
        self.f1_delineation = BinaryF1Score(ignore_index=255) #.to(self.device)
        self.iou_severity = MulticlassJaccardIndex(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)
        self.f1_severity = MulticlassF1Score(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)

        # Metrics on single image
        self.local_F1_delineation = BinaryJaccardIndex(ignore_index=255) #.to(self.device)
        self.local_IOU_delineation = BinaryF1Score(ignore_index=255) #.to(self.device)
        self.local_IOU_severity = MulticlassJaccardIndex(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)
        self.local_F1_severity = MulticlassF1Score(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)
        
        if self.args.test_fold != None:
            self.cfg.satelliteData_path = "cross_validation/" + self.cfg.dataset_path_crossValidation

        if self.args.segmentation_network != None:
            self.segmentation_network = self.args.segmentation_network
        else:
            if "unet" in self.args.model_path.split("_"):
                self.segmentation_network = "unet"
            elif "unetplus" in self.args.model_path.split("_"):
                self.segmentation_network = "unetplus"
            elif "unetAtt" in self.args.model_path.split("_"):
                self.segmentation_network = "unetAtt"
            elif "unetplusAtt" in self.args.model_path.split("_"):
                self.segmentation_network = "unetplusAtt"
            elif "deeplab" in self.args.model_path.split("_"):
                self.segmentation_network = "deeplab"
            elif "deeplabplus" in self.args.model_path.split("_"):
                self.segmentation_network = "deeplabplus"
            elif "pspnet" in self.args.model_path.split("_"):
                self.segmentation_network = "pspnet"
            elif "onnx" in self.args.model_path:
                self.args.onnx = True
            else:
                print("Checkout models name in the model's path")
    
    def get_list_path_images(self):
        
        self.filename_glob = "EMSR*AOI*S2L2A.tiff"
        pathname = os.path.join(Path(self.cfg.testData), "**", self.filename_glob)
        csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.satelliteData_path)

        df = pd.read_csv(csv_satelliteData, index_col=False)
        df = df[ (df["GRA"] == 1) & (df["folder"] == "optimal")]
        
        # ONLY for Cross Validation (LOO cross validation)
        if self.args.test_fold != None:
            df = df.loc[ df["fold"] == int(self.args.test_fold) ] # test/val fold (Leave one out cross validation)

        listPaths = df["folderPath"].str.split("/").str[-1]
        filepaths = []
        for idx in listPaths:
            for f in glob.iglob(pathname, recursive=True):
                if idx in f:
                    filepaths.append(f)

        assert len(filepaths) > 0, f"No images found in {Path(self.cfg.testData)}"

        return filepaths


    def load_model(self):

        # Load model that are in format .pth
        if self.args.onnx is None or self.args.onnx == False:

            if self.segmentation_network == 'unet':
                self.model = Unet_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.segmentation_network == 'unetplus':
                self.model = UnetPlusPlus_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.segmentation_network == 'unetAtt':
                self.model = Unet_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None, decoder_attention_type="scse")
            elif self.segmentation_network == 'unetplusAtt':
                self.model = UnetPlusPlus_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None, decoder_attention_type="scse")
            elif self.segmentation_network == 'deeplab':
                self.model = DeepLabV3_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.segmentation_network == 'deeplabplus':
                self.model = DeepLabV3Plus_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.segmentation_network == 'pspnet':
                self.model = PSPNet_Segmentation_multitask(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            else:
                print("Wrong segmentation_network name")
                return
            
            pre_load = self.model.segmentation_head[0].weight
        
            # Load pre-trained model or checkpoint
            if self.args.model_checkpoint_path is not None:
                checkpoint = torch.load(self.args.model_checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                saved_state_dict = torch.load(self.args.model_path)
                self.model.load_state_dict(saved_state_dict)
                
            self.model.eval()

            post_load = self.model.segmentation_head[0].weight

            # Check correcteness of loading model's weights
            if torch.equal(pre_load, post_load):
                print(" ----  Model upload completed ---- ")        
            
            # self.model.to(self.device)

        # Load model in onnx format
        elif self.args.onnx:

            self.model = onnxruntime.InferenceSession(self.args.model_path, providers=onnxruntime.get_available_providers())
            self.input_name = [x.name for x in self.model.get_inputs()]
            print(f"Input name ONNX MODEL: {self.input_name}")

    def preprocess_severity_mask(self, labels_severity):

        # REMAPPING severity levels (Not used or tested)
        if self.args.severity_convention == 3:
            unique_severity_levels = torch.unique(labels_severity)
            unique_severity_levels.sort()
            # unique_severity_levels = unique_severity_levels[1:] # remove zero value

            if  unique_severity_levels.tolist() == [0,1,2,3,4] or \
                unique_severity_levels.tolist() == [0,2,3,4] or \
                unique_severity_levels.tolist() == [0,1,3,4] or \
                unique_severity_levels.tolist() == [0,1,3] or \
                unique_severity_levels.tolist() == [0,1,4] or \
                unique_severity_levels.tolist() == [0,2,3] or \
                unique_severity_levels.tolist() == [0,2,4] or \
                unique_severity_levels.tolist() == [0,3,4] or \
                unique_severity_levels.tolist() == [0,1]:

                grading_lut = torch.tensor([0,1,1,2,3])
                labels_severity = grading_lut[torch.floor(labels_severity).long()]
                labels_severity = labels_severity.float()
            
            elif unique_severity_levels.tolist() == [0,1,2,4] or \
                 unique_severity_levels.tolist() == [0,1,2,3] or \
                 unique_severity_levels.tolist() == [0,1,2]:
                
                grading_lut = torch.tensor([0,1,2,3,3])
                labels_severity = grading_lut[torch.floor(labels_severity).long()]
                labels_severity = labels_severity.float()

        # NORMALIZATION of severity levels
        min_value = 0
        max_value = self.args.severity_convention
        labels_severity = (labels_severity - min_value) / (max_value - min_value)
        labels_severity[labels_severity > 15] = 255 # restore no data after normalization
        
        return labels_severity
    

    def testing(self):
        
        # load dataset and model
        self.test_dataset = self.get_list_path_images()
        self.load_model()

        if self.args.model_checkpoint_path is not None:
            self.args.saving_folder_path = self.args.saving_folder_path + self.args.model_checkpoint_path.split("/")[1].replace(".pth", "")
        else:
            self.args.saving_folder_path = self.args.saving_folder_path + self.args.model_path.split("/")[1]
        print(f"Saving folder path: {self.args.saving_folder_path}")
        
        # Create .csv file for statistics on every image
        csv_result_path = self.args.saving_folder_path + "/result_image.csv"
        os.makedirs(os.path.dirname(csv_result_path), exist_ok=True)

        columns_names = ["EMSR_AOI","height","width","numPixel","pixelBurned","predBurnedPixel","F1_DEL","IOU_DEL",\
                         "F1_GRA_0","F1_GRA_1","F1_GRA_2","F1_GRA_3","F1_GRA_4","F1_GRA_Mean","IOU_GRA_0","IOU_GRA_1","IOU_GRA_2","IOU_GRA_3","IOU_GRA_4","IOU_GRA_Mean",\
                         "RMSE_0","RMSE_1","RMSE_2","RMSE_3","RMSE_4","RMSE_Mean","RMSE_Total"
                        ]
        results_df = pd.DataFrame(columns=columns_names)
        index = 0

        sq_err = np.zeros(self.args.severity_convention + 2)
        counters = np.zeros(self.args.severity_convention + 2)
        
        # cycle over each image of the dataset
        for image_path in self.test_dataset:
            
            print(image_path)
            image_mask_DEL_path = image_path.replace("_S2L2A.tiff", "_DEL.tif")
            image_mask_GRA_path = image_path.replace("_S2L2A.tiff", "_GRA.tif")
            image_cloud_mask_path = image_path.replace("_S2L2A.tiff", "_CM.tif")

            # read and open the images
            with rasterio.open(image_path) as image, rasterio.open(image_mask_DEL_path) as label_delineation, rasterio.open(image_mask_GRA_path) as label_severity, rasterio.open(image_cloud_mask_path) as cloud_mask:
                
                image_array = image.read()  # read the image as a numpy array
                label_array_delineation = label_delineation.read().squeeze()
                label_array_severity = label_severity.read().squeeze()

                label_array_severity = self.preprocess_severity_mask(label_array_severity)

                cloud_mask_array = cloud_mask.read().squeeze()

                label_array_delineation[(cloud_mask_array == 1) & (cloud_mask_array == 3)] = 255
                label_array_severity[(cloud_mask_array == 1) & (cloud_mask_array == 3)] = 255

                image_tensor = torch.from_numpy(image_array) #.float().to(self.device)
                label_tensor_delineation = torch.from_numpy(label_array_delineation)
                label_tensor_severity = torch.from_numpy(label_array_severity)                

                # without SMOOTHING 
                if self.args.smoothing == False:
                    if self.segmentation_network == 'unet':
                        pad_divisible = 32
                    elif self.segmentation_network == 'deeplab':
                        pad_divisible = 8
                    elif self.segmentation_network== 'deeplabplus':
                        pad_divisible = 16
                    elif self.segmentation_network == 'unetplus':
                        pad_divisible = 32
                    elif self.segmentation_network == 'pspnet':
                        pad_divisible = 8

                    new_height = (image_tensor.shape[1] // pad_divisible) * pad_divisible
                    new_width = (image_tensor.shape[2] // pad_divisible) * pad_divisible

                    image_tensor = image_tensor[:, :new_height, :new_width].unsqueeze(0)
                    label_tensor_severity = label_tensor_severity[:new_height, :new_width]
                    label_tensor_delineation = label_tensor_delineation[:new_height, :new_width]

                    logits_delineation, logits_severity = self.model(image_tensor)
                    logits_delineation = logits_delineation.squeeze()
                    logits_severity = logits_severity.squeeze()

                    pred_delineation_tensor = (torch.sigmoid(logits_delineation) > 0.5).int()
                    pred_severity_tensor = torch.sigmoid(logits_severity)

                    pred_severity_remapped = remap_severity_labels(pred_severity_tensor, num_labels = self.args.severity_convention)
                    labels_severity_remapped = remap_severity_labels(label_tensor_severity, num_labels = self.args.severity_convention)

                # with SMOOTHING
                else:
                    if self.args.onnx:

                        logits_delineation, logits_severity = predict_smooth_windowing( mode = "multitask_onnx",
                                                                                        image=image_tensor,
                                                                                        tile_size =  480,
                                                                                        subdivisions = 2,  # Minimal amount of overlap for windowing. Must be an even number.
                                                                                        batch_size = self.args.batch_size,
                                                                                        channels_first = True,
                                                                                        prediction_fn = (
                                                                                            lambda image: self.model.run(None, input_feed=dict(zip(self.input_name,image))))
                                                                                        )
                        pred_delineation_tensor = (torch.sigmoid(logits_delineation) > 0.5).int()
                        # pred_severity_tensor = torch.sigmoid(logits_severity)
                        # pred_delineation_tensor = logits_delineation
                        pred_severity_tensor = logits_severity # sigmoid is not necessary, mask is clipped with np.clip()
                        

                        pred_severity_remapped = pred_severity_tensor
                        # pred_severity_remapped = remap_severity_labels(pred_severity_tensor, num_labels = self.args.severity_convention)
                        labels_severity_remapped = remap_severity_labels(label_tensor_severity, num_labels = self.args.severity_convention)


                    else:
                        logits_delineation, logits_severity = predict_smooth_windowing( mode = "multitask",
                                                                                        image=image_tensor,
                                                                                        tile_size =  self.args.crop_size,
                                                                                        subdivisions = 2,  # Minimal amount of overlap for windowing. Must be an even number.
                                                                                        batch_size = self.args.batch_size,
                                                                                        channels_first = True,
                                                                                        prediction_fn = (
                                                                                            lambda image: self.model(image))
                                                                                        )
                
                        pred_delineation_tensor = (torch.sigmoid(logits_delineation) > 0.5).int()
                        pred_severity_tensor = torch.sigmoid(logits_severity)

                        pred_severity_remapped = remap_severity_labels(pred_severity_tensor, num_labels = self.args.severity_convention)
                        labels_severity_remapped = remap_severity_labels(label_tensor_severity, num_labels = self.args.severity_convention)

                # Compute metrics over the image
                sq_err_res_image, count_res_image = compute_squared_errors(pred_severity_remapped, labels_severity_remapped, self.args.severity_convention+1)
                sq_err += sq_err_res_image
                counters += count_res_image
                mse_image = np.true_divide(sq_err_res_image, count_res_image, np.full(sq_err_res_image.shape, np.nan), where=count_res_image != 0)
                rmse_image = np.sqrt(mse_image)

                self.iou_delineation(pred_delineation_tensor, label_tensor_delineation)
                self.f1_delineation(pred_delineation_tensor, label_tensor_delineation)
                self.local_F1_delineation(pred_delineation_tensor, label_tensor_delineation)
                self.local_IOU_delineation(pred_delineation_tensor, label_tensor_delineation)

                self.iou_severity(pred_severity_remapped, labels_severity_remapped)
                self.f1_severity(pred_severity_remapped, labels_severity_remapped)
                self.local_F1_severity(pred_severity_remapped, labels_severity_remapped)
                self.local_IOU_severity(pred_severity_remapped, labels_severity_remapped)                

                # tiff format images
                save_path_image = image_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_mask_delineation = image_mask_DEL_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_mask_grading = image_mask_GRA_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_predicted_grading_mask = image_mask_GRA_path.replace(self.cfg.testData, self.args.saving_folder_path).replace("_GRA.tif", "_PRED_GRA.tif")
                save_path_predicted_delineation_mask = image_mask_DEL_path.replace(self.cfg.testData, self.args.saving_folder_path).replace("_DEL.tif", "_PRED_DEL.tif")
                os.makedirs(os.path.dirname(save_path_image), exist_ok=True)
               
                # print(save_path_image)
                # print(save_path_mask_grading)
                # print(save_path_predicted_grading_mask)

                if self.args.save_tif != None:
                    shutil.copyfile(image_path, save_path_image)
                    shutil.copyfile(image_mask_DEL_path, save_path_mask_delineation)
                    shutil.copyfile(image_mask_GRA_path, save_path_mask_grading)

                    matedata_mask = label_severity.meta
                    with rasterio.open(save_path_predicted_grading_mask, 'w', **matedata_mask) as dst:
                        dst.write(np.expand_dims(pred_severity_remapped.numpy(), axis=0))
                    with rasterio.open(save_path_predicted_delineation_mask, 'w', **matedata_mask) as dst:
                        dst.write(np.expand_dims(pred_delineation_tensor.numpy(), axis=0))

                # RGB format images
                save_path_image_RGB = save_path_image.replace(".tiff", ".png")
                save_path_mask_delineation_RGB = save_path_mask_delineation.replace(".tif", ".png")
                save_path_mask_grading_RGB = save_path_mask_grading.replace(".tif", ".png")
                save_path_predicted_mask_grading_RGB = save_path_predicted_grading_mask.replace(".tif", ".png")
                save_path_predicted_mask_delineation_RGB = save_path_predicted_delineation_mask.replace(".tif", ".png")

                # print(save_path_image_RGB)
                # print(save_path_mask_grading_RGB)
                # print(save_path_predicted_mask_grading_RGB)

                if self.args.save_RGB != None:
                    shutil.copyfile(image_path.replace(".tiff", ".png"), save_path_image_RGB)
                    shutil.copyfile(image_mask_DEL_path.replace(".tif", ".png"), save_path_mask_delineation_RGB)
                    shutil.copyfile(image_mask_GRA_path.replace(".tif", ".png"), save_path_mask_grading_RGB)

                    to_RGB_Mask(pred_delineation_tensor*3, save_path_predicted_mask_delineation_RGB)
                    to_RGB_Mask(pred_severity_remapped, save_path_predicted_mask_grading_RGB)

                image_F1_delineation = self.local_F1_delineation.compute()
                image_IOU_delineation = self.local_IOU_delineation.compute()
                image_F1_severity = self.local_F1_severity.compute()
                image_IOU_severity = self.local_IOU_severity.compute()

                # save statistics for each image in a .csv file
                emsr_aoi = [sub_folder for sub_folder in image_path.split("/") if "EMSR" in sub_folder][1]
                pixelBurned = np.count_nonzero( (label_array_severity > 0) & (label_array_severity < 120))
                predPixelBurned = np.count_nonzero(pred_severity_remapped.numpy() > 0)
                # conf_matr = compute_confusion_matrix(pred_severity_remapped, labels_severity_remapped, labels_list=list(range(0, self.args.severity_convention + 1)))

                print(f"Severity image : {image_F1_severity.numpy()}")

                results_df.loc[index] = [emsr_aoi, image_array.shape[1], image_array.shape[2], image_array.shape[1]*image_array.shape[2], pixelBurned, predPixelBurned, image_F1_delineation.numpy(), image_IOU_delineation.numpy(),\
                                         image_F1_severity.numpy()[0],image_F1_severity.numpy()[1],image_F1_severity.numpy()[2],image_F1_severity.numpy()[3],image_F1_severity.numpy()[4],np.mean(image_F1_severity.numpy()), \
                                         image_IOU_severity.numpy()[0],image_IOU_severity.numpy()[1],image_IOU_severity.numpy()[2],image_IOU_severity.numpy()[3],image_IOU_severity.numpy()[4],np.mean(image_IOU_severity.numpy()), \
                                         rmse_image[0],rmse_image[1],rmse_image[2],rmse_image[3],rmse_image[4],np.mean(rmse_image[:5]),rmse_image[5]
                                        ]

                self.local_F1_delineation.reset()
                self.local_IOU_delineation.reset() 
                self.local_F1_severity.reset()
                self.local_IOU_severity.reset()

                index += 1

        test_f1_delineation = self.f1_delineation.compute()
        test_IOU_delineation = self.iou_delineation.compute()
        test_f1_severity = self.f1_severity.compute()
        test_IOU_severity = self.iou_severity.compute()
        mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
        rmse = np.sqrt(mse)

        dict_results = {"Model_tested": self.args.model_path.split("/")[1],
                        "F1_delineation": test_f1_delineation.numpy().tolist(),
                        "IOU_delineation": test_IOU_delineation.numpy().tolist(),
                        "F1_severity": test_f1_severity.numpy().tolist(),
                        "F1_severity_mean": str(np.mean(test_f1_severity.numpy())),
                        "IOU_severity": test_IOU_severity.numpy().tolist(),
                        "IOU_severity_mean": str(np.mean(test_IOU_severity.numpy())),
                        "RMSE_severity":rmse.tolist()[:5],
                        "RMSE_severity_mean": str(np.mean(rmse[:5])),
                        "RMSE_severity_Total": str(np.mean(rmse[5]))
        }

        json_result_path = self.args.saving_folder_path + "/results.json"
        with open(json_result_path, 'w') as f:
            json.dump(dict_results, f)
        
        satellite_data_df_sorted = results_df.sort_values(['EMSR_AOI'], ascending = True)
        satellite_data_df_sorted.to_csv(csv_result_path, index=False)

        print(f"F1 delineation: {test_f1_delineation.numpy()}")
        print(f"IOU delineation: {test_IOU_delineation.numpy()}")
        print(f"F1 severity: {test_f1_severity.numpy()} - mean: {np.mean(test_f1_severity.numpy())}")
        print(f"IOU severity: {test_IOU_severity.numpy()} - mean: {np.mean(test_IOU_severity.numpy())}")
        print(f"RMSE severity: {rmse} - mean: {np.mean(rmse[:5])} - total: {rmse[5]}")

        self.iou_delineation.reset()
        self.f1_delineation.reset()
        self.iou_severity.reset()
        self.f1_severity.reset()


###########################################################################################################################################################################################
#
#                                                                        SEVERITY
#
###########################################################################################################################################################################################


class Test_Severity:

    def __init__(self, args):

        self.args = args
        self.line_csv = ""

        self.cfg = Settings()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # General metrics on whole dataset selected
        self.iou_delineation = BinaryJaccardIndex(ignore_index=255) #.to(self.device)
        self.f1_delineation = BinaryF1Score(ignore_index=255) #.to(self.device)
        self.iou_severity = MulticlassJaccardIndex(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)
        self.f1_severity = MulticlassF1Score(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)

        # Metrics on single image
        self.local_F1_delineation = BinaryJaccardIndex(ignore_index=255) #.to(self.device)
        self.local_IOU_delineation = BinaryF1Score(ignore_index=255) #.to(self.device)
        self.local_IOU_severity = MulticlassJaccardIndex(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)
        self.local_F1_severity = MulticlassF1Score(num_classes=self.args.severity_convention+1, ignore_index=255, average = 'none')#.to(self.device)
      
        if self.args.test_fold != None:
            self.cfg.satelliteData_path = "cross_validation/" + self.cfg.dataset_path_crossValidation

        if self.args.segmentation_network != None:
            self.segmentation_network = self.args.segmentation_network
        else:
            if "unet" in self.args.model_path.split("_"):
                self.segmentation_network = "unet"
            elif "unetplus" in self.args.model_path.split("_"):
                self.segmentation_network = "unetplus"
            elif "deeplab" in self.args.model_path.split("_"):
                self.segmentation_network = "deeplab"
            elif "deeplabplus" in self.args.model_path.split("_"):
                self.segmentation_network = "deeplabplus"
            elif "pspnet" in self.args.model_path.split("_"):
                self.segmentation_network = "pspnet"
            elif self.args.onnx != None:
                pass
            else:
                print("Checkout models name in the model's path")
    
    def get_list_path_images(self):
        
        self.filename_glob = "EMSR*AOI*S2L2A.tiff"
        pathname = os.path.join(Path(self.cfg.testData), "**", self.filename_glob)
        csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.satelliteData_path)

        df = pd.read_csv(csv_satelliteData, index_col=False)
        df = df[ (df["GRA"] == 1) & (df["folder"] == "optimal")]
        
        # ONLY for Cross Validation (LOO cross validation)
        if self.args.test_fold != None:
            df = df.loc[ df["fold"] == int(self.args.test_fold) ] # test/val fold (Leave one out cross validation)

        listPaths = df["folderPath"].str.split("/").str[-1]
        filepaths = []
        for idx in listPaths:
            for f in glob.iglob(pathname, recursive=True):
                if idx in f:
                    filepaths.append(f)

        assert len(filepaths) > 0, f"No images found in {Path(self.cfg.testData)}"

        return filepaths


    def load_model(self):

        # Load model that are in format .pth
        if self.args.onnx is None or self.args.onnx == False:

            if self.segmentation_network == 'unet':
                self.model = smp.Unet(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None) #, decoder_attention_type="scse")
            elif self.segmentation_network == 'unetplus':
                self.model = smp.UnetPlusPlus(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None) #, decoder_attention_type="scse")
            elif self.segmentation_network == 'deeplab':
                self.model = smp.DeepLabV3(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.segmentation_network == 'deeplabplus':
                self.model = smp.DeepLabV3Plus(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            elif self.segmentation_network == 'pspnet':
                self.model = smp.PSPNet(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
            else:
                print("Wrong segmentation_network name")
                return
            
            pre_load = self.model.segmentation_head[0].weight
        
            # Load pre-trained model or checkpoint
            if self.args.model_checkpoint_path is not None:
                checkpoint = torch.load(self.args.model_checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                saved_state_dict = torch.load(self.args.model_path)
                self.model.load_state_dict(saved_state_dict)
                
            self.model.eval()

            post_load = self.model.segmentation_head[0].weight

            # Check correcteness of loading model's weights
            if torch.equal(pre_load, post_load):
                print(" ----  Model upload completed ---- ")        
            
            # self.model.to(self.device)

        # Load model in onnx format
        elif self.args.onnx:

            self.model = onnxruntime.InferenceSession(self.args.model_path, providers=onnxruntime.get_available_providers())
            self.input_name = [x.name for x in self.model.get_inputs()]
            print(f"Input name ONNX MODEL: {self.input_name}")

    def preprocess_severity_mask(self, labels_severity):

        # REMAPPING severity levels (Not used or tested)
        if self.args.severity_convention == 3:
            unique_severity_levels = torch.unique(labels_severity)
            unique_severity_levels.sort()
            # unique_severity_levels = unique_severity_levels[1:] # remove zero value

            if  unique_severity_levels.tolist() == [0,1,2,3,4] or \
                unique_severity_levels.tolist() == [0,2,3,4] or \
                unique_severity_levels.tolist() == [0,1,3,4] or \
                unique_severity_levels.tolist() == [0,1,3] or \
                unique_severity_levels.tolist() == [0,1,4] or \
                unique_severity_levels.tolist() == [0,2,3] or \
                unique_severity_levels.tolist() == [0,2,4] or \
                unique_severity_levels.tolist() == [0,3,4] or \
                unique_severity_levels.tolist() == [0,1]:

                grading_lut = torch.tensor([0,1,1,2,3])
                labels_severity = grading_lut[torch.floor(labels_severity).long()]
                labels_severity = labels_severity.float()
            
            elif unique_severity_levels.tolist() == [0,1,2,4] or \
                 unique_severity_levels.tolist() == [0,1,2,3] or \
                 unique_severity_levels.tolist() == [0,1,2]:
                
                grading_lut = torch.tensor([0,1,2,3,3])
                labels_severity = grading_lut[torch.floor(labels_severity).long()]
                labels_severity = labels_severity.float()

        # NORMALIZATION of severity levels
        min_value = 0
        max_value = self.args.severity_convention
        labels_severity = (labels_severity - min_value) / (max_value - min_value)
        labels_severity[labels_severity > 15] = 255 # restore no data after normalization
        
        return labels_severity
    

    def testing(self):
        
        # load dataset and model
        self.test_dataset = self.get_list_path_images()
        self.load_model()

        if self.args.model_checkpoint_path is not None:
            self.args.saving_folder_path = self.args.saving_folder_path + self.args.model_checkpoint_path.split("/")[1].replace(".pth", "")
        else:
            self.args.saving_folder_path = self.args.saving_folder_path + self.args.model_path.split("/")[1]
        print(f"Saving folder path: {self.args.saving_folder_path}")
        
        # Create .csv file for statistics on every image
        csv_result_path = self.args.saving_folder_path + "/result_image.csv"
        os.makedirs(os.path.dirname(csv_result_path), exist_ok=True)

        columns_names = ["EMSR_AOI","height","width","numPixel","pixelBurned","predBurnedPixel","F1_DEL","IOU_DEL",\
                         "F1_GRA_0","F1_GRA_1","F1_GRA_2","F1_GRA_3","F1_GRA_4","F1_GRA_Mean","IOU_GRA_0","IOU_GRA_1","IOU_GRA_2","IOU_GRA_3","IOU_GRA_4","IOU_GRA_Mean",\
                         "RMSE_0","RMSE_1","RMSE_2","RMSE_3","RMSE_4","RMSE_Mean","RMSE_Total"
                        ]
        results_df = pd.DataFrame(columns=columns_names)
        index = 0

        sq_err = np.zeros(self.args.severity_convention + 2)
        counters = np.zeros(self.args.severity_convention + 2)
        
        # cycle over each image of the dataset
        for image_path in self.test_dataset:
            
            print(image_path)
            image_mask_DEL_path = image_path.replace("_S2L2A.tiff", "_DEL.tif")
            image_mask_GRA_path = image_path.replace("_S2L2A.tiff", "_GRA.tif")
            image_cloud_mask_path = image_path.replace("_S2L2A.tiff", "_CM.tif")

            # read and open the images
            with rasterio.open(image_path) as image, rasterio.open(image_mask_DEL_path) as label_delineation, rasterio.open(image_mask_GRA_path) as label_severity, rasterio.open(image_cloud_mask_path) as cloud_mask:
                
                image_array = image.read()  # read the image as a numpy array
                label_array_delineation = label_delineation.read().squeeze()
                label_array_severity = label_severity.read().squeeze()

                label_array_severity = self.preprocess_severity_mask(label_array_severity)

                cloud_mask_array = cloud_mask.read().squeeze()

                label_array_delineation[(cloud_mask_array == 1) & (cloud_mask_array == 3)] = 255
                label_array_severity[(cloud_mask_array == 1) & (cloud_mask_array == 3)] = 255

                image_tensor = torch.from_numpy(image_array) #.float().to(self.device)
                label_tensor_delineation = torch.from_numpy(label_array_delineation)
                label_tensor_severity = torch.from_numpy(label_array_severity)                

                # without SMOOTHING 
                if self.args.smoothing == False:
                    if self.segmentation_network == 'unet':
                        pad_divisible = 32
                    elif self.segmentation_network == 'deeplab':
                        pad_divisible = 8
                    elif self.segmentation_network== 'deeplabplus':
                        pad_divisible = 16
                    elif self.segmentation_network == 'unetplus':
                        pad_divisible = 32
                    elif self.segmentation_network == 'pspnet':
                        pad_divisible = 8

                    new_height = (image_tensor.shape[1] // pad_divisible) * pad_divisible
                    new_width = (image_tensor.shape[2] // pad_divisible) * pad_divisible

                    image_tensor = image_tensor[:, :new_height, :new_width].unsqueeze(0)
                    label_tensor_severity = label_tensor_severity[:new_height, :new_width]
                    label_tensor_delineation = label_tensor_delineation[:new_height, :new_width]

                    logits_delineation, logits_severity = self.model(image_tensor)
                    logits_delineation = logits_delineation.squeeze()
                    logits_severity = logits_severity.squeeze()

                    pred_delineation_tensor = (torch.sigmoid(logits_delineation) > 0.5).int()
                    pred_severity_tensor = torch.sigmoid(logits_severity)

                    pred_severity_remapped = remap_severity_labels(pred_severity_tensor, num_labels = self.args.severity_convention)
                    labels_severity_remapped = remap_severity_labels(label_tensor_severity, num_labels = self.args.severity_convention)

                # with SMOOTHING
                else:
                    logits_severity = predict_smooth_windowing( mode = "GRA",
                                                                                    image=image_tensor,
                                                                                    tile_size =  self.args.crop_size,
                                                                                    subdivisions = 2,  # Minimal amount of overlap for windowing. Must be an even number.
                                                                                    batch_size = self.args.batch_size,
                                                                                    channels_first = True,
                                                                                    prediction_fn = (
                                                                                        lambda image: self.model(image))
                                                                                    )
            
                    pred_delineation_tensor = (torch.sigmoid(logits_severity) >= 0.2).int()
                    pred_severity_tensor = torch.sigmoid(logits_severity)

                    pred_severity_remapped = remap_severity_labels(pred_severity_tensor, num_labels = self.args.severity_convention)
                    labels_severity_remapped = remap_severity_labels(label_tensor_severity, num_labels = self.args.severity_convention)



                # Compute metrics over the image
                sq_err_res_image, count_res_image = compute_squared_errors(pred_severity_remapped, labels_severity_remapped, self.args.severity_convention+1)
                sq_err += sq_err_res_image
                counters += count_res_image
                mse_image = np.true_divide(sq_err_res_image, count_res_image, np.full(sq_err_res_image.shape, np.nan), where=count_res_image != 0)
                rmse_image = np.sqrt(mse_image)

                self.iou_delineation(pred_delineation_tensor, label_tensor_delineation)
                self.f1_delineation(pred_delineation_tensor, label_tensor_delineation)
                self.local_F1_delineation(pred_delineation_tensor, label_tensor_delineation)
                self.local_IOU_delineation(pred_delineation_tensor, label_tensor_delineation)

                self.iou_severity(pred_severity_remapped, labels_severity_remapped)
                self.f1_severity(pred_severity_remapped, labels_severity_remapped)
                self.local_F1_severity(pred_severity_remapped, labels_severity_remapped)
                self.local_IOU_severity(pred_severity_remapped, labels_severity_remapped)                

                # tiff format images
                save_path_image = image_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_mask_delineation = image_mask_DEL_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_mask_grading = image_mask_GRA_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_predicted_grading_mask = image_mask_GRA_path.replace(self.cfg.testData, self.args.saving_folder_path).replace("_GRA.tif", "_PRED_GRA.tif")
                save_path_predicted_delineation_mask = image_mask_DEL_path.replace(self.cfg.testData, self.args.saving_folder_path).replace("_DEL.tif", "_PRED_DEL.tif")
                os.makedirs(os.path.dirname(save_path_image), exist_ok=True)
               
                # print(save_path_image)
                # print(save_path_mask_grading)
                # print(save_path_predicted_grading_mask)

                if self.args.save_tif != None:
                    shutil.copyfile(image_path, save_path_image)
                    shutil.copyfile(image_mask_DEL_path, save_path_mask_delineation)
                    shutil.copyfile(image_mask_GRA_path, save_path_mask_grading)

                    matedata_mask = label_severity.meta
                    with rasterio.open(save_path_predicted_grading_mask, 'w', **matedata_mask) as dst:
                        dst.write(np.expand_dims(pred_severity_remapped.numpy(), axis=0))
                    with rasterio.open(save_path_predicted_delineation_mask, 'w', **matedata_mask) as dst:
                        dst.write(np.expand_dims(pred_delineation_tensor.numpy(), axis=0))

                # RGB format images
                save_path_image_RGB = save_path_image.replace(".tiff", ".png")
                save_path_mask_delineation_RGB = save_path_mask_delineation.replace(".tif", ".png")
                save_path_mask_grading_RGB = save_path_mask_grading.replace(".tif", ".png")
                save_path_predicted_mask_grading_RGB = save_path_predicted_grading_mask.replace(".tif", ".png")
                save_path_predicted_mask_delineation_RGB = save_path_predicted_delineation_mask.replace(".tif", ".png")

                # print(save_path_image_RGB)
                # print(save_path_mask_grading_RGB)
                # print(save_path_predicted_mask_grading_RGB)

                if self.args.save_RGB != None:
                    shutil.copyfile(image_path.replace(".tiff", ".png"), save_path_image_RGB)
                    shutil.copyfile(image_mask_DEL_path.replace(".tif", ".png"), save_path_mask_delineation_RGB)
                    shutil.copyfile(image_mask_GRA_path.replace(".tif", ".png"), save_path_mask_grading_RGB)

                    to_RGB_Mask(pred_delineation_tensor*3, save_path_predicted_mask_delineation_RGB)
                    to_RGB_Mask(pred_severity_remapped, save_path_predicted_mask_grading_RGB)

                image_F1_delineation = self.local_F1_delineation.compute()
                image_IOU_delineation = self.local_IOU_delineation.compute()
                image_F1_severity = self.local_F1_severity.compute()
                image_IOU_severity = self.local_IOU_severity.compute()

                # save statistics for each image in a .csv file
                emsr_aoi = [sub_folder for sub_folder in image_path.split("/") if "EMSR" in sub_folder][1]
                pixelBurned = np.count_nonzero( (label_array_severity > 0) & (label_array_severity < 120))
                predPixelBurned = np.count_nonzero(pred_severity_remapped.numpy() > 0)
                # conf_matr = compute_confusion_matrix(pred_severity_remapped, labels_severity_remapped, labels_list=list(range(0, self.args.severity_convention + 1)))

                print(f"Severity image : {image_F1_severity.numpy()}")

                results_df.loc[index] = [emsr_aoi, image_array.shape[1], image_array.shape[2], image_array.shape[1]*image_array.shape[2], pixelBurned, predPixelBurned, image_F1_delineation.numpy(), image_IOU_delineation.numpy(),\
                                         image_F1_severity.numpy()[0],image_F1_severity.numpy()[1],image_F1_severity.numpy()[2],image_F1_severity.numpy()[3],image_F1_severity.numpy()[4],np.mean(image_F1_severity.numpy()), \
                                         image_IOU_severity.numpy()[0],image_IOU_severity.numpy()[1],image_IOU_severity.numpy()[2],image_IOU_severity.numpy()[3],image_IOU_severity.numpy()[4],np.mean(image_IOU_severity.numpy()), \
                                         rmse_image[0],rmse_image[1],rmse_image[2],rmse_image[3],rmse_image[4],np.mean(rmse_image[:5]),rmse_image[5]
                                        ]

                self.local_F1_delineation.reset()
                self.local_IOU_delineation.reset() 
                self.local_F1_severity.reset()
                self.local_IOU_severity.reset()

                index += 1

        test_f1_delineation = self.f1_delineation.compute()
        test_IOU_delineation = self.iou_delineation.compute()
        test_f1_severity = self.f1_severity.compute()
        test_IOU_severity = self.iou_severity.compute()
        mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
        rmse = np.sqrt(mse)

        dict_results = {"Model_tested": self.args.model_path.split("/")[1],
                        "F1_delineation": test_f1_delineation.numpy().tolist(),
                        "IOU_delineation": test_IOU_delineation.numpy().tolist(),
                        "F1_severity": test_f1_severity.numpy().tolist(),
                        "F1_severity_mean": str(np.mean(test_f1_severity.numpy())),
                        "IOU_severity": test_IOU_severity.numpy().tolist(),
                        "IOU_severity_mean": str(np.mean(test_IOU_severity.numpy())),
                        "RMSE_severity":rmse.tolist()[:5],
                        "RMSE_severity_mean": str(np.mean(rmse[:5])),
                        "RMSE_severity_Total": str(np.mean(rmse[5]))
        }

        json_result_path = self.args.saving_folder_path + "/results.json"
        with open(json_result_path, 'w') as f:
            json.dump(dict_results, f)
        
        satellite_data_df_sorted = results_df.sort_values(['EMSR_AOI'], ascending = True)
        satellite_data_df_sorted.to_csv(csv_result_path, index=False)

        print(f"F1 delineation: {test_f1_delineation.numpy()}")
        print(f"IOU delineation: {test_IOU_delineation.numpy()}")
        print(f"F1 severity: {test_f1_severity.numpy()} - mean: {np.mean(test_f1_severity.numpy())}")
        print(f"IOU severity: {test_IOU_severity.numpy()} - mean: {np.mean(test_IOU_severity.numpy())}")
        print(f"RMSE severity: {rmse} - mean: {np.mean(rmse[:5])} - total: {rmse[5]}")

        self.iou_delineation.reset()
        self.f1_delineation.reset()
        self.iou_severity.reset()
        self.f1_severity.reset()

#######################################################################################################################################################################################
#
#                                                                        DELINEATION
#
###########################################################################################################################################################################################

class Test_Delineation:

    def __init__(self, args):

        self.args = args
        self.line_csv = ""

        self.cfg = Settings()
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.iou_delineation = BinaryJaccardIndex(ignore_index=255) #.to(self.device)
        self.f1_delineation = BinaryF1Score(ignore_index=255) #.to(self.device)

        self.local_F1_delineation = BinaryJaccardIndex(ignore_index=255) #.to(self.device)
        self.local_IOU_delineation = BinaryF1Score(ignore_index=255) #.to(self.device)

        if self.args.test_fold != None:
            self.cfg.satelliteData_path = "cross_validation/" + self.cfg.dataset_path_crossValidation

        if self.args.segmentation_network != None:
            self.segmentation_network = self.args.segmentation_network
        else:
            if "unet" in self.args.model_path.split("_"):
                self.segmentation_network = "unet"
            elif "unetplus" in self.args.model_path.split("_"):
                self.segmentation_network = "unetplus"
            elif "deeplab" in self.args.model_path.split("_"):
                self.segmentation_network = "deeplab"
            elif "deeplabplus" in self.args.model_path.split("_"):
                self.segmentation_network = "deeplabplus"
            elif "pspnet" in self.args.model_path.split("_"):
                self.segmentation_network = "pspnet"
            elif self.args.onnx != None:
                pass
            else:
                print("Checkout models name in the model's path")

    def get_list_path_images(self):

        self.filename_glob = "EMSR*AOI*S2L2A.tiff"
        pathname = os.path.join(Path(self.cfg.testData), "**", self.filename_glob)
        csv_satelliteData=Path(self.args.dataset_CSV_folder + self.cfg.satelliteData_path)

        df = pd.read_csv(csv_satelliteData, index_col=False)
        df = df[ (df["GRA"] == 1) & (df["folder"] == "optimal")]
        
        # ONLY for Cross Validation
        if self.args.test_fold != None:
            df = df.loc[ df["fold"] == int(self.args.test_fold) ] # test/val fold (Leave one out cross validation)

        listPaths = df["folderPath"].str.split("/").str[-1]
        filepaths = []
        for idx in listPaths:
            for f in glob.iglob(pathname, recursive=True):
                if idx in f:
                    filepaths.append(f)

        assert len(filepaths) > 0, f"No images found in {Path(self.cfg.testData)}"

        return filepaths


    def load_model(self):
        
        if self.segmentation_network == 'unet':
            self.model = smp.Unet(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None) #, decoder_attention_type="scse")
        elif self.segmentation_network == 'unetplus':
            self.model = smp.UnetPlusPlus(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None) #, decoder_attention_type="scse")
        elif self.segmentation_network == 'deeplab':
            self.model = smp.DeepLabV3(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
        elif self.segmentation_network == 'deeplabplus':
            self.model = smp.DeepLabV3Plus(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
        elif self.segmentation_network == 'pspnet':
            self.model = smp.PSPNet(encoder_name=self.args.encoder_name, encoder_weights=None, in_channels=12, classes=1, activation=None)
        else:
            print("Wrong segmentation_network name")
            return
        
        pre_load = self.model.segmentation_head[0].weight
    
        # Load pre-trained model or checkpoint
        if self.args.model_checkpoint_path is not None:
            checkpoint = torch.load(self.args.model_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            saved_state_dict = torch.load(self.args.model_path)
            self.model.load_state_dict(saved_state_dict)
            
        self.model.eval()

        post_load = self.model.segmentation_head[0].weight

        if torch.equal(pre_load, post_load):
            print(" ----  Model upload completed ---- ")         
        
        # self.model.to(self.device)

    def testing(self):
        
        self.test_dataset = self.get_list_path_images()
        self.load_model()

        if self.args.model_checkpoint_path is not None:
            self.args.saving_folder_path = self.args.saving_folder_path + self.args.model_checkpoint_path.split("/")[1].replace(".pth", "")
        else:
            self.args.saving_folder_path = self.args.saving_folder_path + self.args.model_path.split("/")[1]

        print(self.args.saving_folder_path)
        
        csv_result_path = self.args.saving_folder_path + "/results.csv"
        os.makedirs(os.path.dirname(csv_result_path), exist_ok=True)
        columns_names = ["EMSR_AOI","height","width","numPixel","pixelBurned","predBurnedPixel","F1","IOU","TP","TN","FP","FN"]
        results_df = pd.DataFrame(columns=columns_names)
        index = 0
        
        for image_path in self.test_dataset:

            print(image_path)
            image_mask_path = image_path.replace("_S2L2A.tiff", "_DEL.tif")
            image_cloud_mask_path = image_path.replace("_S2L2A.tiff", "_CM.tif")

            with rasterio.open(image_path) as image, rasterio.open(image_mask_path) as label, rasterio.open(image_cloud_mask_path) as cloud_mask:
                image_array = image.read()  # read the image as a numpy array
                label_array = label.read()
                label_array = label_array.squeeze()
                cloud_mask_array = cloud_mask.read()
                cloud_mask_array = cloud_mask_array.squeeze()

                label_array[(cloud_mask_array == 1) & (cloud_mask_array == 3)] = 255

                image_tensor = torch.from_numpy(image_array) #.float().to(self.device)  # transpose the array to get the channels as the last dimension (if necessary)
                label_tensor = torch.from_numpy(label_array)                

                # without SMOOTHING
                if self.args.smoothing == False:
                    if self.segmentation_network == 'unet':
                        pad_divisible = 32
                    elif self.segmentation_network == 'deeplab':
                        pad_divisible = 8
                    elif self.segmentation_network== 'deeplabplus':
                        pad_divisible = 16
                    elif self.segmentation_network == 'unetplus':
                        pad_divisible = 32
                    elif self.segmentation_network == 'pspnet':
                        pad_divisible = 8

                    new_height = (image_tensor.shape[1] // pad_divisible) * pad_divisible
                    new_width = (image_tensor.shape[2] // pad_divisible) * pad_divisible
                    image_tensor = image_tensor[:, :new_height, :new_width].unsqueeze(0)
                    label_tensor = label_tensor[:new_height, :new_width]
                    logits_delineation = self.model(image_tensor)
                    logits_delineation = logits_delineation.squeeze()
                
                # with SMOOTHING
                else:
                    logits_delineation = predict_smooth_windowing(  mode = "DEL",
                                                                    image = image_tensor,
                                                                    tile_size =  self.args.crop_size,
                                                                    subdivisions = 2,  # Minimal amount of overlap for windowing. Must be an even number.
                                                                    batch_size = self.args.batch_size,
                                                                    channels_first = True,
                                                                    prediction_fn = (
                                                                        lambda image: self.model(image))
                                                                    )

                predicted_tensor = (torch.sigmoid(logits_delineation) > 0.5).int()

                self.iou_delineation(logits_delineation, label_tensor)
                self.f1_delineation(logits_delineation, label_tensor)
                self.local_F1_delineation(logits_delineation, label_tensor)
                self.local_IOU_delineation(logits_delineation, label_tensor)

                save_path_image = image_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_mask = image_mask_path.replace(self.cfg.testData, self.args.saving_folder_path)
                save_path_predicted_mask = image_mask_path.replace(self.cfg.testData, self.args.saving_folder_path).replace("_DEL.tif", "_PRED.tif")
                os.makedirs(os.path.dirname(save_path_image), exist_ok=True)
               
                # print(save_path_image)
                # print(save_path_mask)
                # print(save_path_predicted_mask)

                if self.args.save_tif != None:
                    shutil.copyfile(image_path, save_path_image)
                    shutil.copyfile(image_mask_path, save_path_mask)

                    matedata_mask = label.meta
                    with rasterio.open(save_path_predicted_mask, 'w', **matedata_mask) as dst:
                        dst.write(np.expand_dims(predicted_tensor.numpy(), axis=0))

                # RGB images
                save_path_image_RGB = save_path_image.replace(".tiff", ".png")
                save_path_mask_RGB = save_path_mask.replace(".tif", ".png")
                save_path_predicted_mask_RGB = save_path_predicted_mask.replace(".tif", ".png")

                # print(save_path_image_RGB)
                # print(save_path_mask_RGB)
                # print(save_path_predicted_mask_RGB)

                if self.args.save_RGB != None:
                    shutil.copyfile(image_path.replace(".tiff", ".png"), save_path_image_RGB)
                    shutil.copyfile(image_mask_path.replace(".tif", ".png"), save_path_mask_RGB)

                    to_RGB_Mask(predicted_tensor*3, save_path_predicted_mask_RGB)

                image_F1 = self.local_F1_delineation.compute()
                image_IOU = self.local_IOU_delineation.compute()                
                emsr_aoi = [sub_folder for sub_folder in image_path.split("/") if "EMSR" in sub_folder][1]
                pixelBurned = np.count_nonzero( (label_array > 0) & (label_array < 120))
                predPixelBurned = np.count_nonzero(predicted_tensor.numpy() > 0)
                conf_matr = compute_confusion_matrix(predicted_tensor, label_tensor, labels_list=[0,1])

                results_df.loc[index] = [emsr_aoi,image_array.shape[1],image_array.shape[2],image_array.shape[1]*image_array.shape[2],pixelBurned,predPixelBurned,image_F1.numpy(),image_IOU.numpy(),conf_matr[1][1],conf_matr[0][0],conf_matr[0][1],conf_matr[1][0]]

                self.local_F1_delineation.reset()
                self.local_IOU_delineation.reset()

                index += 1

        test_f1_delineation = self.f1_delineation.compute()
        test_IOU_delineation = self.iou_delineation.compute()

        print(f"F1: {test_f1_delineation}")
        print(f"IOU: {test_IOU_delineation}")

        dict_results = {"Model_tested": self.args.model_path.split("/")[1],
                        "F1_delineation": test_f1_delineation.numpy().tolist(),
                        "IOU_delineation": test_IOU_delineation.numpy().tolist(),
        }

        json_result_path = self.args.saving_folder_path + "/results.json"
        with open(json_result_path, 'w') as f:
            json.dump(dict_results, f)

        satellite_data_df_sorted = results_df.sort_values(['EMSR_AOI'], ascending = True)
        satellite_data_df_sorted.to_csv(csv_result_path, index=False)

        self.iou_delineation.reset()
        self.f1_delineation.reset() 

          
def main(params):

    # basic parameters
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default="multitask", help='Mode to test: delineation (DEL) or severity (multitask)')
    parser.add_argument('--severity_convention', type=int, default=4, help="Convention severity of copernicus, it can 4 or 3 different levels")

    # network
    parser.add_argument('--onnx', type=bool, default=None, help='Is model Onnx type?')
    parser.add_argument('--encoder_name', type=str, default="resnet50", help='Models encoder used, chose between resnet34, resnet50')
    parser.add_argument('--segmentation_network', type=str, default=None, help='Type of segmentation network used to train')
    parser.add_argument('--crop_size', type=int, default=512, help='Cropped size for smoothing')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for smoothing')
    
    # parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')

    # paths
    parser.add_argument('--dataset_CSV_folder', type=str, default=None, help='path to dataset csv')
    parser.add_argument('--model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--model_checkpoint_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--saving_folder_path', type=str, default=None, help='path to pretrained model')

    # training
    parser.add_argument('--test_fold', type=int, default=None, help='folds used as test set')

    # flag
    parser.add_argument('--save_tif', type=bool, default=None, help='save tiff images for the mask?')
    parser.add_argument('--save_RGB', type=bool, default=None, help='save RGB images for the mask?')
    parser.add_argument('--smoothing', type=bool, default=True, help='would you use smoothing to make prediction?')

    args = parser.parse_args(params)
    print(params)
    
    # list_models = [ # "CV_6F_GRA_unet_24000_0528_17",
    #                 # "CV_6F_GRA_unet_24000_0529_00",
    #                 # "CV_6F_GRA_unet_24000_0528_00",
    #                 # "CV_6F_GRA_unet_24000_0527_10",
    #                 # "CV_6F_GRA_unet_24000_0530_00",
    #                 # "CV_6F_GRA_unet_24000_0526_23",
    #                 # "CV_6F_GRA_unetplus_24000_0530_19",
    #                 # "CV_6F_GRA_unetplus_24000_0529_00",
    #                 # "CV_6F_GRA_unetplus_24000_0528_00",
    #                 # "CV_6F_GRA_unetplus_24000_0530_17",
    #                 # "CV_6F_GRA_unetplus_24000_0530_00",
    #                 # "CV_6F_GRA_unetplus_24000_0526_23",
    #                 # "CV_6F_GRA_deeplab_24000_0528_17",
    #                 # "CV_6F_GRA_deeplab_24000_0530_23",
    #                 # "CV_6F_GRA_deeplab_24000_0528_00",
    #                 # "CV_6F_GRA_deeplab_24000_0527_10",
    #                 # "CV_6F_GRA_deeplab_24000_0530_00",
    #                 # "CV_6F_GRA_deeplab_24000_0526_23",
    #                 # "CV_6F_GRA_deeplabplus_24000_0523_21",
    #                 # "CV_6F_GRA_deeplabplus_24000_0529_00",
    #                 # "CV_6F_GRA_deeplabplus_24000_0531_09",
    #                 # "CV_6F_GRA_deeplabplus_24000_0530_17",
    #                 # "CV_6F_GRA_deeplabplus_24000_0525_17",
    #                 # "CV_6F_GRA_deeplabplus_24000_0525_18",
    #                 # "CV_6F_GRA_pspnet_24000_0523_23",
    #                 # "CV_6F_GRA_pspnet_24000_0525_11",
    #                 # "CV_6F_GRA_pspnet_24000_0530_10",
    #                 # "CV_6F_GRA_pspnet_24000_0526_23",
    #                 # "CV_6F_GRA_pspnet_24000_0530_11",
    #                 # "CV_6F_GRA_pspnet_24000_0530_17",
    #                 # "CV_6F_GRA_unetplusAtt_24000_0525_17" #ATTENTION UNET
    #                 ]

    # base_path = args.model_path
    # for model in list_models:
    #     args.model_path = base_path.replace(base_path.split("/")[1], model)
    #     args.saving_folder_path = 'prediction_test/'
    #     print(args.model_path)

    if args.mode == "DEL":
        test_model = Test_Delineation(args)
        test_model.testing()
    elif args.mode == "GRA":
        test_model = Test_Severity(args)
        test_model.testing()
    elif args.mode == "multitask":
        test_model = Test_Multitask(args)
        test_model.testing()
    else:
        print("Wrong mode selected, multitask or DEL.")


if __name__ == '__main__': 

    params = [

        # mode
        '--mode', "GRA", # DEL, GRA, multitask
        # '--severity_convention', "4", # 4, 3

        # dataloader and sampler
        # '--onnx', 'True',
        # '--encoder_name', "resnet50",
        # '--segmentation_network', 'deeplabplus', # unet, deeplab, deeplabplus, unetplus, pspnet
        '--crop_size', '512', #512 #128
        '--batch_size', '12', #12 #5       

        # hardware
        # '--num_workers', '0', 
        # '--use_gpu', 'True',

        # model path
        '--dataset_CSV_folder', 'dataset/',
        '--model_path', 'models/GRA_unet_24000_0724_09/GRA_unet_24000_0724_09.pth', #'models/DoubleStep/concat_unet-v1.0.onnx', #'models/CV_6F_GRA_deeplabplus_24000_0529_00/CV_foldTest_5.pth', #'/home/merlo/multitask-segmentation/Semantic Segmentation/models/CV_6F_unet_5040_00030_0413_16/CV_foldTest_6.pth',
        # '--model_checkpoint_path', 'checkpoints/CP20_F0_0423_20.pth',
        '--saving_folder_path', 'prediction_test/',

        # date
        # '--date', datetime.today().strftime('%Y%m%d_%H'),

        # testing
        '--test_fold', '0',

        # flag
        '--save_RGB', 'True',
        # '--save_tif', 'True',
    ]

    main(params)



    #---------------------------------------------- COMMANDS ---------------------------------------------------------------

    # run test
    # CUDA_VISIBLE_DEVICES=3 python test.py

    # usage GPU
    # nvidia-smi

    # usage CPU/RAM
    # htop
