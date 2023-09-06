from pathlib import Path
from typing import List, Optional
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import glob
import os
import sys
import json
import rasterio
import torch
from torch import nn
import pandas as pd
import numpy as np
import random
from datetime import datetime
from PIL import Image
from sklearn.metrics import mean_squared_error, f1_score

from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42) -> None:

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def weight_init(model, seed=None):

    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d) or isinstance(model, nn.Linear):
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.xavier_normal_(model.weight.data)
        # nn.init.normal_(model.bias.data)


def save_log2df(tensorboard_path, tensorboard_csv):

    dirname = tensorboard_path
    
    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    scalar_names = ea.Tags()['scalars']
    
    for n in scalar_names:
        dframes[n] = pd.DataFrame( ea.Scalars(n), columns=["wall_time", "step", "value"])
        dframes[n].drop("wall_time", axis=1, inplace=True)
        dframes[n] = dframes[n].rename(columns={"step": "epoch", "value": n})
        dframes[n] = dframes[n].set_index("epoch")

    if bool(dframes) == False: # dict is empty
        return
    
    df = pd.concat([v for k,v in dframes.items()], axis=1)
    df.to_csv(tensorboard_csv)


def save_model_configuration(json_save_path, args, test_IoU_delineation, test_f1_delineation, test_rmse=None, test_IoU_severity=None, test_f1_severity=None, conf_matr_delineation = None, conf_matr_severity = None, cfg = None):

    time_execution = round(( datetime.strptime(datetime.now().strftime('%m%d_%H%M'), '%m%d_%H%M' ) - datetime.strptime(args.date.strftime( '%m%d_%H%M' ), '%m%d_%H%M')).total_seconds() / 60) 
    args.date = args.date.strftime('%m%d_%H')
    dict_args = vars(args)
    dict_args.update({"Time_execution_in_minutes": str( time_execution )})
    
    if conf_matr_delineation is not None:
        # tp = conf_matr_delineation[0][0].item()
        # tn = conf_matr_delineation[1][1].item()
        # fp = conf_matr_delineation[1][0].item()
        # fn = conf_matr_delineation[0][1].item()
        # dict_args.update({'Confusion_Matrix_Delineation': {"TP": tp, "TN": tn, "FP": fp, "FN": fn} })
        dict_args.update({"Test_IOU_delineation": test_IoU_delineation.cpu().numpy().tolist() })
        dict_args.update({"Test_F1_delineation": test_f1_delineation.cpu().numpy().tolist() })
        dict_args.update({'Confusion_Matrix_Delineation': conf_matr_delineation.tolist() })
    
        prec, recall, f1, acc = compute_prec_recall_f1_acc(conf_matr_delineation)
        dict_args.update({"Precision_Delineation_[0,1]": prec.tolist(), "Recall_Delineation_[0,1]": recall.tolist(), "F1_Delineation_[0,1]": f1.tolist(), "Accuracy_Delineation": acc})
    
    if conf_matr_severity is not None:
        dict_args.update({"Test_RMSE_Severity": test_rmse.tolist()[:cfg.severity_convention+1]})
        dict_args.update({"Test_RMSE_Mean_Severity": np.mean(test_rmse.tolist()[:cfg.severity_convention+1])})
        dict_args.update({"Test_RMSE_Total_Severity": test_rmse.tolist()[cfg.severity_convention+1]})
        dict_args.update({"Test_F1_Severity": test_f1_severity.cpu().numpy().tolist() })
        dict_args.update({"Test_F1_Mean_Severity": np.mean(test_f1_severity.cpu().numpy().tolist()) })
        dict_args.update({"Test_IoU_Severity": test_IoU_severity.cpu().numpy().tolist() })
        dict_args.update({"Test_IoU_Mean_Severity": np.mean(test_IoU_severity.cpu().numpy().tolist()) })
        dict_args.update({"Confusion_Matrix_Severity":  conf_matr_severity.tolist() })

        prec, recall, f1, acc = compute_prec_recall_f1_acc(conf_matr_severity)
        dict_args.update({"Precision_Severity_[0,1,2,3,4]": prec.tolist(), "Recall_Severity_[0,1,2,3,4]": recall.tolist(), "F1_Severity_[0,1,2,3,4]": f1.tolist(), "Accuracy_Severity": acc})

    json_object = json.dumps(dict_args, indent=4)
    with open(json_save_path, "w") as outfile:
        outfile.write(json_object)


def save_configuration_tb(writer: SummaryWriter, args, test_IoU_delineation, test_f1_delineation, test_rmse=None, test_IoU_severity=None, test_f1_severity=None, conf_matr_delineation = None, conf_matr_severity = None, cfg = None):

    writer.add_text("Config_Hyperparameter_Model", f"Mode: {args.mode},  \nEncoder_name: {args.encoder_name},  \Segmentation_network: {args.segmentation_network},  \nEpochs: {args.num_epochs},  \nSeed: {args.seed}")
    writer.add_text("Config_Hyperparameter_Batch", f"Crop_size: {args.crop_size},  \nStride: {args.stride},  \nBatch_size: {args.batch_size},  \nNum_epochs: {args.num_epochs},  \nSample_per_epoch: {args.sample_per_epoch},  \nBurn_prop_batch: {args.burn_prop_batch},  \nBurn_area_prop: {args.burn_area_prop}")
    writer.add_text("Config_Optimizer_Loss", f"Optimizer: {args.optimizer},  \nLearning_rate: {args.learning_rate},  \nScheduler: {args.lr_policy},  \nLoss: {args.loss}" )
    writer.add_text("Config_Time_Execution", f"Execution time in minutes: {round(( datetime.strptime(datetime.today().strftime('%m%d_%H%M'), '%m%d_%H%M' ) - datetime.strptime(args.date.strftime( '%m%d_%H%M' ), '%m%d_%H%M')).total_seconds() / 60)}")

    if conf_matr_delineation is not None:
        # tp = conf_matr_delineation[0][0].item()
        # tn = conf_matr_delineation[1][1].item()
        # fp = conf_matr_delineation[1][0].item()
        # fn = conf_matr_delineation[0][1].item()
        # writer.add_text("Results_Delineation", f"Confusion_Matrix_Delineation:  \nTP: {tp},  \nTN: {tn},  \nFP: {fp},  \nFN: {fn}")
        writer.add_text("Results_Delineation", f"Test IoU Delineation: {test_IoU_delineation.cpu().numpy().tolist()},  \nTest F1 Delineation: {test_f1_delineation.cpu().numpy().tolist()}") #{test_IOU.cpu().numpy().tolist()},  \ntest_F1: {test_f1.cpu().numpy().tolist()}")
        writer.add_text("Results_Delineation", f"Confusion_Matrix_Delineation: \n{conf_matr_delineation}")

        prec, recall, f1, acc = compute_prec_recall_f1_acc(conf_matr_delineation)
        writer.add_text("Results_Delineation", f"Precision [0,1]: \t{prec},  \nRecall [0,1]: \t{recall},  \nF1 [0,1]: \t\t{f1},  \nAccuracy: \t\t{acc}")

    if conf_matr_severity is not None or test_IoU_severity is not None:
        writer.add_text("Results_Severity", f"Test RMSE Severity: {test_rmse.tolist()[:cfg.severity_convention+1]}")
        writer.add_text("Results_Severity", f"Test RMSE_Mean Severity: {np.mean(test_rmse.tolist()[:cfg.severity_convention+1])}")
        writer.add_text("Results_Severity", f"Test RMSE_Total Severity: {test_rmse.tolist()[cfg.severity_convention+1]}")
        writer.add_text("Results_Severity", f"Confusion_Matrix_Severity:  \n{conf_matr_severity}")
        writer.add_text("Results_Severity", f"Test IoU Severity Mean: {np.mean(test_IoU_severity.cpu().numpy()).tolist()},  \nTest F1 Severity Mean: {np.mean(test_f1_severity.cpu().numpy()).tolist()}")
        writer.add_text("Results_Severity", f"Test IoU Severity: {test_IoU_severity.cpu().numpy().tolist()},  \nTest F1 Severity: {test_f1_severity.cpu().numpy().tolist()}") #{test_IOU.cpu().numpy().tolist()},  \ntest_F1: {test_f1.cpu().numpy().tolist()}")

        prec, recall, f1, acc = compute_prec_recall_f1_acc(conf_matr_severity)
        writer.add_text("Results_Severity", f"Precision [0,1,2,3,4]: \t{prec},  \nRecall [0,1,2,3,4]: \t{recall},  \nF1 [0,1,2,3,4]: \t\t{f1},  \nAccuracy: \t\t{acc}")

    return writer


def compute_confusion_matrix(mask_pred, mask_truth, labels_list = None):

        mask_truth = mask_truth.cpu().numpy().flatten()
        mask_pred = mask_pred.cpu().numpy().flatten()

        if labels_list == None:
            labels_list = [0, 1]

        conf_matr = confusion_matrix(mask_truth, mask_pred, labels=labels_list)

        return conf_matr


def compute_prec_recall_f1_acc(conf_matr):
    
    accuracy = np.trace(conf_matr) / conf_matr.sum()

    predicted_sum = conf_matr.sum(axis=0)
    gt_sum = conf_matr.sum(axis=1)
                
    diag = np.diag(conf_matr)
    precision = diag / predicted_sum.clip(min=1e-5)
    recall = diag / gt_sum.clip(min=1e-5)
    f1 = 2 * (precision * recall) / (precision + recall).clip(min=1e-5)
    return precision, recall, f1, accuracy


def remap_severity_labels(mask, num_labels: int = 4):
    
    if num_labels == 4:
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
        # ranges = [(0, 0.125), (0.125, 0.375), (0.375, 0.625), (0.625, 0.875), (0.875, 1.00001)]
    elif num_labels == 3:
        ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]
        # ranges = [(0, 0.165), (0.165, 0.5), (0.5, 0.825), (0.825, 1.00001)]
    else:
        print("Severity labels levels wrong, is 3 or 4?")
    
    remapped_mask = torch.zeros_like(mask, dtype=torch.float) # Create an empty tensor to hold the remapped values

    for i, (lower, upper) in enumerate(ranges):
        remapped_mask = torch.where((mask >= lower) & (mask < upper), i, remapped_mask)    
    
    remapped_mask = torch.where(mask > 15, 255, remapped_mask) # no data label restored

    return remapped_mask


def compute_squared_errors(_prediction, _ground_truth, n_classes, check=True):

    """
    Separately for each class, compute total squared error (sq_err_res), and total count (count_res)
    
    Returns:
    -----
    (tuple)
    sq_errors: np.array([sq_err_class0, sq_err_class1,
               ..., total_sq_Err_all_classes])
               
    counts: np.array([n_pixels_class0, n_pixels_class1,
            ..., total_pixels])
    """
    
    prediction = _prediction[ _ground_truth != 255 ] # same position of no data point -> mse equal 0
    ground_truth = _ground_truth[_ground_truth != 255]
    
    squared_errors = []
    counters = []

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.squeeze().cpu().numpy()

    if len(ground_truth.shape) == 3 and ground_truth.shape[-1] == 1:
        ground_truth = ground_truth.squeeze(axis=-1)

    if len(prediction.shape) == 3 and prediction.shape[-1] == 1:
        prediction = prediction.squeeze(axis=-1)

    mse_check = []

    for idx in range(n_classes):
        mask = ground_truth == idx
        pred_data = prediction[mask]    # Predicted pixels corresponding to ground truth elements of class idx
        gt_data = ground_truth[mask]    # Ground truth pixels with class idx
        sq_err = np.square(pred_data - gt_data).sum()   # Squared error for those pixels
        n_elem = mask.sum()                             # Number of considered pixels

        squared_errors.append(sq_err)
        counters.append(n_elem)

        if check:
            if n_elem > 0:
                mse_check.append(mean_squared_error(gt_data, pred_data))
            else:
                mse_check.append(0)

    sq_err = np.square((prediction - ground_truth).flatten()).sum()     # Total squared error (all classes)

    if check:
        mse_check.append(mean_squared_error(ground_truth.flatten(), prediction.flatten()))
    n_elem = prediction.size
    squared_errors.append(sq_err)   # [sq_err_class0, sq_err_class1,..., total_sq_Err_all_classes]
    counters.append(n_elem)         # total n. pixels

    sq_err_res = np.array(squared_errors)
    count_res = np.array(counters)

    if check:
        mymse = sq_err_res / count_res.clip(min=1e-5)
        mymse[np.isnan(mymse)] = 0

        mse_check = np.array(mse_check)
        assert (np.abs(mymse - mse_check) < 1e-6).all()

    return sq_err_res, count_res


def to_RGB_Mask(_image_mask, path: str):

        """Save a RGB mask in .png format

        Args:
            image_mask (np.ndarray): image with only one channel
            path (str): save path 
           
        Returns:
            None
        """

        if torch.is_tensor(_image_mask):
            image_mask = torch.clone(_image_mask)
            if image_mask.is_cuda:
                image_mask = image_mask.cpu().numpy()
        else:
            image_mask = _image_mask.copy()        

        if len(image_mask.shape) > 3:
            image_mask = image_mask.squeeze()

        if image_mask.shape[0] == 1:
            image_mask = np.transpose(image_mask, (1, 2, 0))

        image_mask[ image_mask > 200 ] = 10
        unique_values = set(np.unique(image_mask).astype(int).tolist()) # getting unique classes
        
        colors = [  (0,0,0),            # 0 = Black = No burned
                    (181,254,142),      # 1 = Greenish = Negligible damage
                    (254,217,142),      # 2 = Light Orange = Moderate damage
                    (254,153,41),       # 3 = Orange = High damage
                    (204,76,2),         # 4 = Reddish = Destruction
                    (240,20,240),       # 5 = Purple = Cloud overlap with burned area
                    (103,190,224),      # 6 = Light blue = Clear sky
                    (220,220,220),      # 7 = White = Cloud
                    (180,180,180),      # 8 = Grey = Light cloud
                    (60,60,60),         # 9 = Dark grey = Cloud's shadow
                    (255,255,255),      # 10 = White = No Data
                    (255,255,255)       # 11 = White = No Data
                    ]

        Rarr = np.zeros_like(image_mask, dtype = 'uint8') # Red
        Garr = np.zeros_like(image_mask, dtype = 'uint8') # Green
        Barr = np.zeros_like(image_mask, dtype = 'uint8') # Blue
        for val, col in zip(unique_values, colors):
            Rarr[image_mask == val ] = colors[val][0] #col[0]
            Garr[image_mask == val ] = colors[val][1] #col[1]
            Barr[image_mask == val ] = colors[val][2] #col[2]

        RGB_mask = np.dstack((Rarr,Garr,Barr)) # Combining three channels

        image_RGB = Image.fromarray(RGB_mask, 'RGB')
        image_RGB.save(path)
        

def to_RGB_sample(_image_sample, path: str, RGB_image: bool = False, false_color = False):

        """Save a RGB sample in .png format

        Args:
            image_mask (np.ndarray): image with only one channel
            path (str): save path 
           
        Returns:
            None
        """

        if torch.is_tensor(_image_sample):
            image_sample = torch.clone(_image_sample)
            if image_sample.is_cuda:
                image_sample = image_sample.cpu().detach().numpy()
        else:
            image_sample = _image_sample.copy()

        if len(image_sample.shape) > 3:
            image_sample = image_sample.squeeze()

        if image_sample.shape[0] == 12 or image_sample.shape[0] == 3 or image_sample.shape[0] == 1:
            image_sample = np.transpose(image_sample, (1, 2, 0))

        contrast = 10000/255*3.5*5.7        # image color contrast
        brightness = 30 #30                 # image brightness 

        # imageRGB = image_np[:,:,[3,2,1]]
        if false_color == True:
            imageRGB = np.multiply(image_sample[:,:,[7,3,2]], contrast) + brightness
        elif RGB_image == False:
            imageRGB = np.multiply(image_sample[:,:,[3,2,1]], contrast) + brightness
        elif RGB_image == True:
            imageRGB = np.multiply(image_sample, contrast) + brightness
        
        imageRGB[ imageRGB > 255 ] = 255
        imageRGB[ imageRGB < 0 ] = 0        
        image_RGB = Image.fromarray(imageRGB.astype(np.uint8))
        image_RGB.save(path)
    
       
