from wildfires.training_delineation import Training_Delineation
from wildfires.training_severity import Training_Severity
from wildfires.training_multitask import Training_Multitask

from wildfires.crossvalidator_delineation import CrossValidator_Delineation
from wildfires.crossvalidator_severity import CrossValidator_Severity
from wildfires.crossvalidator_multitask import CrossValidator_Multitask

from wildfires.settings import Settings
import os
import json
import pandas as pd
from datetime import datetime


class TrainModel:

    def __init__(self, args):
        self.args = args
        self.cfg = Settings()
     
     
    def start(self):
        
        # delineation only
        if self.args.mode == "DEL":
            # cross validation
            if "cross_validation" in self.args.dataset_CSV_folder.split("/"):
                train_with_cross_validation = CrossValidator_Delineation(self.args)
                train_with_cross_validation.start_cross_validation()
                self.create_summary_json()
            
            # training using CSV train-val-test
            else:
                train_without_cross_validation = Training_Delineation(self.args)
                train_without_cross_validation.start_initialize_and_train()

        # severity only
        elif self.args.mode == "GRA":
            # cross validation
            if "cross_validation" in self.args.dataset_CSV_folder.split("/"):
                train_with_cross_validation = CrossValidator_Severity(self.args)
                train_with_cross_validation.start_cross_validation()
                self.create_summary_json()
                
            # training using CSV train-val-test
            else:
                train_without_cross_validation = Training_Severity(self.args)
                train_without_cross_validation.start_initialize_and_train()

        # multitask approach
        elif self.args.mode == "multitask":
            # cross validation
            if "cross_validation" in self.args.dataset_CSV_folder.split("/"):
                train_with_cross_validation = CrossValidator_Multitask(self.args)
                train_with_cross_validation.start_cross_validation()
                self.create_summary_json()
                
            # training using CSV train-val-test
            else:
                train_without_cross_validation = Training_Multitask(self.args)
                train_without_cross_validation.start_initialize_and_train()
            

    def create_summary_json(self):

        print(f" --- Creation summary Cross-Validation --- {datetime.today().strftime('%Y%m%d_%H')}")

        folds = 6
        model_name_folder = 'CV_{}_{}_{}_{}/'.format( self.cfg.dataset_path_crossValidation.split("_")[1].replace(".csv",""), self.args.segmentation_network, self.args.num_epochs * self.args.sample_per_epoch, self.args.date)
        folder_configurations = self.cfg.tb_runs_configuration_folder + model_name_folder 

        # search for each fold the the results in configuration.json
        paths_json = []
        for root, dirs, files in os.walk(folder_configurations):
            for file in files:
                if file.endswith("configuration.json"):
                    
                    path_file = os.path.join(root,file)
                    paths_json.append(path_file)
        
        # if the number of file found is different from expected, exit, avoiding to do the summary results
        if len(paths_json) != int(folds):
            return
        
        json_originale = None
        IoU = 0
        f1 = 0
        total_TP = 0
        total_TN = 0
        total_FP = 0
        total_FN = 0
        precision_01 = [0,0]
        recall_01 = [0,0]
        f1_01 = [0,0]
        accuracy = 0

        # for each configuration file read and save the result in summary
        for json_file_path in paths_json:
            
            with open(json_file_path, "r") as f:
                json_file = json.load(f)
                json_originale = json_file

                IoU += json_file["test_IOU"]
                f1 += json_file["test_F1"]

                total_TP += json_file["confusion_matrix"]["TP"]
                total_TN += json_file["confusion_matrix"]["TN"]
                total_FP += json_file["confusion_matrix"]["FP"]
                total_FN += json_file["confusion_matrix"]["FN"]
                
                precision_01 = [sum(x) for x in zip(precision_01, json_file["precision [0,1]"])]
                recall_01 = [sum(x) for x in zip(recall_01, json_file["recall [0,1]"])] 
                f1_01 = [sum(x) for x in zip(f1_01, json_file["f1 [0,1]"])] 
                accuracy += json_file["accuracy"]
        
        mean_IOU = IoU/folds
        mean_f1 = f1/folds
        mean_TP = int(total_TP / folds)
        mean_TN = int(total_TN / folds)
        mean_FP = int(total_FP / folds)
        mean_FN = int(total_FN / folds)
        mean_conf_matr = {
            "confusion_matrix": {
                "TP": mean_TP,
                "TN": mean_TN,
                "FP": mean_FP,
                "FN": mean_FN
                }
            }
        
        for i in range(2):
            precision_01[i]/=folds
            recall_01[i]/=folds
            f1_01[i]/=folds

        accuracy /= folds

        # delete useless keys
        del json_originale["test_IOU"]
        del json_originale["test_F1"]
        del json_originale["confusion_matrix"]
        del json_originale["precision [0,1]"]
        del json_originale["recall [0,1]"]
        del json_originale["f1 [0,1]"]
        del json_originale["accuracy"]
        del json_originale["time_execution_in_hours"]
        del json_originale['test_folds']
        del json_originale['pretrained_model_path']
        del json_originale['epoch_start_i']

        json_originale.update([("test_IOU", mean_IOU), 
                               ("test_F1", mean_f1), 
                               ("confusion_matrix", mean_conf_matr),
                               ("precision [0,1]", precision_01),
                               ("recall [0,1]", recall_01),
                               ("f1 [0,1]", f1_01),
                               ("accuracy", accuracy),
                               ])

        # save json with summary results
        with open("tb_runs_configuration/" + model_name_folder + "summary_config.json", "w") as f:
            json.dump(json_originale, f)

        return
