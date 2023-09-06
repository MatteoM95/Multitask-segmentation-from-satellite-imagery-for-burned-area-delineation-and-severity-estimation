from pydantic import BaseSettings


class Settings(BaseSettings):

    # path folders
    working_folder: str = "../../"
    rootData = working_folder + "data/dataOptimal"

    # path csv
    satelliteData_path: str = "satelliteData.csv"
    
    dataset_path_train: str = "satelliteDataTrain.csv"
    dataset_path_val: str = "satelliteDataVal.csv"
    dataset_path_test: str = "satelliteDataTest.csv"

    dataset_path_crossValidation: str = "satelliteData_6folds.csv"
    dataset_path_crossValidation_severity: str = "satelliteData_6folds_Severity.csv"
    
    # path tensorboard and runs
    tb_runs_configuration_folder = "tb_runs_configuration/"

    annotation_type_delineation = ["del_mask", "cloud_mask"] # "lc_esa_mask", "lc_annual_mask",
    annotation_type_severity = ["del_mask", "gra_mask", "cloud_mask"] # "lc_esa_mask", "lc_annual_mask",

    severity_convention = 4 # 3 or 4


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
