"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : utils.py
* Description       : This file contains utils functions.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with average computation and json, csv saving functions.
*
******************************************************************"""


import json
import datetime 
import pytz
from typing import Any, Dict, List, Union
import os
import csv


def compute_averages(data_list: List[Dict[str, Union[float, Dict[str, Union[float, str]]]]]) -> Dict[str, float]:
    """
    Calculate the averages of 'hwords', 'precision', 'recall' and 'f1' at the end of all iterations.

    Parameters:
        data_list (List[Dict[str, Union[float, Dict[str, Union[float, str]]]]]): A list of dictionaries containing
            'hwords' (float) and 'hbert' (a dictionary containing 'precision', 'recall', 'f1' as floats).

    Returns:
        Dict[str, float]: A dictionary containing the averages of 'hwords', 'precision', 'recall' and 'f1' over the iterations.
    """
    all_hwords = []
    all_precision = []
    all_recall = []
    all_f1 = []

    for item in data_list:
        all_hwords.append(float(item['hwords']))
        all_precision.append(float(item['hbert']['precision'][0]))
        all_recall.append(float(item['hbert']['recall'][0]))
        all_f1.append(float(item['hbert']['f1'][0]))

    average_hwords = sum(all_hwords) / len(all_hwords)
    average_precision = sum(all_precision) / len(all_precision)
    average_recall = sum(all_recall) / len(all_recall)
    average_f1 = sum(all_f1) / len(all_f1)

    return {
        "avg_hwords": average_hwords,
        "avg_precision": average_precision,
        "avg_recall": average_recall,
        "avg_f1": average_f1
    }


def upgrade_file_version(folder_path: str, file_type: str) -> str:
    """
    Upgrade the version of the file to save.

    Parameters:
        folder_path (str): The folder to save the files.
        file_type (str): The type of file to upgrade the version.

    Returns:
        str: The new version of the file.

    Warning:
        This function is specific to the file naming convention used in this project.
        We assume 2 files in the folder:
        - file name starts with 'info_json' and are json files; 
        They are the information files.
        - file name starts with 'final_modif' and are .parquet files.
        They are the final modified dataset files.
    """

    # find the json file in the folder 
    new_version = "0"
    for f in os.listdir(folder_path):
        if (f.endswith(file_type) and f.startswith('info_json')):
            file_path = os.path.join(folder_path, f) 
            version = file_path.split("_")[-1].split(".")[0][1:]
            new_version = str(int(version) + 1)

        elif (f.endswith(file_type) and f.startswith('final_modif')):
            file_path = os.path.join(folder_path, f) 
            version = file_path.split("_")[-1].split(".")[0][1:]
            new_version = str(int(version) + 1)

    return new_version
    


def save_json(data: Dict[str, Any], folder_path: str) -> str:
    """ 
    Save a dictionary as a JSON file in the specified folder and return the path to the saved file.

    Parameters:
        data (Dict[str, Any]): The dictionary to save.
        folder_path (str): The folder to save the JSON file.
    
    Returns:
        str: The path to the saved JSON file.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    json_file_version = upgrade_file_version(folder_path, ".json")
    json_file_path = folder_path + "info_json" + get_local_date_time() + "_v" + json_file_version + ".json" 
    
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return json_file_path
    



def get_local_date_time() -> str:
    """
    Get the current local date and time.
    Returns:
        str: The current local date and time in the format "YYYY-MM-DD HH:MM:SS.mmmmmm+HH:MM".
        .2024-04-04 11:40:11.136104+02:00
        The time zone is the local time zone of the machine running the script.

    """
    # Get the current time in UTC
    utc_now = datetime.datetime.now(pytz.utc)
    local_timezone = pytz.timezone('Europe/Paris')  # Example: 'Europe/Paris'

    # Convert UTC time to the local time
    local_now = utc_now.astimezone(local_timezone).replace(microsecond=0).isoformat()

    return local_now


def create_initial_csv(csv_filename: str, fieldnames: List[str]) -> None:
    """
    Create a CSV file with the specified fieldnames if it does not exist.

    Parameters:
        csv_filename (str): The name of the CSV file to create.
        fieldnames (List[str]): The fieldnames of the CSV file.
    """

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def select_data_to_save(data_dict: dict, json_path: str, final_modif_dataset_path: str) -> dict:
    """
    Select the information to save in the CSV file.

    Parameters:
        data_dict (dict): The dictionary containing the information to save.
        json_path (str): The path to the JSON file containing the information.
        final_modif_dataset_path (str): The path to the final modified dataset file.

    Returns:
        dict: A dictionary containing the selected information to save in the CSV file.
    """
    
    selected_info_dict = {
    "date": data_dict.get("date_time", ""),
    "nb_iter": data_dict.get("number_iterations", ""),
    "partial_dataset_size": data_dict.get("partial_dataset_size", ""),
    "check_fidelity": data_dict.get("Inference_mode : check_fidelity_mode = ", ""),
    "THR1_hwords": data_dict.get("Thresholds if check_fidelity_mode ", {}).get("THR1", ""),
    "THR2_hbert": data_dict.get("Thresholds if check_fidelity_mode ", {}).get("THR2", ""),
    "average_hwords": data_dict.get("averages", {}).get("avg_hwords", ""),
    "average_hbert_precision": data_dict.get("averages", {}).get("avg_precision", ""),
    "average_hbert_recall": data_dict.get("averages", {}).get("avg_recall", ""),
    "average_hbert_f1": data_dict.get("averages", {}).get("avg_f1", ""),
    "initial_classifier_results_precision": data_dict.get("initial_classifier_results", {}).get("precision", ""),
    "initial_classifier_results_recall": data_dict.get("initial_classifier_results", {}).get("recall", ""),
    "initial_classifier_results_f1": data_dict.get("initial_classifier_results", {}).get("f1", ""),
    "prompt_used": data_dict.get("prompt_used", ""),
    "few_shot_strategy": data_dict.get("few_shot_strategy", ""),
    "ensemble_strategy": data_dict.get("ensemble_strategy", ""),
    "attack_success_rate": data_dict.get("attack_success_rate", ""),
    "final_modif_success_rate": data_dict.get("final_modif_success_rate", ""),
    "final_classifier_results_precision": data_dict.get("final_classifier_results", {}).get("precision", ""),
    "final_classifier_results_recall": data_dict.get("final_classifier_results", {}).get("recall", ""),
    "final_classifier_results_f1": data_dict.get("final_classifier_results", {}).get("f1", ""),

    "loaded_dataset": data_dict.get("loaded_dataset", ""),
    "fooled_model_name": data_dict.get("fooled_model_info", {}).get("model_name", ""),
    "fooled_tokenizer_name": data_dict.get("fooled_tokenizer_info", {}).get("tokenizer_name", ""),
    "fooled_tokenizer_vocab_size": data_dict.get("fooled_tokenizer_info", {}).get("vocab_size", ""),
    "fooled_tokenizer_is_fast": data_dict.get("fooled_tokenizer_info", {}).get("is_fast", ""),
    "fooled_tokenizer_padding_side": data_dict.get("fooled_tokenizer_info", {}).get("padding_side", ""),
    "fooled_tokenizer_truncation_side": data_dict.get("fooled_tokenizer_info", {}).get("truncation_side", ""),
    "fooled_tokenizer_clean_up_tokenization_spaces": data_dict.get("fooled_tokenizer_info", {}).get("clean_up_tokenization_spaces", ""),
    "initial_classifier_name": data_dict.get("initial_classifier", ""),
    
    "info_json_path": json_path,
    "final_modif_dataset_path": final_modif_dataset_path
    }

    return selected_info_dict


def save_info_to_csv(info_dict_csv: dict, csv_filename: str) -> None:
    """
    Save the information to a CSV file.

    Parameters:
        info_dict_csv (dict): The dictionary containing the information to save.
        csv_filename (str): The name of the CSV file to save the information.
    """

    # The fieldnames are the keys of the dictionary that will be saved in the CSV file
    fieldnames = list(info_dict_csv.keys())
    
    # Create the CSV file if it does not exist
    if not os.path.isfile(csv_filename):
        print("Creating new CSV file")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    # Add the information to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(info_dict_csv)