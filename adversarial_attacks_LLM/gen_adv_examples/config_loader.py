"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : config_loader.py
* Description       : This file contains functions to load configuration from an INI file.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with load_config function.
*
******************************************************************"""

from typing import Dict, Any
import configparser


def load_config(filename: str) -> Dict[str, Dict[str, Any]]:
    """
    Load configuration from an INI file.

    Parameters:
        filename (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing configuration data.
              The keys are section names, and the values are dictionaries
              mapping each option name to its corresponding value.
    """
    config = configparser.ConfigParser()
    config.read(filename)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option, value in config.items(section):
            config_dict[section][option] = value

    return config_dict