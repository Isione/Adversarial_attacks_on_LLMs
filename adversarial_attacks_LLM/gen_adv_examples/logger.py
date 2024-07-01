"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : logger.py
* Description       : This file contains functions to set up a logger.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with setup_logger function info and debug(filename, line number, and colorlog).
*
******************************************************************"""

import logging
import logging.config
import sys
import colorlog
import datetime
import os

def setup_logger(name, logs_folder=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    stdout = colorlog.StreamHandler(stream=sys.stdout)
    # fmt = colorlog.ColoredFormatter("%(name)s: %(asctime)s | %(filename)s | %(levelname)s | %(message)s")
    # include line number in the log output
    fmt = colorlog.ColoredFormatter("%(name)s: %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    
    stdout.setFormatter(fmt)
    logger.addHandler(stdout)  

    if logs_folder:
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_datetime}_{name}.log"
        file_handler = logging.FileHandler(os.path.join(logs_folder, filename))
        file_handler.setFormatter(logging.Formatter("%(name)s: %(asctime)s | %(filename)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
    
    return logger