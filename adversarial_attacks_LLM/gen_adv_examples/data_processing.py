"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : data_processing.py
* Description       : This file contains functions to process input data.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with loading and processing functions.
*
******************************************************************"""

import pandas as pd
from typing import Tuple
import string
import pickle

def load_sst2_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load sst2 dataset from a .parquet file saved in a local path and preprocess it by 
    - removing punctuation, 
    - splitting text into words, 
    - selecting only sentences with more than 7 words.

    The remote sst2 dataset is available at: https://huggingface.co/datasets/stanfordnlp/sst2

    Parameters:
        dataset_path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed dataset.

    """
    df = pd.read_parquet(dataset_path)
    df = df[["sentence", "label"]]
    df = df.rename(columns={"sentence": "text", "label": "sentiment"})
    # Remove punctuation from text
    df['text_no_punct'] = df['text'].apply(remove_punctuation)

    # Split text into words
    df['words'] = df['text_no_punct'].apply(lambda x: x.split())

    # Select only sentences with more than 7 words
    df = df[df['words'].apply(lambda x: len(x) >= 7)]
    
    # Drop intermediate columns
    df = df.drop(columns=['text_no_punct', 'words'])

    return df

def select_negative_sentences(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Select the negative sentences (with label 0) from the dataset.

    Parameters:
        dataset (pd.DataFrame): The dataset containing the sentences.

    Returns:
        Tuple[pd.DataFrame, pd.Index]: A tuple containing the negative sentences and their indexes.

    """
    negative_dataset = dataset[dataset['predicted_labels'] == 0] 
    idx_negative_dataset = negative_dataset.index

    return negative_dataset, idx_negative_dataset

def select_positive_sentences(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Select the positive sentences (with label 1) from the dataset.

    Parameters:
        dataset (pd.DataFrame): The dataset containing the sentences.

    Returns:
        Tuple[pd.DataFrame, pd.Index]: A tuple containing the positive sentences and their indexes.

    """
    positive_dataset = dataset[dataset['predicted_labels'] == 1] 
    idx_positive_dataset = positive_dataset.index

    return positive_dataset, idx_positive_dataset


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from a text.

    Parameters:
        text (str): The text to process.

    Returns:
        str: The text without punctuation.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def load_fs_examples(path: str) -> pd.DataFrame:
    """
    Load the few-shot examples from a local pickle file.

    The pickle file is downloaded from the following link: https://github.com/GodXuxilie/PromptAttack/blob/main/info/sst2_info.pkl

    Parameters:
        path (str): The path to the pickle file.

    Returns:
        pd.DataFrame: A DataFrame containing the few-shot examples.
    
    """
    with open(path, "rb") as f: 
        fs_examples = pickle.load(f)

    return fs_examples