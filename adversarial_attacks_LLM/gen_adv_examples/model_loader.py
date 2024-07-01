"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : model_loader.py
* Description       : This file contains functions to load LLM models and tokenizers.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with loading and processing functions.
*
******************************************************************"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_fooled_model_and_tokenizer(model_name: str, device: str, access_token: str) -> tuple:
    """
    Load a causal language model and its corresponding tokenizer with quantization enabled.
    The model is the one to be fooled by the adversarial attacks.

    Parameters:
        - model_name (str): The name or path of the pretrained model.
        - device (torch.device): The device where the model will be placed.
        - access_token (str): The access token for the model.

    Returns:
        tuple: A tuple containing the loaded model and its tokenizer.
    """

    # quantization config
    nf4_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "auto",
        quantization_config = nf4_config,
        use_cache = False
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            padding_side="left",
                                            add_bos_token=True,
                                            add_eos_token=True,
                                            token = access_token, 
                                            padding = "max_length",
                                            )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer

def load_llama(model_name: str, token: str) -> tuple:
  """
  Load a llama2 model and its corresponding tokenizer.

    Parameters:
        - model_name (str): The name or path of the pretrained model.
        - token (str): The token for authentication.

    Returns:
        tuple: A tuple containing the loaded model and its tokenizer.

  """
  
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True,  use_auth_token=token)
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=token)

  return model, tokenizer


def load_classification_model_and_tokenizer(model_name: str, device: str, access_token: str) -> tuple:
    """
    Load a causal language model and its corresponding tokenizer with quantization enabled for classification tasks.
    The model is the one to be asked for sentiment analysis.

    Parameters:
        - model_name (str): The name or path of the pretrained model.
        - device (torch.device): The device where the model will be placed.
        - access_token (str): The access token for the model.

    Returns:
        tuple: A tuple containing the loaded model and its tokenizer.
    """
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        token = access_token
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            padding_side="left",
                                            add_bos_token=True,
                                            add_eos_token=True,
                                            token = access_token, 
                                            padding = "max_length",
                                            )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer
