"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : model_utils.py
* Description       : This file contains functions to get information about models and tokenizers.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with get information functions.
*
******************************************************************"""


from typing import Dict, Any

def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get model information.

    Parameters:
        - model: The loaded model object.

    Returns:
        dict: A dictionary containing model information.
    """
    all_param = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_config = {}
    if hasattr(model, 'config'):
        config = model.config
        model_config['model_type'] = config.model_type

    model_info = {
        "model_name": str(model.config.name_or_path),
        "model_type": str(type(model).__name__),
        "trainable_params": trainable_params,
        "all_params": all_param,
        "trainable_percentage": 100 * trainable_params / all_param if all_param != 0 else 0,
        "model_config": model_config,
        "model_architecture": str(model)  
    }
    return model_info


def get_tokenizer_info(tokenizer: Any) -> Dict[str, Any]:
    """
    Get tokenizer information.

    Parameters:
        - tokenizer: The tokenizer object.

    Returns:
        dict: A dictionary containing all information about the tokenizer.
    """
    tokenizer_info = {
        "tokenizer_name": str(tokenizer.__class__.__name__)
    }
    # Check if each attribute exists and include it in the dictionary if it does
    if hasattr(tokenizer, "vocab_size"):
        tokenizer_info["vocab_size"] = tokenizer.vocab_size
    if hasattr(tokenizer, "model_max_length"):
        tokenizer_info["model_max_length"] = tokenizer.model_max_length
    if hasattr(tokenizer, "is_fast"):
        tokenizer_info["is_fast"] = tokenizer.is_fast
    if hasattr(tokenizer, "padding_side"):
        tokenizer_info["padding_side"] = tokenizer.padding_side
    if hasattr(tokenizer, "truncation_side"):
        tokenizer_info["truncation_side"] = tokenizer.truncation_side
    if hasattr(tokenizer, "special_tokens"):
        tokenizer_info["special_tokens"] = tokenizer.special_tokens
    if hasattr(tokenizer, "clean_up_tokenization_spaces"):
        tokenizer_info["clean_up_tokenization_spaces"] = tokenizer.clean_up_tokenization_spaces

    return tokenizer_info