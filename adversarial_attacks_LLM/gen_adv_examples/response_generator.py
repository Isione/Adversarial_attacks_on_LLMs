"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : response_generator.py
* Description       : This file contains functions to generate responses from a model.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with response generation functions (batch, single, and llama)
*
******************************************************************"""


import torch
from tqdm import tqdm

def generate_response(prompt: str, model, tokenizer) -> str:
  """
  Generate a response to a prompt using a model and tokenizer.
  We assume that a cuda GPU is available for this function.

  Parameters:
    - prompt: str representing formatted prompt
    - model: model object
    - tokenizer: tokenizer object


  Returns:
    - str response of the model
  """

  encoded_input = tokenizer(prompt,  return_tensors="pt")
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(
      **model_inputs,
      max_new_tokens=256,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id
  )

  decoded_output = tokenizer.batch_decode(generated_ids)

  # return only the generated response (not the prompt) as output
  generated_response = decoded_output[0].split("[/INST]")[-1]
  response = generated_response

  return response

def llama_generate_response(prompt: str, model, tokenizer) -> str:
  """
  Generate a response to a prompt using a llama model and tokenizer.
  We assume that a cuda GPU is available for this function.

  Parameters:
    - prompt: str representing formatted prompt
    - model: model object
    - tokenizer: tokenizer object


  Returns:
    - str response of the model
  """

  encoded_input = tokenizer(prompt,  return_tensors="pt")

  model_inputs = encoded_input.to('cuda')

  with torch.no_grad() and torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id, 
        num_return_sequences = 1
    )

    decoded_output = tokenizer.batch_decode(generated_ids)
    
    # return only the generated response (not the prompt) as output
    generated_response = decoded_output[0].split("[/INST]")[-1]
    response = generated_response

  return response

def generate_batch_response(prompts: list, model, tokenizer, max_batch_tokens=4096, max_batch_size=256) -> list:
    """
    Generate responses to a list of prompts using a model and tokenizer.
    We assume that a cuda GPU is available for this function.
    This function generates responses in batches and is optimized for memory usage.


    Parameters:
        - prompts: list of str representing formatted prompts
        - model: model object
        - tokenizer: tokenizer object
        - max_batch_tokens: int representing the maximum number of tokens to generate
        - max_batch_size: int representing the maximum batch size

    Returns:
        - list of str responses of the model

    Warning:
        This function has NOT been tested with a large number of prompts due to CUDA OOM errors.
        It is a work in progress and may require further optimization.
    """

    generated_responses = []

    # Split prompts into batches
    for i in tqdm(range(0, len(prompts), max_batch_size)):
        try: 
          batch_prompts = prompts[i:i + max_batch_size]

          encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)

          accumulated_tokens = 0
          current_batch_size = 0
          batch_responses = []

          for t in encoded_inputs:
              if torch.is_tensor(encoded_inputs[t]):
                  encoded_inputs[t] = encoded_inputs[t].to("cuda")

          # generate response and set desired generation parameters
          with torch.no_grad():
              while accumulated_tokens < max_batch_tokens and current_batch_size < len(batch_prompts):
                  # Adjust batch size based on remaining tokens and available space
                  remaining_tokens = max_batch_tokens - accumulated_tokens
                  tokens_in_current_input = encoded_inputs.input_ids[current_batch_size].numel()

                  if accumulated_tokens + tokens_in_current_input <= max_batch_tokens:
                      accumulated_tokens += tokens_in_current_input
                      current_batch_size += 1
                  else:
                      break

              generated_ids = model.generate(
                  **{key: value[:current_batch_size] for key, value in encoded_inputs.items()},
                  max_new_tokens=1024,
                  do_sample=True,
                  pad_token_id=tokenizer.eos_token_id
              )

              decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
              batch_responses.extend([decoded.split("[/INST]")[-1] for decoded in decoded_outputs])

        except RuntimeError as e:
          if "CUDA out of memory" in str(e):
              # If still out of memory, further reduce batch size
              max_batch_size = max_batch_size // 2
              print(f"CUDA out of memory error: Reducing batch size to {max_batch_size}")
          else:
              raise e  # Re-raise other runtime errors
        generated_responses.extend(batch_responses)
    return generated_responses