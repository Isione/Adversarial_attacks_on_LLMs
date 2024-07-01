"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : sentence_analysis.py
* Description       : This file contains functions to analyze/classify sentences and evaluate attacks.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with classification and evaluation functions.
*
******************************************************************"""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from tqdm import tqdm
import numpy as np
from typing import Union

def classify_in_batches(dataset: pd.DataFrame, text_column: str, model, tokenizer) -> Union[list, list]:
    """ 
    Classify the sentences in the dataset as positive or negative sentiment using the a LL model and tokenizer provided.
    This function classifies sentences in batches and is optimized for memory usage.

    Parameters:
        dataset (pd.DataFrame): The dataset containing the sentences to classify.
        text_column (str): The name of the column containing the sentences.
        model: The LL model to use for classification.
        tokenizer: The tokenizer to use for classification.

    Returns:
        (predicted_class, pred_logits, pred_confidence_score): A tuple containing the predicted class, the logits, and the confidence scores.
    
    Warning:
        This function has NOT been tested with a large number of sentences due to CUDA OOM errors.
        It is a work in progress and may require further optimization.  
    """

    batch_size = 256  

    predicted_class = []
    pred_logits = []
    pred_confidence_score = []

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    neg_preds = []
    pos_preds = []
    neg_probas = []
    pos_probas = []
    predicted_classes = []

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(dataset))


        prompts = [row[text_column][0] for _, row in dataset.iloc[start_idx:end_idx].iterrows()]

        inputs_modify = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        for t in inputs_modify:
            if torch.is_tensor(inputs_modify[t]):
                inputs_modify[t] = inputs_modify[t].to("cuda")


        with torch.no_grad():
            generate_ids_modify = model.generate(**inputs_modify, num_return_sequences=1, output_scores=True, 
                                                return_dict_in_generate=True, max_new_tokens=1024)

            tokenizer_batch_decode = tokenizer.batch_decode(generate_ids_modify.sequences, skip_special_tokens=True)


            for idx in range(len(prompts)):
                neg_preds.append(float(generate_ids_modify.scores[0][idx][7087].item()))  # negative
                pos_preds.append(float(generate_ids_modify.scores[0][idx][5278].item()))  # positive

                neg_probas.append(torch.nn.functional.softmax(torch.tensor([neg_preds[idx], pos_preds[idx]]), dim=0)[0].item())
                pos_probas.append(torch.nn.functional.softmax(torch.tensor([neg_preds[idx], pos_preds[idx]]), dim=0)[1].item())

                predicted_classes.append(torch.argmax(torch.tensor([neg_preds[idx], pos_preds[idx]])).item())


    pred_logits.extend([neg_preds, pos_preds])
    predicted_class.extend(predicted_classes)
    pred_confidence_score.extend([neg_probas, pos_probas])

    return predicted_class, pred_logits, pred_confidence_score


def classify(dataset: pd.DataFrame, text_column: str, model, tokenizer) -> Union[list, list]:
    """ 
    Classify the sentences in the dataset as positive or negative sentiment using the a LL model and tokenizer provided.

    Parameters:
        dataset (pd.DataFrame): The dataset containing the sentences to classify.
        text_column (str): The name of the column containing the sentences.
        model: The LL model to use for classification.
        tokenizer: The tokenizer to use for classification.

    Returns:
        (predicted_class, pred_logits, pred_confidence_score): A tuple containing the predicted class, the logits, and the confidence scores.
    
    """
    predicted_class = []
    pred_logits = []
    pred_confidence_score = []
    for i in tqdm(range(len(dataset))):
        prompt = dataset.iloc[i][text_column][0]
        chat_modify = [{"role": "user", "content": prompt}]

        inputs_modify = tokenizer.apply_chat_template(chat_modify, tokenize=True, return_tensors="pt").to("cuda")
        with torch.no_grad() and torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            generate_ids_modify = model.generate(input_ids = inputs_modify, 
                                                 num_return_sequences = 1, output_scores=True, 
                                                 return_dict_in_generate=True, max_new_tokens=1024)
            tokenizer.decode(generate_ids_modify.sequences[0], skip_special_tokens=True)
            neg_pred = float(generate_ids_modify.scores[0][0][7087].item()) # negative
            pos_pred = float(generate_ids_modify.scores[0][0][5278].item()) # positive
            
            neg_proba = torch.nn.functional.softmax(torch.tensor([neg_pred, pos_pred]), dim=0)[0].item()
            pos_proba = torch.nn.functional.softmax(torch.tensor([neg_pred, pos_pred]), dim=0)[1].item()
            pred_logits.append([neg_pred, pos_pred])
            predicted_class.append(torch.argmax(torch.tensor([neg_pred, pos_pred])).item())
            pred_confidence_score.append([neg_proba, pos_proba])
    
    return predicted_class, pred_logits, pred_confidence_score

def is_successful_attack(pred_ans: str, label: int, tokenizer, model) -> bool:
    """
    Check if the attack is successful by comparing the predicted class of the original sentence and the adversarial sentence.
    Note: we assume that the adversarial sentence has passed the fidelity filter and is a valid sentence.

    Parameters:
        pred_ans (str): The original sentence to attack.
        label (int): The label of the original sentence.
        tokenizer: The tokenizer to use for classification.
        model: The LL model to use for classification.

    Returns:
        bool: True if the attack is successful, False otherwise.

    """

    prompt = generate_ens_classification_prompt(pred_ans)
    chat_modify = [{"role": "user", "content": prompt}]
    inputs_modify = tokenizer.apply_chat_template(chat_modify, tokenize=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generate_ids_modify = model.generate(input_ids = inputs_modify, 
                                            num_return_sequences = 1, output_scores=True, 
                                            return_dict_in_generate=True, max_new_tokens=1024)
        tokenizer.decode(generate_ids_modify.sequences[0], skip_special_tokens=True)
        neg_pred = float(generate_ids_modify.scores[0][0][7087].item()) # negative
        pos_pred = float(generate_ids_modify.scores[0][0][5278].item()) # positive
        predicted_class = torch.argmax(torch.tensor([neg_pred, pos_pred])).item()
    
    if predicted_class != label: 
        return True
    else:
        return False

def evaluate_classifier(y_true: list, y_pred: list) -> dict:
    """
    Evaluate the classifier (which is a LLM) using the true labels and the predicted labels.

    Parameters:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        (accuracy, class_report, conf_matrix, precision, recall, f1): A dict containing the accuracy, classification report, confusion matrix, precision, recall, and f1 score.
    
    """


    mapping = {'positive': 1, 'negative': 0, 'none': 2}
    def map_func(x):
        return mapping.get(x, 1)

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)

    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped))
                        if y_true_mapped[i] == label]
        label_y_true_mapped = [y_true_mapped[i] for i in label_indices]
        label_y_pred_mapped = [y_pred_mapped[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true_mapped, label_y_pred_mapped)

    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped)

    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=[0, 1, 2])

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_mapped, y_pred_mapped, average = 'weighted')
    return {"accuracy": accuracy, "classification_report": class_report, "confusion_matrix": conf_matrix, "precision": precision, "recall": recall, "f1": f1}

def evaluate_initial_classifier(y_true: list, y_pred: list) -> dict:
    """
    Evaluate the classifier (which is a LLM) using the true labels and the predicted labels.

    Parameters:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        (accuracy, class_report, conf_matrix, precision, recall, f1): A dict containing the accuracy, classification report, confusion matrix, precision, recall, and f1 score.
    
    """

    labels = ['positive', 'negative', 'none']
    mapping = {'positive': 1, 'negative': 0, 'none': 2}
    def map_func(x):
        return mapping.get(x, 1)

    # y_pred_mapped = np.vectorize(map_func)(y_pred)
    y_pred_mapped = y_pred

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_mapped)

    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                        if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred_mapped = [y_pred_mapped[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred_mapped)

    class_report = classification_report(y_true=y_true, y_pred=y_pred_mapped)

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_mapped, labels=[0, 1, 2])

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_mapped, average = 'weighted')
    return {"accuracy": accuracy, "classification_report": class_report, "confusion_matrix": conf_matrix, "precision": precision, "recall": recall, "f1": f1}

def attack_success_rate(ref_dataset: pd.DataFrame, predicted_dataset: pd.DataFrame) -> float:
    """
    Calculate the attack success rate given the reference dataset and the predicted dataset.

    Parameters:
        ref_dataset (pd.DataFrame): The reference dataset containing the true labels.
        predicted_dataset (pd.DataFrame): The predicted dataset containing the predicted labels.

    Returns:
        ASR (float): The attack success rate.
    
    """

    assert len(ref_dataset) == len(predicted_dataset)
    ref_labels = ref_dataset['predicted_labels']
    predicted_labels = predicted_dataset['predicted_labels']

    success_rate = (ref_labels != predicted_labels).sum() / len(ref_labels)

    return success_rate*100


def generate_classification_prompt(data_point: dict) -> str:
    """
    Generate a classification prompt for LLM, to classify the sentiment of the data point.

    Parameters:
        data_point (dict): The data point containing the text to classify.

    Returns:
        str: The classification prompt for the LLM.
    """

    return f"""
            [INST]Analyze the sentiment of the Movie Review enclosed in square brackets,
            determine if it is positive, or negative, and return the answer as
            the corresponding sentiment label "positive" if it is positive, or "negative" if
            it is negative. Only output the corresponding sentiment label. Never ever answer
            anything else.[/INST]

            [{data_point["text"]}] = """.strip()

def generate_ens_classification_prompt(sentence: str):
    """
    Generate a classification prompt for LLM, to classify the sentiment the sentence (when performing an ensemble attack)

    Parameters:
        sentence (str): The sentence to be classified.

    Returns:
        str: The classification prompt for the LLM.
    """
    return f"""
            [INST]Analyze the sentiment of the Movie Review enclosed in square brackets,
            determine if it is positive, or negative, and return the answer as
            the corresponding sentiment label "positive" if it is positive, or "negative" if
            it is negative. Only output the corresponding sentiment label. Never ever answer
            anything else.[/INST]

            [{sentence}] = """.strip()

def llama_classify(dataset: pd.DataFrame, text_column: str, model, tokenizer) -> Union[list, list, list]:
  """ 
    Classify the sentences in the dataset as positive or negative sentiment using the Llama2 7B model and tokenizer provided.

    Parameters:
        dataset (pd.DataFrame): The dataset containing the sentences to classify.
        text_column (str): The name of the column containing the sentences.
        model: The LL model to use for classification.
        tokenizer: The tokenizer to use for classification.

    Returns:
        (predicted_class, pred_logits, pred_confidence_score): A tuple containing the predicted class, the logits, and the confidence scores.
    
    """

  predicted_class = []
  pred_logits = []
  pred_confidence_score = []
  for i in tqdm(range(len(dataset))):
      prompt = dataset.iloc[i]["classification_prompts"][0]

      # chat_modify = [{"role": "user", "content": prompt}]
      # chat_modify = [{"role": "system", "content": prompt}]

      # inputs_modify = tokenizer.apply_chat_template(chat_modify, tokenize=True, return_tensors="pt").to("cuda")
      inputs_modify = tokenizer(prompt, return_tensors="pt").to("cuda")
      

      with torch.no_grad() and torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
          generate_ids_modify = model.generate(**inputs_modify,
                                                num_return_sequences = 1, output_scores=True,
                                                return_dict_in_generate=True, max_new_tokens=250)
          
          tokenizer.decode(generate_ids_modify.sequences[0], skip_special_tokens=True)
          neg_pred = float(generate_ids_modify.scores[0][0][12610].item()) # negative
          pos_pred = float(generate_ids_modify.scores[0][0][10321].item()) # positive


          neg_proba = torch.nn.functional.softmax(torch.tensor([neg_pred, pos_pred]), dim=0)[0].item()
          pos_proba = torch.nn.functional.softmax(torch.tensor([neg_pred, pos_pred]), dim=0)[1].item()
          pred_logits.append([neg_pred, pos_pred])
          predicted_class.append(torch.argmax(torch.tensor([neg_pred, pos_pred])).item())
          pred_confidence_score.append([neg_proba, pos_proba])
  return predicted_class, pred_logits, pred_confidence_score