"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : sentence_modification.py
* Description       : This file contains functions to modify sentences by the LLM (perform the attack) + evaluation functions of modification performances.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with modification functions (single, batch, and llama) + ensemble attack functions + evaluation functions of modification performances.
*
******************************************************************"""

import pandas as pd
from typing import Tuple, List
from response_generator import generate_response, generate_batch_response, llama_generate_response
from evaluate import load
import copy
from sentiment_analysis import is_successful_attack

def llama_create_modification_prompt(prompt:str, fs_example: list, inquiry:str, FS_STRATEGY: str, REWARD_STRATEGY: str, prediction_logits_pos: float, 
                               prediction_logits_neg: float, prediction_confidence_score_pos: float, prediction_confidence_score_neg: float) -> str:
    """
    Create a modification prompt to perform attacks with/without the few-shot and reward strategies for the llama 2 7B model.

    Parameters:
        prompt (str): The initial prompt.
        fs_example (list): The few-shot example.
        inquiry (str): The inquiry to modify.
        FS_STRATEGY (str): Using or not the few-shot strategy.
        REWARD_STRATEGY (str): Using or not the reward strategy.
        prediction_logits_pos (float): The prediction logits for the positive sentiment.
        prediction_logits_neg (float): The prediction logits for the negative sentiment.
        prediction_confidence_score_pos (float): The prediction confidence score for the positive sentiment.
        prediction_confidence_score_neg (float): The prediction confidence score for the negative sentiment.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.


    Returns:
        modification_prompt(str): The modification prompt.    
    """

    if (FS_STRATEGY == "True" and 
        REWARD_STRATEGY == "True"):
        modification_prompt = (
            f"""<s>[INST] <<SYS>>{prompt}
            {add_few_shot_strategy(fs_example)}
            {add_reward_prediction_confidence(prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg, init_sentiment, desired_sentiment)}
            <</SYS>>

            <<< 
            Inquiry: {inquiry}
            >>>
            [/INST]
            """
        )
    
    
    elif FS_STRATEGY == "True" or REWARD_STRATEGY == "True":
        # This case covers all the configurations where at least one strategy is enabled
        few_shot = "" if FS_STRATEGY != "True" else add_few_shot_strategy(fs_example)
        reward_confidence = "" if REWARD_STRATEGY != "True" else add_reward_prediction_confidence(prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg, init_sentiment, desired_sentiment)
        
        modification_prompt = (
            f"""<s>[INST] <<SYS>>{prompt}
            {few_shot}
            {reward_confidence}
            <</SYS>>
            <<< 
            Inquiry: {inquiry}
            >>>
            [/INST]
            """
        )
        
    else:
        modification_prompt = (
            f"""<s>[INST] <<SYS>>{prompt}
            <</SYS>>
            <<<
            Inquiry: {inquiry}
            >>>
            [/INST]
            """
        )   
    return modification_prompt

def create_modification_prompt(prompt:str, fs_example: list, inquiry:str, FS_STRATEGY: str, REWARD_STRATEGY: str, prediction_logits_pos: float, 
                               prediction_logits_neg: float, prediction_confidence_score_pos: float, prediction_confidence_score_neg: float, init_sentiment: str, desired_sentiment: str) -> str:  
    """
    Create a modification prompt to perform attacks with/without the few-shot and reward strategies.

    Parameters:
        prompt (str): The initial prompt.
        fs_example (list): The few-shot example.
        inquiry (str): The inquiry to modify.
        FS_STRATEGY (str): Using or not the few-shot strategy.
        REWARD_STRATEGY (str): Using or not the reward strategy.
        prediction_logits_pos (float): The prediction logits for the positive sentiment.
        prediction_logits_neg (float): The prediction logits for the negative sentiment.
        prediction_confidence_score_pos (float): The prediction confidence score for the positive sentiment.
        prediction_confidence_score_neg (float): The prediction confidence score for the negative sentiment.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.


    Returns:
        modification_prompt(str): The modification prompt.        
    """
    
    if (FS_STRATEGY == "True" and 
        REWARD_STRATEGY == "True"):
        modification_prompt = (
            f"""{prompt}
            {add_few_shot_strategy(fs_example)}
            {add_reward_prediction_confidence(prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg, init_sentiment, desired_sentiment)}
            <<< 
            Inquiry: {inquiry}
            >>>
            """
        )
    
    
    elif FS_STRATEGY == "True" or REWARD_STRATEGY == "True":
        # This case covers all the configurations where at least one strategy is enabled
        few_shot = "" if FS_STRATEGY != "True" else add_few_shot_strategy(fs_example)
        reward_confidence = "" if REWARD_STRATEGY != "True" else add_reward_prediction_confidence(prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg, init_sentiment, desired_sentiment)
        
        modification_prompt = (
            f"""{prompt}
            {few_shot}
            {reward_confidence}
            <<< 
            Inquiry: {inquiry}
            >>>
            """
        )
        
    else:
        modification_prompt = (
            f"""{prompt}
            <<<
            Inquiry: {inquiry}
            >>>
            """
        )   
    return modification_prompt

def add_few_shot_strategy(fs_example: list) -> str:
    """
    Add the few-shot strategy to the modification prompt.

    Parameters:
        fs_example (list): The few-shot example.

    Returns:
        few_shot_example(str): The few-shot example prompt to add to the modification prompt.
    """


    few_shot_example = f"""
    #### Here is 1 example that fits the guidance:

    Original Inquiry: {fs_example[0]} -> Modified Inquiry: {fs_example[1]};

    ###
    """
    return few_shot_example

    

def add_reward_prediction_confidence(prediction_logits_pos: float, prediction_logits_neg:float, prediction_confidence_score_pos: float, prediction_confidence_score_neg: float, init_sentiment: str, desired_sentiment: str) -> str:
    """
    Add the reward prediction confidence to the modification prompt.

    Parameters:
        prediction_logits_pos (float): The prediction logits for the positive sentiment.
        prediction_logits_neg (float): The prediction logits for the negative sentiment.
        prediction_confidence_score_pos (float): The prediction confidence score for the positive sentiment.
        prediction_confidence_score_neg (float): The prediction confidence score for the negative sentiment.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.

    Returns:
        reward_prediction_confidence(str): The reward prediction confidence prompt to add to the modification prompt.
    
    """
    
    reward_prediction_confidence = f"""
    #### 
    The prediction confidence score of the classifier when predicting the positive sentiment of the original inquiry
    is equal to : {prediction_confidence_score_pos}. 
    The prediction confidence score of the classifier when predicting the negative sentiment of the original inquiry
    is equal to : {prediction_confidence_score_neg}.

    Modify the original inquiry after <<<>>> so that it decreases a lot the {init_sentiment} prediction confidence score and increases a lot the {desired_sentiment} prediction confidence score.
    """
    return reward_prediction_confidence


def eval_modif_perf(initial_dataset: pd.DataFrame, final_dataset: pd.DataFrame) -> float:
  """
  Evaluate the performance of the modification by comparing the initial and final datasets.

    Parameters:
        initial_dataset (pd.DataFrame): The initial dataset.
        final_dataset (pd.DataFrame): The final dataset.

    Returns:
        result (float): The percentage of modified sentences.
  """


  total_sentences = len(initial_dataset)

  if total_sentences == 0:
      return 0.0

  modified_sentences = 0
  for sentence_initial_dataset, sentence_final_dataset in zip(initial_dataset['text'], final_dataset['text']):
      if sentence_initial_dataset != sentence_final_dataset:
          modified_sentences += 1

  result = (modified_sentences/total_sentences)*100

  return result



def compute_hword(modif_sentence: str, ref_sentence: str) -> float:
    """
    Compute the number of words that have been modified in the sentence.

    Parameters:
        modif_sentence (str): The modified sentence.
        ref_sentence (str): The reference sentence.

    Returns:
        hwords (float): The number of modified words.
    """

    nb_total_words = len(ref_sentence.split())
    nb_modif_words = 0
    for word in modif_sentence.split():
        if word not in ref_sentence.split():
            nb_modif_words += 1

    hwords = nb_modif_words / nb_total_words
    return hwords

def compute_BERTScore(modif_sentence: str, ref_sentence: str) -> list:
    """
    Compute the BERTScore between the modified sentence and the reference sentence.

    Parameters:
        modif_sentence (str): The modified sentence.
        ref_sentence (str): The reference sentence.

    Returns:
        hbert (list): The BERTScore between the modified sentence and the reference sentence.
        The list contains the BertScores precision, recall, and f1 scores.
    """

    bertscore = load("bertscore")

    hbert = bertscore.compute(predictions=[modif_sentence], references=[ref_sentence], lang="en")

    return hbert



def check_fidelity(modif_sentence: str, ref_sentence: str, bertscore_metric: str, thr_hwords: float, thr_hbert: float) -> Tuple[int, float, float]:
    """
    Check the fidelity of the modification by comparing the modified sentence to the reference sentence.

    Parameters:
        modif_sentence (str): The modified sentence.
        ref_sentence (str): The reference sentence.
        bertscore_metric (str): The metric to use for the BERTScore (precision, recall, or f1).
        thr_hwords (float): The threshold for the number of modified words.
        thr_hbert (float): The threshold for the BERTScore.

    Returns:
        Tuple[int, float, float]: A tuple containing the activation function (tells if the fidelity check is passed or not), the number of modified words, and the BERTScore.

    """

    hbert = compute_BERTScore(modif_sentence, ref_sentence) #'precision', 'recall', 'f1'
    hwords = compute_hword(modif_sentence, ref_sentence)

    if hwords <= thr_hwords and hbert[bertscore_metric][0] >= thr_hbert:
        activation_fct = 1
    else:
        activation_fct = 0

    return activation_fct, hwords, hbert

def batch_ensemble_attack(sentences: pd.DataFrame, model: object, tokenizer: object, perturbation_instructions: list, FS_STRATEGY: str, REWARD_STRATEGY: str, THR_HWORDS: float, 
                          THR_HBERT: float, fs_examples: pd.DataFrame, prediction_logits: list, prediction_confidence_score: list, init_sentiment: str, desired_sentiment: str) -> Tuple[pd.DataFrame, float, dict]:
    """
    Perform a batch ensemble attack on the sentences using the model and tokenizer.
    For each sentence, generate a new inquiry that satisfies the desired sentiment and keeps the semantic meaning unchanged, using 3 word levels perturbation instructions.
    The attack is performed with/without the few-shot and reward strategies.
    The fidelity of the modification is checked using the number of modified words and the BERTScore.
    The modified sentence selected is the one that has the highest BERTScore and successfully fools the model.
    This function generates attacks in batches and is optimized for memory usage.

    Parameters:
        sentences (pd.DataFrame): The sentences to modify.
        model: The model to use for the attack.
        tokenizer: The tokenizer to use for the attack.
        perturbation_instructions (list): The perturbation instructions to use for the attack.
        FS_STRATEGY (str): Using or not the few-shot strategy.
        REWARD_STRATEGY (str): Using or not the reward strategy.
        THR_HWORDS (float): The threshold for the number of modified words.
        THR_HBERT (float): The threshold for the BERTScore.
        fs_examples (pd.DataFrame): The few-shot examples to use for the attack.
        prediction_logits (list): The prediction logits of the model.
        prediction_confidence_score (list): The prediction confidence score of the model.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.

    Returns:
        modified_df (pd.DataFrame): The modified sentences.
        hwords (float): The number of modified words.
        hbert (dict): The BERTScore of the modified sentences.

    Warning:
        This function has NOT been tested with a large number of sentences due to CUDA OOM errors.
        It is a work in progress and may require further optimization.   
    """
    

    batch_size = len(sentences)
    modified_df = pd.DataFrame(index=sentences.index)  # Preserve the original indexes
    hwords = 0
    hbert = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    check_fidelity_mode = "True"
    
    iteration = 0
    
    num_batches = (len(sentences) + batch_size - 1) // batch_size

    generated_prompts = []

    for i in range(len(perturbation_instructions)):
        perturbation_instruction = perturbation_instructions[i]
        if FS_STRATEGY == "True":
            few_shot_examples = fs_examples['fs_example'][i][0]
        else:
            few_shot_examples = []

        modif_prompt = f"The original inquiry after <<<>>> is classified as {init_sentiment} when performing sentiment analysis. \
        Your task is to generate a new inquiry which must satisfy the following conditions: \
        - The sentiment of the new inquiry must be classified as {desired_sentiment} when performing sentiment analysis. \
        - Keeping the semantic meaning of the new inquiry unchanged. \
        You can finish the task by modifying the original inquiry after <<<>>> using the following guidance: \
        - {perturbation_instruction} \
        - Only output the new inquiry without anything else. \
        - Never ever output the original inquiry or notes or explanations."

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sentences))
          
            batch_neg_sentences = [row["text"].lower() for idx, row in sentences.iloc[start_idx:end_idx].iterrows()]
            batch_labels = [row["predicted_labels"] for idx, row in sentences.iloc[start_idx:end_idx].iterrows()]
            batch_indices = sentences.iloc[start_idx:end_idx].index

            prediction_logits_pos, prediction_logits_neg = prediction_logits[1][iteration], prediction_logits[0][iteration]
            prediction_confidence_score_pos = prediction_confidence_score[1][iteration]
            prediction_confidence_score_neg = prediction_confidence_score[0][iteration]

            for idx, neg_sentence in enumerate(batch_neg_sentences):
                generated_prompts.append(create_modification_prompt(modif_prompt, few_shot_examples, neg_sentence, FS_STRATEGY, REWARD_STRATEGY, prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg))

            answers = generate_batch_response(generated_prompts, model, tokenizer, max_batch_size=batch_size)
            answers = [a.lower().split(">>>")[-1].split("inquiry:")[-1].split("</s>")[0].strip() for a in answers]

            if check_fidelity_mode == "True":
                thr_hwords = 1.0 if i > 7 else float(THR_HWORDS)
                thr_hbert = float(THR_HBERT)

                fidelities = []
                hwords_list = []
                hbert_list = []

                for answer, neg_sentence in zip(answers, batch_neg_sentences):
                    fidelity, hwords, hbert = check_fidelity(answer, neg_sentence, "f1", thr_hwords, thr_hbert)
                    fidelities.append(fidelity)
                    hwords_list.append(hwords)
                    hbert_list.append(hbert)

                modif_answers = [neg_sentence if fidelity == 0 else answer for answer, fidelity, neg_sentence in zip(answers, fidelities, batch_neg_sentences)]
            else:
                hbert_list = [compute_BERTScore(answer, neg_sentence) for answer, neg_sentence in zip(answers, batch_neg_sentences)]
                hwords_list = [compute_hword(answer, neg_sentence) for answer, neg_sentence in zip(answers, batch_neg_sentences)]
                modif_answers = answers

            tmp_answers = modif_answers.copy()
            tmp_hberts = copy.deepcopy(hbert_list)

            for idx, (modif_answer, label, tmp_hbert) in enumerate(zip(tmp_answers, batch_neg_sentences, tmp_hberts)):
                if is_successful_attack(modif_answer, label, tokenizer, model) and tmp_hbert['f1'][0] >= hbert['f1'][0]:
                    modif_answers[idx] = tmp_answers[idx]
                    hbert = tmp_hbert
                else:
                    pass

            for idx, modif_answer in enumerate(modif_answers):
                modified_df.at[batch_indices[idx], "text"] = modif_answer  # Use the original index for the modified answer
            iteration += len(batch_neg_sentences)

        iteration = 0    

    return modified_df, hwords, hbert

def ensemble_attack(sentences: pd.DataFrame,  model: object, tokenizer: object, perturbation_instructions: list, FS_STRATEGY: str, REWARD_STRATEGY: str, THR_HWORDS: float, 
                    THR_HBERT: float, fs_examples: pd.DataFrame, prediction_logits: list, prediction_confidence_score: list, init_sentiment: str, desired_sentiment: str) -> Tuple[pd.DataFrame, float, dict]:
    """
    Perform an ensemble attack on the sentences using the model and tokenizer.
    For each sentence, generate a new inquiry that satisfies the desired sentiment and keeps the semantic meaning unchanged, using 3 word levels perturbation instructions. 
    The attack is performed with/without the few-shot and reward strategies.
    The fidelity of the modification is checked using the number of modified words and the BERTScore.
    The modified sentence selected is the one that has the highest BERTScore and successfully fools the model.

    Parameters:
        sentences (pd.DataFrame): The sentences to modify.
        model: The model to use for the attack.
        tokenizer: The tokenizer to use for the attack.
        perturbation_instructions (list): The perturbation instructions to use for the attack.
        FS_STRATEGY (str): Using or not the few-shot strategy.
        REWARD_STRATEGY (str): Using or not the reward strategy.
        THR_HWORDS (float): The threshold for the number of modified words.
        THR_HBERT (float): The threshold for the BERTScore.
        fs_examples (pd.DataFrame): The few-shot examples to use for the attack.
        prediction_logits (list): The prediction logits of the model.
        prediction_confidence_score (list): The prediction confidence score of the model.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.

    Returns:
        modified_df (pd.DataFrame): The modified sentences.
        hwords (float): The number of modified words.
        hbert (dict): The BERTScore of the modified sentences.
    """
    
    
    modified_df = pd.DataFrame()
    hwords = 0
    hbert = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    check_fidelity_mode = "True"
    
    iteration = 0

    
    for i in range(len(perturbation_instructions)):
        perturbation_instruction = perturbation_instructions[i]
        if FS_STRATEGY == "True":
            few_shot_examples = fs_examples['fs_example'][i][0]
        else:
            few_shot_examples = []


        modif_prompt = f"The original inquiry after <<<>>> is classified as {init_sentiment} when performing sentiment analysis. \
        Your task is to generate a new inquiry which must satisfy the following conditions: \
        - The sentiment of the new inquiry must be classified as {desired_sentiment} when performing sentiment analysis. \
        - Keeping the semantic meaning of the new inquiry unchanged. \
        You can finish the task by modifying the original inquiry after <<<>>> using the following guidance: \
        - {perturbation_instruction} \
        - Only output the new inquiry without anything else.\
        - Output format is : new inquiry: <modified inquiry> "
        for i, (idx, row) in enumerate(sentences.iterrows()):
            sentence = row["text"].lower()
            label = row['predicted_labels']
          
            prediction_logits_pos = prediction_logits[i][1]
            prediction_logits_neg = prediction_logits[i][0]
            prediction_confidence_score_pos = prediction_confidence_score[i][1]
            prediction_confidence_score_neg = prediction_confidence_score[i][0]
        
            generated_prompt = create_modification_prompt(modif_prompt, few_shot_examples, sentence, FS_STRATEGY, REWARD_STRATEGY, prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg, init_sentiment, desired_sentiment)

            answer = generate_response(generated_prompt, model, tokenizer)
            answer = answer.lower()
            answer = answer.split(">>>")[-1].split("inquiry:")[-1].split("</s>")[0].strip()
       
            if "explanation" in answer:
                answer = answer.split("explanation")[0].strip()
            
            if "note" in answer:
                answer = answer.split("note")[0].strip()

            if "justification:" in answer:
                answer = answer.split("justification:")[0].strip()

            if "reason:" in answer:
                answer = answer.split("reason:")[0].strip()
                
            if check_fidelity_mode == "True":
                if i > 7:
                    thr_hwords = 1.0
                    thr_hbert = float(THR_HBERT)
                    fidelity, hwords, hbert = check_fidelity(answer, sentence, "f1", thr_hwords, thr_hbert)
                else:
                    thr_hwords = float(THR_HWORDS)
                    thr_hbert = float(THR_HBERT)
                    fidelity, hwords, hbert = check_fidelity(answer, sentence, "f1", thr_hwords, thr_hbert)

                if fidelity == 0:
                    modif_answer = sentence
                else:
                    modif_answer = answer

            else:
                hbert = compute_BERTScore(answer, sentence)  # 'precision', 'recall', 'f1'
                hwords = compute_hword(answer, sentence)
                modif_answer = answer
            
            tmp_answer = copy.deepcopy(modif_answer)
            tmp_hbert = copy.deepcopy(hbert)
            if is_successful_attack(tmp_answer, label, tokenizer, model) and tmp_hbert["f1"][0] >= hbert["f1"][0]: 
                modif_answer = tmp_answer
                hbert = tmp_hbert

            modified_df.at[idx, "text"] = modif_answer  
            iteration += 1

        iteration = 0    

    return modified_df, hwords, hbert
            



def llama_ensemble_attack(sentences: pd.DataFrame,  model: object, tokenizer: object, perturbation_instructions: list, FS_STRATEGY: str, REWARD_STRATEGY: str, THR_HWORDS: float, 
                          THR_HBERT: float, fs_examples: pd.DataFrame, prediction_logits: list, prediction_confidence_score: list, init_sentiment: str, desired_sentiment: str) -> Tuple[pd.DataFrame, float, dict]:
    """
    Perform an ensemble attack on the sentences for the llama2 7B model.
    For each sentence, generate a new inquiry that satisfies the desired sentiment and keeps the semantic meaning unchanged, using 3 word levels perturbation instructions. 
    The attack is performed with/without the few-shot and reward strategies.
    The fidelity of the modification is checked using the number of modified words and the BERTScore.
    The modified sentence selected is the one that has the highest BERTScore and successfully fools the model.

    Parameters:
        sentences (pd.DataFrame): The sentences to modify.
        model: The model to use for the attack.
        tokenizer: The tokenizer to use for the attack.
        perturbation_instructions (list): The perturbation instructions to use for the attack.
        FS_STRATEGY (str): Using or not the few-shot strategy.
        REWARD_STRATEGY (str): Using or not the reward strategy.
        THR_HWORDS (float): The threshold for the number of modified words.
        THR_HBERT (float): The threshold for the BERTScore.
        fs_examples (pd.DataFrame): The few-shot examples to use for the attack.
        prediction_logits (list): The prediction logits of the model.
        prediction_confidence_score (list): The prediction confidence score of the model.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.

    Returns:
        modified_df (pd.DataFrame): The modified sentences.
        hwords (float): The number of modified words.
        hbert (dict): The BERTScore of the modified sentences.
         
    """
    
    
    modified_df = pd.DataFrame()
    hwords = 0
    hbert = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    check_fidelity_mode = "True"
    
    iteration = 0

    for i in range(len(perturbation_instructions)):
        perturbation_instruction = perturbation_instructions[i]
        if FS_STRATEGY == "True":
            few_shot_examples = fs_examples['fs_example'][i][0]
        else:
            few_shot_examples = []


        modif_prompt = f"The original inquiry after <<<>>> is classified as {init_sentiment} when performing sentiment analysis. \
        Your task is to generate a new inquiry which must satisfy the following conditions: \
        - The sentiment of the new inquiry must be classified as {desired_sentiment} when performing sentiment analysis. \
        - Keeping the semantic meaning of the new inquiry unchanged. \
        You can finish the task by modifying the original inquiry after <<<>>> using the following guidance: \
        - {perturbation_instruction} \
        - Only output the new inquiry without anything else.\
        - Output format is : new inquiry: <modified inquiry> "
        for i, (idx, row) in enumerate(sentences.iterrows()):
            sentence = row["text"].lower()
            label = row['predicted_labels']
            # prediction_logits_pos, prediction_logits_neg = prediction_logits[1][iteration], predictio<n_logits[0][iteration]
          
            prediction_logits_pos = prediction_logits[i][1]
            prediction_logits_neg = prediction_logits[i][0]
            prediction_confidence_score_pos = prediction_confidence_score[i][1]
            prediction_confidence_score_neg = prediction_confidence_score[i][0]
        
            generated_prompt = llama_create_modification_prompt(modif_prompt, few_shot_examples, sentence, FS_STRATEGY, REWARD_STRATEGY, prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg)

            answer = generate_response(generated_prompt, model, tokenizer)
            answer = answer.lower()
            answer = answer.split(">>>")[-1].split("inquiry:")[-1].split("</s>")[0].strip()
         
            if "explanation" in answer:
                answer = answer.split("explanation")[0].strip()
            
            if "note" in answer:
                answer = answer.split("note")[0].strip()

            if "justification:" in answer:
                answer = answer.split("justification:")[0].strip()

            if "reason:" in answer:
                answer = answer.split("reason:")[0].strip()
                
            
            
            if check_fidelity_mode == "True":
                if i > 7:
                    thr_hwords = 1.0
                    thr_hbert = float(THR_HBERT)
                    fidelity, hwords, hbert = check_fidelity(answer, sentence, "f1", thr_hwords, thr_hbert)
                else:
                    thr_hwords = float(THR_HWORDS)
                    thr_hbert = float(THR_HBERT)
                    fidelity, hwords, hbert = check_fidelity(answer, sentence, "f1", thr_hwords, thr_hbert)

                if fidelity == 0:
                    modif_answer = sentence
                else:
                    modif_answer = answer

            else:
                hbert = compute_BERTScore(answer, sentence)  # 'precision', 'recall', 'f1'
                hwords = compute_hword(answer, sentence)
                modif_answer = answer
            
            tmp_answer = copy.deepcopy(modif_answer)
            tmp_hbert = copy.deepcopy(hbert)
            if is_successful_attack(tmp_answer, label, tokenizer, model) and tmp_hbert["f1"][0] >= hbert["f1"][0]: 
                modif_answer = tmp_answer
                hbert = tmp_hbert

            modified_df.at[idx, "text"] = modif_answer  
            iteration += 1

        iteration = 0    

    return modified_df, hwords, hbert

def modify_sentences(dataset: pd.DataFrame, model: object, tokenizer: object, perturbation_instructions: list, FS_STRATEGY: str, REWARD_STRATEGY: str, THR_HWORDS: float, 
                     THR_HBERT: float, PROMPTING_GUIDANCE: str, fs_examples: pd.DataFrame, prediction_logits: list, prediction_confidence_score: list, init_sentiment: str, desired_sentiment: str) -> Tuple[pd.DataFrame, float, List[float]] :
    """
    Modify the sentences using the model and tokenizer and perform the attack.

    Parameters:
        dataset (pd.DataFrame): The dataset containing the sentences to modify.
        model: The model to use for the attack.
        tokenizer: The tokenizer to use for the attack.
        perturbation_instructions (list): The perturbation instructions to use for the attack.
        FS_STRATEGY (str): Using or not the few-shot strategy.
        REWARD_STRATEGY (str): Using or not the reward strategy.
        THR_HWORDS (float): The threshold for the number of modified words.
        THR_HBERT (float): The threshold for the BERTScore.
        PROMPTING_GUIDANCE (str): The prompting guidance to use for the attack.
        fs_examples (pd.DataFrame): The few-shot examples to use for the attack.
        prediction_logits (list): The prediction logits of the model.
        prediction_confidence_score (list): The prediction confidence score of the model.
        init_sentiment (str): The initial sentiment of the inquiry.
        desired_sentiment (str): The desired sentiment of the inquiry.

    Returns:
        modified_df (pd.DataFrame): The modified sentences.
        hwords (float): The number of modified words.
        hbert (list): The BERTScore of the modified sentences.
    """
    
    
    modified_df = pd.DataFrame()
    hwords = 0
    hbert = []
    check_fidelity_mode = "True"
    iteration = 0

    if len(perturbation_instructions) >= 9:
        if PROMPTING_GUIDANCE == "OI_AO_AG_W1_prompt":
            perturbation_instruction = perturbation_instructions[4]
            if FS_STRATEGY == "True":
                few_shot_examples = fs_examples['fs_example'][4][0]
            else:
                few_shot_examples = []
        elif PROMPTING_GUIDANCE == "OI_AO_AG_W2_prompt":
            perturbation_instruction = perturbation_instructions[5]
            if FS_STRATEGY == "True":
                few_shot_examples = fs_examples['fs_example'][5][0]
            else:
                few_shot_examples = []
        elif PROMPTING_GUIDANCE == "OI_AO_AG_W3_prompt":
            perturbation_instruction = perturbation_instructions[6]
            if FS_STRATEGY == "True":
                few_shot_examples = fs_examples['fs_example'][6][0]
            else:
                few_shot_examples = []
        else:
            perturbation_instruction = ""

    else: #using only the 3 word perturbation instructions
        if PROMPTING_GUIDANCE == "OI_AO_AG_W1_prompt":
            perturbation_instruction = perturbation_instructions[0]
            if FS_STRATEGY == "True":
                few_shot_examples = fs_examples['fs_example'][0][0]
            else:
                few_shot_examples = []
        elif PROMPTING_GUIDANCE == "OI_AO_AG_W2_prompt":
            perturbation_instruction = perturbation_instructions[1]
            if FS_STRATEGY == "True":
                few_shot_examples = fs_examples['fs_example'][1][0]
            else:
                few_shot_examples = []
        elif PROMPTING_GUIDANCE == "OI_AO_AG_W3_prompt":
            perturbation_instruction = perturbation_instructions[2]
            if FS_STRATEGY == "True":
                few_shot_examples = fs_examples['fs_example'][2][0]
            else:
                few_shot_examples = []
        else:
            perturbation_instruction = ""
    
    modif_prompt = f"The original inquiry after <<<>>> is classified as {init_sentiment} when performing sentiment analysis. \
    Your task is to generate a new inquiry which must satisfy the following conditions: \
    - The sentiment of the new inquiry must be classified as {desired_sentiment} when performing sentiment analysis. \
    - Keeping the semantic meaning of the new inquiry unchanged. \
    You can finish the task by modifying the original inquiry after <<<>>> using the following guidance: \
    - {perturbation_instruction} \
    - Only output the new inquiry without anything else.\
    - Output format is : new inquiry: <modified inquiry> "
    
    for i, (idx, row) in enumerate(dataset.iterrows()):
        sentence = row["text"].lower()

        prediction_logits_pos = prediction_logits[i][1]
        prediction_logits_neg = prediction_logits[i][0]
        prediction_confidence_score_pos = prediction_confidence_score[i][1]
        prediction_confidence_score_neg = prediction_confidence_score[i][0]

        generated_prompt = create_modification_prompt(modif_prompt,few_shot_examples, sentence, FS_STRATEGY, REWARD_STRATEGY, prediction_logits_pos, prediction_logits_neg, prediction_confidence_score_pos, prediction_confidence_score_neg, init_sentiment, desired_sentiment)

        answer = generate_response(generated_prompt, model, tokenizer)
        answer = answer.lower()
        answer = answer.split(">>>")[-1].split("inquiry:")[-1].split("</s>")[0].strip()
        

        if "explanation" in answer:
            answer = answer.split("explanation")[0].strip()
        
        if "note" in answer:
            answer = answer.split("note")[0].strip()

        if "justification:" in answer:
            answer = answer.split("justification:")[0].strip()

        if "reason:" in answer:
            answer = answer.split("reason:")[0].strip()
        
        if check_fidelity_mode == "True":
       
            fidelity, hwords, hbert = check_fidelity(answer, sentence, "f1", THR_HWORDS, THR_HBERT)

            if fidelity == 0:
                modif_answer = sentence
            else:
                modif_answer = answer

        else:
            hbert = compute_BERTScore(answer, sentence)  # 'precision', 'recall', 'f1'
            hwords = compute_hword(answer, sentence)
            modif_answer = answer

        modified_df.at[idx, "text"] = modif_answer
        iteration += 1

    iteration = 0
    return modified_df, hwords, hbert

