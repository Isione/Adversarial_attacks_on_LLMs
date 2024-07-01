"""*******************************************************************
* Copyright         : 2024 Bonvalot Isione
* File Name         : main.py
* Description       : This file contains the main script to generate adversarial examples.
*                    
* Revision History  : v.0.0
* Date				Author    		    Comments
* ---------------------------------------------------------------------------
* 01/07/2024		Bonvalot Isione		Created with main function.
*
******************************************************************"""
# Libraries
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import logging
import wandb
from huggingface_hub import login

# Local imports
from config_loader import load_config
from sentence_modification import eval_modif_perf, modify_sentences, ensemble_attack, llama_ensemble_attack
from sentiment_analysis import generate_classification_prompt
from data_processing import load_sst2_dataset, select_negative_sentences, select_positive_sentences, load_fs_examples
from model_loader import load_fooled_model_and_tokenizer, load_classification_model_and_tokenizer, load_llama
from model_utils import get_model_info, get_tokenizer_info
from sentiment_analysis import evaluate_classifier, evaluate_initial_classifier, attack_success_rate, classify, llama_classify
from utils import compute_averages, save_json, get_local_date_time, upgrade_file_version, save_info_to_csv
from logger import setup_logger


# main function
if __name__ == "__main__":
    # Load configuration
    config = load_config("config.ini")
    logs_folder = config["Paths"]["logs_folder"]

    # config variables
    FS_STRATEGY = config["Prompting_mode"]["few_shot_strategy"]
    ENS_STRATEGY = config["Prompting_mode"]["ensemble_strategy"]
    REWARD_STRATEGY = config["Prompting_mode"]["reward_prediction_confidence"]
    PROMPTING_GUIDANCE = config["Prompting_mode"]["prompting_guidance"]
    THR_HWORDS = float(config["Parameters"]["threshold_hwords"])
    THR_HBERT = float(config["Parameters"]["threshold_hbert"])

    # Set up logger
    if config["Debug"]["debug_mode"] == "True":
        logger = setup_logger(__name__, logs_folder, level=logging.DEBUG)
        logger.debug("Debug mode activated")
    else:
        logger = setup_logger(__name__, logs_folder, level=logging.INFO)
        logger.info("Info mode activated")

    # Load environment variables
    load_dotenv()
    token = os.getenv("ACCESS_TOKEN")
    login(token=token)

    wandb_token = os.getenv("WANDB_API_KEY")
    logger.info("Environment variables loaded")

    # Set up wandb
    wandb_project_name = config["Wandb"]["project_name"]
    if config["Debug"]["debug_mode"] == "False":
        wandb.login(key=wandb_token)

        if ENS_STRATEGY == "True":
            job_name = f"ENS_{ENS_STRATEGY}_FS_{FS_STRATEGY}_Reward_{REWARD_STRATEGY}_{THR_HWORDS}_{THR_HBERT}_{get_local_date_time()}"        
        else:
            job_name = f"PROMPT_{PROMPTING_GUIDANCE}_FS_{FS_STRATEGY}_Reward_{REWARD_STRATEGY}_{THR_HWORDS}_{THR_HBERT}_{get_local_date_time()}"

        group_name = "group_" + wandb_project_name
        wandb.init(project=wandb_project_name, name = job_name, config=config, group=group_name)
        logger.info(f"Wandb initialized with project {wandb_project_name} and job name {job_name}")


    # Load fs examples
    fs_examples = load_fs_examples(config["Paths"]["fs_examples_path"])

    # Load dataset
    loaded_dataset = load_sst2_dataset(config["Paths"]["dataset_path"])
    loaded_dataset = loaded_dataset[:int(config["Parameters"]["partial_dataset_size"])]
    logger.info("Dataset %s loaded with first %s rows", config["Paths"]["dataset_path"], int(config["Parameters"]["partial_dataset_size"]))


    # load fooled model and tokenizer
    if wandb_project_name == "llama2_gen_adv_examples":
        llama_model = "meta-llama/Llama-2-7b-chat-hf"
        fooled_model, fooled_tokenizer = load_llama(llama_model, token)
        classifier_model, classifier_tokenizer = load_llama(llama_model, token)  
    else:
        fooled_model, fooled_tokenizer = load_fooled_model_and_tokenizer(config["Models"]["fooled_model"], config["Settings"]["device"], token)
        classifier_model, classifier_tokenizer = load_classification_model_and_tokenizer(config["Models"]["classifier_model"], config["Settings"]["device"], token)
    
    
    fooled_model_info = get_model_info(fooled_model)
    fooled_tokenizer_info = get_tokenizer_info(fooled_tokenizer)
    classifier_model_info = get_model_info(classifier_model)
    classifier_tokenizer_info = get_tokenizer_info(classifier_tokenizer)
    
    logger.info(f"Fooled model loaded")
    logger.info(f"Classifier model loaded")

    pos_modif_success_rate = []
    neg_modif_success_rate = []
    final_modified_sentence = ""
    hwords_hbert_list = []
    attack_success_rate_per_iter = []
    perturbation_instructions = [
                            "Replace at most two words in the sentence with synonyms.",
                            "Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
                            "Add at most two semantically neutral words to the sentence.",                                                
                                ] #only W1, W2, W3

    for i in tqdm(range(int(config["Parameters"]["number_iterations"]))):
        if i == 0:
            torch.cuda.empty_cache()
            # classify sentiment
            classification_prompts = pd.DataFrame(loaded_dataset.apply(generate_classification_prompt, axis = 1), columns=["text"]).values.tolist()
            prompt_used = config["Prompting_mode"]["prompting_guidance"]
            loaded_dataset["classification_prompts"] = classification_prompts
            ground_truth_sentiment_list = loaded_dataset["sentiment"].tolist()

            if wandb_project_name == "llama2_gen_adv_examples":
                initial_predicted_sentiment_list, initial_predicted_logits, initial_predicted_confidence_score = llama_classify(loaded_dataset, "classification_prompts", classifier_model, classifier_tokenizer)
            else:    
                initial_predicted_sentiment_list, initial_predicted_logits, initial_predicted_confidence_score = classify(loaded_dataset, "classification_prompts", classifier_model, classifier_tokenizer)
            
            initial_classifier_results = evaluate_initial_classifier(ground_truth_sentiment_list, initial_predicted_sentiment_list)
            logger.info(f"Initial classifier results: {initial_classifier_results}")

            if initial_classifier_results['f1'] < 0.8 or initial_classifier_results['precision'] < 0.8 or initial_classifier_results[
            'recall'] < 0.8:
                logger.warning("Classifier performance is too low, please select another classifier")

            # Select negative and positive sentences
            initial_classified_dataset = loaded_dataset.copy()
            initial_classified_dataset["predicted_labels"] = initial_predicted_sentiment_list

            predicted_initial_positive_dataset, initial_idx_positive_dataset = select_positive_sentences(initial_classified_dataset)
            predicted_initial_negative_dataset, initial_idx_negative_dataset = select_negative_sentences(initial_classified_dataset)

            logger.info(f"Inital negative dataset size: {len(predicted_initial_negative_dataset)}")
            logger.info(f"Inital positive dataset size: {len(predicted_initial_positive_dataset)}")
            
            modified_negative_dataset = predicted_initial_negative_dataset
            modified_positive_dataset = predicted_initial_positive_dataset

            final_modified_dataset = initial_classified_dataset.copy()
            logger.info("Iteration 0 done")

            predicted_confidence_score = initial_predicted_confidence_score
            predicted_logits = initial_predicted_logits

        
        else:
            torch.cuda.empty_cache()
            if modified_negative_dataset.empty:
                logger.info("No more negative sentences to modify")
                pass
            if modified_positive_dataset.empty:
                logger.info("No more positive sentences to modify")
                pass

            # check if modified_negative_dataset or modified_positive_dataset is empty
            if modified_positive_dataset.empty and modified_negative_dataset.empty:
                logger.info("No more sentences to modify")
                #TODO: handle issue when no more positive or negative sentences to modify for final step
                break

            else:
                logger.info(f"Starting iteration {i}")
            
                if ENS_STRATEGY == "True":
                    if wandb_project_name == "llama2_gen_adv_examples":
                        modified_pos_dataset, hwords, hbert = llama_ensemble_attack(modified_positive_dataset, fooled_model, fooled_tokenizer, perturbation_instructions, FS_STRATEGY, REWARD_STRATEGY, THR_HWORDS, THR_HBERT, fs_examples, predicted_logits, predicted_confidence_score, "positive", "negative")
                        modified_neg_dataset, hwords, hbert = llama_ensemble_attack(modified_negative_dataset, fooled_model, fooled_tokenizer, perturbation_instructions, FS_STRATEGY, REWARD_STRATEGY, THR_HWORDS, THR_HBERT, fs_examples, predicted_logits, predicted_confidence_score, "negative", "positive")

                    else:
                        modified_pos_dataset, hwords, hbert = ensemble_attack(modified_positive_dataset, fooled_model, fooled_tokenizer, perturbation_instructions, FS_STRATEGY, REWARD_STRATEGY, THR_HWORDS, THR_HBERT, fs_examples, predicted_logits, predicted_confidence_score, "positive", "negative")
                        modified_neg_dataset, hwords, hbert = ensemble_attack(modified_negative_dataset, fooled_model, fooled_tokenizer, perturbation_instructions, FS_STRATEGY, REWARD_STRATEGY, THR_HWORDS, THR_HBERT, fs_examples, predicted_logits, predicted_confidence_score, "negative", "positive")
                else:
                    modified_pos_dataset, hwords, hbert = modify_sentences(modified_positive_dataset, fooled_model, fooled_tokenizer, perturbation_instructions, FS_STRATEGY, REWARD_STRATEGY, THR_HWORDS, THR_HBERT, PROMPTING_GUIDANCE, fs_examples, predicted_logits, predicted_confidence_score, "positive", "negative")
                    modified_neg_dataset, hwords, hbert = modify_sentences(modified_negative_dataset, fooled_model, fooled_tokenizer, perturbation_instructions, FS_STRATEGY, REWARD_STRATEGY, THR_HWORDS, THR_HBERT, PROMPTING_GUIDANCE, fs_examples, predicted_logits, predicted_confidence_score, "negative", "positive")
                           
                hwords_hbert_dict = {'hwords': hwords, 'hbert': hbert}
                hwords_hbert_list.append(hwords_hbert_dict)

                # Modification success rate
                pos_msr_per_iter = eval_modif_perf(modified_positive_dataset, modified_pos_dataset)
                neg_msr_per_iter = eval_modif_perf(modified_negative_dataset, modified_neg_dataset)
                pos_modif_success_rate.append(pos_msr_per_iter)
                neg_modif_success_rate.append(neg_msr_per_iter)

                logger.info(f"Positive modification success rate: {pos_msr_per_iter}")
                logger.info(f"Negative modification success rate: {neg_msr_per_iter}")

                modified_combined_dataset = pd.concat([modified_pos_dataset, modified_neg_dataset])
                classification_prompts = pd.DataFrame(modified_combined_dataset.apply(generate_classification_prompt, axis=1), columns=["text"]).values.tolist()
                modified_combined_dataset["classification_prompts"] = classification_prompts

                if wandb_project_name == "llama2_gen_adv_examples":
                    predicted_sentiment_list, predicted_logits, predicted_confidence_score = llama_classify(modified_combined_dataset, "classification_prompts", classifier_model, classifier_tokenizer)
                else:
                    predicted_sentiment_list, predicted_logits, predicted_confidence_score = classify(modified_combined_dataset, "classification_prompts", classifier_model, classifier_tokenizer)
         
                classified_combined_dataset = modified_combined_dataset.copy()
                classified_combined_dataset["predicted_labels"] = predicted_sentiment_list


                modified_positive_dataset, idx_positive_dataset = select_positive_sentences(classified_combined_dataset)
                modified_negative_dataset, idx_negative_dataset = select_negative_sentences(classified_combined_dataset)

                final_modified_dataset.loc[idx_positive_dataset.to_list()] = modified_positive_dataset
                final_modified_dataset.loc[idx_negative_dataset.to_list()] = modified_negative_dataset

                asr_per_iter = attack_success_rate(initial_classified_dataset, final_modified_dataset)


                attack_success_rate_per_iter.append(asr_per_iter)


                if config["Debug"]["debug_mode"] == "False":
                    wandb_logs = {
                    "pos_modif_success_rate_per_iter": pos_msr_per_iter,
                    "neg_modif_success_rate_per_iter": neg_msr_per_iter,
                    "asr_per_iter": asr_per_iter,
                    "hwords": hwords,
                    "hbert_f1": hbert["f1"][0],
                    "hbert_precision": hbert["precision"][0],
                    "hbert_recall": hbert["recall"][0],
                    "iteration_step": i
                    }
                    wandb.log(wandb_logs)
                

    # Final modification success rate
    final_modified_negative_dataset, _ = select_negative_sentences(final_modified_dataset)
    final_modif_success_rate = eval_modif_perf(predicted_initial_negative_dataset, final_modified_negative_dataset)
    final_modified_positive_dataset, _ = select_positive_sentences(final_modified_dataset)
    final_modif_success_rate = eval_modif_perf(predicted_initial_positive_dataset, final_modified_positive_dataset)
    logger.info(f"Final modification success rate (between initial neg/pos and final neg/pos): {final_modif_success_rate}")

    # Compute averages
    averages = compute_averages(hwords_hbert_list)
    logger.info(f"Averages: {averages}")

    # ask the same classifier to predict the new dataset labels (classify final_modified_dataset sentiment)
    classification_prompts = pd.DataFrame(final_modified_dataset.apply(generate_classification_prompt, axis = 1), columns=["text"]).values.tolist()
    final_modified_dataset["classification_prompts"] = classification_prompts


    if wandb_project_name == "llama2_gen_adv_examples":
        final_predicted_sentiment_list, final_predicted_logits, final_predicted_confidence_score = llama_classify(final_modified_dataset, "classification_prompts", classifier_model, classifier_tokenizer)
    else:
        final_predicted_sentiment_list, final_predicted_logits, final_predicted_confidence_score = classify(final_modified_dataset, "classification_prompts", classifier_model, classifier_tokenizer)

    final_classifier_results = evaluate_classifier(initial_predicted_sentiment_list, final_predicted_sentiment_list)
    logger.info(f"Final classifier results after attacking (between initial classified and final classified): {final_classifier_results}")
    final_classified_dataset = final_modified_dataset.copy()
    final_classified_dataset["predicted_labels"] = final_predicted_sentiment_list

    # Compute attack success rate
    attack_success_rate = attack_success_rate(initial_classified_dataset, final_classified_dataset)
    logger.info(f"Attack success rate: {attack_success_rate}")

    if config["Debug"]["debug_mode"] == "False":
    # Save final modified dataset
        final_modif_dataset_fileversion = upgrade_file_version(config["Paths"]["final_modified_dataset_folder"], ".parquet") 
        final_modif_dataset_folder = config["Paths"]["final_modified_dataset_folder"] 
        final_modif_dataset_path = final_modif_dataset_folder + "final_modif_dataset" + get_local_date_time() + "_v" + final_modif_dataset_fileversion + ".parquet"
        final_modified_dataset.to_parquet(final_modif_dataset_path)
        logger.info(f"Final_modified dataset saved in {final_modif_dataset_path}")


    #save information in a json file
    info_dict_json = {
        "date_time": get_local_date_time(),
        "number_iterations": int(config["Parameters"]["number_iterations"]),
        "loaded_dataset": config["Paths"]["dataset_path"],
        "partial_dataset_size": int(config["Parameters"]["partial_dataset_size"]),
        "Inference_mode : check_fidelity_mode = ": config["Inference_mode"]["check_fidelity_mode"],
        "Thresholds if check_fidelity_mode ": {"THR1": config["Parameters"]["threshold_hwords"], "THR2": config["Parameters"]["threshold_hbert"]},
        "averages": averages,
        "initial_classified_dataset[:50]": initial_classified_dataset.head(50).to_dict(),
        "initial_classifier": config["Models"]["classifier_model"],
        "initial_classifier_precision": initial_classifier_results["precision"],
        "initial_classifier_recall": initial_classifier_results["recall"],
        "initial_classifier_f1": initial_classifier_results["f1"],
        "prompt_used": config["Prompting_mode"]["prompting_guidance"],
        "few_shot_strategy": config["Prompting_mode"]["few_shot_strategy"],
        "reward_prediction_confidence": config["Prompting_mode"]["reward_prediction_confidence"],
        "ensemble_strategy": config["Prompting_mode"]["ensemble_strategy"],
        "attack_success_rate": attack_success_rate,
        "final_modif_success_rate": final_modif_success_rate,
        "final_classifier_precision": final_classifier_results["precision"],
        "final_classifier_recall": final_classifier_results["recall"],
        "final_classifier_f1": final_classifier_results["f1"],
        "final_modified_dataset[:50]": final_modified_dataset.head(50).to_dict(),
        "attack_success_rate_per_iter": attack_success_rate_per_iter,
        "pos_modif_success_rate": pos_modif_success_rate,
        "neg_modif_success_rate": neg_modif_success_rate,
        "hwords_hbert_list": hwords_hbert_list,
        "fooled_model_info": fooled_model_info,
        "fooled_tokenizer_info": fooled_tokenizer_info,
    }

    # Save information in a JSON file
    json_path_folder = config["Paths"]["info_json_folder"] 
    json_path = save_json(info_dict_json, json_path_folder)
    logger.info(f"Information Json saved in {json_path}")


    # Save information in a CSV file
    info_dict_csv = { 
        "date_time": get_local_date_time(),
        "number_iterations": int(config["Parameters"]["number_iterations"]),
        "partial_dataset_size": int(config["Parameters"]["partial_dataset_size"]),
        "Check_fidelity_mode": config["Inference_mode"]["check_fidelity_mode"],
        "THR1_hwords": config["Parameters"]["threshold_hwords"],
        "THR2_hbert": config["Parameters"]["threshold_hbert"],
        "average_hwords": averages["avg_hwords"],
        "average_hbert_precision": averages["avg_precision"],
        "average_hbert_recall": averages["avg_recall"],
        "average_hbert_f1": averages["avg_f1"],
        "initial_classifier": config["Models"]["classifier_model"],
        "initial_classifier_precision": initial_classifier_results["precision"],
        "initial_classifier_recall": initial_classifier_results["recall"],
        "initial_classifier_f1": initial_classifier_results["f1"],
        "prompt_used": config["Prompting_mode"]["prompting_guidance"],
        "few_shot_strategy": config["Prompting_mode"]["few_shot_strategy"],
        "reward_prediction_confidence": config["Prompting_mode"]["reward_prediction_confidence"],
        "ensemble_strategy": config["Prompting_mode"]["ensemble_strategy"],
        "attack_success_rate": attack_success_rate,
        "final_modif_success_rate": final_modif_success_rate,
        "final_classifier_precision": final_classifier_results["precision"],
        "final_classifier_recall": final_classifier_results["recall"],
        "final_classifier_f1": final_classifier_results["f1"],
        "pos_modif_success_rate": pos_modif_success_rate,
        "neg_modif_success_rate": neg_modif_success_rate,
        "asr_per_iter": attack_success_rate_per_iter,
        "hwords_hbert_list": hwords_hbert_list,
    }
    if config["Debug"]["debug_mode"] == "False":
        wandb_final_logs = { 
            "average_hwords": averages["avg_hwords"],
            "average_hbert_precision": averages["avg_precision"],
            "average_hbert_recall": averages["avg_recall"],
            "average_hbert_f1": averages["avg_f1"],
            "initial_classifier_precision": initial_classifier_results["precision"],
            "initial_classifier_recall": initial_classifier_results["recall"],
            "initial_classifier_f1": initial_classifier_results["f1"],
            "attack_success_rate": attack_success_rate,
            "final_modif_success_rate": final_modif_success_rate,
            "final_classifier_precision": final_classifier_results["precision"],
            "final_classifier_recall": final_classifier_results["recall"],
            "final_classifier_f1": final_classifier_results["f1"],
            "pos_modif_success_rate": pos_modif_success_rate,
            "neg_modif_success_rate": neg_modif_success_rate,
            "hwords_hbert_list": hwords_hbert_list,
            "iteration_step": i
            }
        
        wandb.log(wandb_final_logs)
        wandb.finish()
        logger.info("Wandb finished")

    if config["Debug"]["debug_mode"] == "False":
        # save information in a csv file
        save_info_to_csv(info_dict_csv, config["Paths"]["csv_info_path"])
        logger.info(f"Information CSV saved in {config['Paths']['csv_info_path']}")
        
    logger.info("End of the script")