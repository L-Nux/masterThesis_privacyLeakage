import torch
import tensorflow
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import random
import numpy as np
import pandas as pd

import sys
from timeit import default_timer as timer

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

def load_preprocess(name_of_file):
    path = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/"
    with open(path + name_of_file, encoding = 'utf-8') as f:
        name_list = f.read()
    #cleaning
    name_list = name_list.replace("\'", "")
    name_list = name_list.split(", ")
    name_list = [n.replace("\"+", "") for n in name_list]
    name_list = [n.rstrip() for n in name_list]
    return name_list

def get_BERT_scores(name_sentence, nlp):
    """
    This function feeds an input sequence to an NER tagger and retrieves information on the detected PER name and
    the score for the PER tags. The scores for the multi-token name are averaged.
    :param name_sentence: sequence (string) containing a PER name.
    :param nlp: NER tagger
    :return: averaged score for the PER tags, the name as identified by the model from the sequence if model output is not empty.
    Should the output be empty 'ND' (not detected) is returned.
    """

    model_prediction = nlp(name_sentence)
    if model_prediction:
        #check detected name
        pred_name = [k['word'] for k in model_prediction]
        pred_name = " ".join(pred_name)
        #calcuate mean of scores for name
        scores = [k['score'] for k in model_prediction]
        name_score = np.mean(scores)
    else:
        #in case model prediction comes back empty --> no PER tags detected in input string
        name_score = "ND" #ND --> Not Detected
        pred_name = "ND"

    return name_score, pred_name

def model_prediction(negative_list, context_simple, nlp):

    print("Model Predictions started...")
    results = pd.DataFrame(columns=['context_sentence', 'pair_ID', 'is_in_train', 'name', 'output_name', 'score'])
    i = 0
    j = 0
    for i in range(len(context_simple)):
        test_sentence = context_simple[i]
        for j in range(len(negative_list)):
            name_neg = negative_list[j]
            name_neg = name_neg.replace("\"+", "")
            #negative sample
            temp_sentence_neg = test_sentence.replace("MASK", name_neg)
            score_neg, pred_name_neg = get_BERT_scores(temp_sentence_neg, nlp)
            dict_neg = {'context_sentence' : test_sentence, 'pair_ID' : j, 'is_in_train': 0, 'name' : name_neg, 'output_name' : pred_name_neg, 'score' : score_neg}
            results = results.append(dict_neg, ignore_index=True)

            # keeping track
            if j % 100 == 0:
                print("Progress: {} names processed, current PER name {} ".format(j, negative_list[j]))
                print("*****")
            i += 1
            j += 1

    return results


def main():

    ################Loading BERT NER models################
    #BERT Base NER
    tokenizer_base = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model_base = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    #BERT Large NER
    tokenizer_large = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model_large = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    #model setup
    chosen_tokenizer = tokenizer_base
    chosen_model = model_base
    nlp = pipeline("ner", model=chosen_model, tokenizer=chosen_tokenizer)

    #save details
    model_name = "BERT_Base"
    sys.stdout = open("reports/expv2_report_"+model_name+".txt", "a")
    print("This information is on model: ", model_name)
    print("---"*10)
    start = timer()

    ################Loading experiment data################
    print("Loading data...")
    path_for_experiment_data = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/"
    # negative: names only in wikidata
    negative = load_preprocess("negative_wikidata_UL.txt")
    print("There are {} negative PER names".format(len(negative)))


    print("Random sampling of negative names in different dimensions:")
    # There are 826 positive names in the dev set
    # Basic idea: sample negative names, so that positive names present a certain fraction in a common dataset
    # fraction: 10% positives
    dim_10 = 7434
    # set random seed
    random.seed(202)
    neg_dim10 = random.sample(negative, dim_10)
    neg_dim1 = random.sample(negative, dim_10*10)
    neg_dim01 = random.sample(negative, dim_10*100)
    print("Number of negative names: {}, {}, {}.".format(len(neg_dim10), len(neg_dim1), len(neg_dim01)))
    print("Negative name sampling finished!")

    #context sentences (simple, single PER)
    with open(path_for_experiment_data + "experiment_sentences.txt", encoding='utf-8') as f:
        context_simple = f.read()

    context_simple = context_simple.split("\n")
    print("Context sentences for experiment: ", context_simple)

    ################Model predictions################

    results_dim10 = model_prediction(neg_dim10, context_simple, nlp)
    results_dim1 = model_prediction(neg_dim1, context_simple, nlp)
    #results_dim01 = model_prediction(neg_dim01, context_simple, nlp)

    #save results
    results_dim10.to_csv(path_for_experiment_data+"results/results_expv2_dim10_"+model_name+".csv", encoding ='utf-8')
    results_dim1.to_csv(path_for_experiment_data+"results/results_expv2_dim1_"+model_name+".csv", encoding ='utf-8')
    #results_dim01.to_csv(path_for_experiment_data+"results/results_expv2_dim01_"+model_name+".csv", encoding ='utf-8')
    print("These are the first rows of the saved dataframe dim10:")
    print(results_dim10.head())
    print("Results for {} using simple context_sentences are saved!".format(model_name))

    time_needed = timer() - start
    print("Time needed for experiment:", time_needed)
    sys.stdout.close()

if __name__ == "__main__":
    main()
