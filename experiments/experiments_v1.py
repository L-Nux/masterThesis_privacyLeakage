import torch
import tensorflow
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import pickle
import numpy as np
import pandas as pd

import sys
from timeit import default_timer as timer

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


def main():

    ################Loading BERT NER models################
    #BERT Base NER
    tokenizer_base = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model_base = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    #BERT Large NER
    tokenizer_large = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model_large = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    #model setup
    chosen_tokenizer = tokenizer_large
    chosen_model = model_large
    nlp = pipeline("ner", model=chosen_model, tokenizer=chosen_tokenizer)

    #save details
    model_name = "BERT_Large"
    sys.stdout = open("reports/expv1_report_"+model_name+".txt", "a")
    print("This information is on model: ", model_name)
    print("---"*10)
    start = timer()

    ################Loading experiment data################
    print("Loading data...")
    path_for_experiment_data = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/"
    #name list
    dev_name_file = open(path_for_experiment_data+"dev_test/dev_balanced_UL", "rb")
    dev_names = pickle.load(dev_name_file)
    print("There are {} PER name pairs".format(len(dev_names)))
    print(dev_names)

    #context sentences (simple, single PER)
    with open(path_for_experiment_data + "experiment_sentences.txt", encoding='utf-8') as f:
        context_simple = f.read()

    context_simple = context_simple.split("\n")
    print("Context sentences for experiment: ", context_simple)

    ################Model predictions################
    print("Model Predictions started...")
    results = pd.DataFrame(columns=['context_sentence', 'pair_ID', 'is_in_train', 'name', 'output_name', 'score'])
    i = 0
    j = 0
    for i in range(len(context_simple)):
        test_sentence = context_simple[i]
        for j in range(len(dev_names)):
            name_neg = dev_names[j][0]
            name_neg = name_neg.replace("\"+", "")
            name_pos = dev_names[j][1]
            name_pos = name_pos.replace("\"+", "")
            #negative sample
            temp_sentence_neg = test_sentence.replace("MASK", name_neg)
            score_neg, pred_name_neg = get_BERT_scores(temp_sentence_neg, nlp)
            dict_neg = {'context_sentence' : test_sentence, 'pair_ID' : j, 'is_in_train': 0, 'name' : name_neg, 'output_name' : pred_name_neg, 'score' : score_neg}
            results = results.append(dict_neg, ignore_index=True)
            #positive sample
            temp_sentence_pos = test_sentence.replace("MASK", name_pos)
            score_pos, pred_name_pos = get_BERT_scores(temp_sentence_pos, nlp)
            dict_pos = {'context_sentence' : test_sentence, 'pair_ID' : j, 'is_in_train': 1, 'name' : name_pos, 'output_name' : pred_name_pos, 'score' : score_pos}
            results = results.append(dict_pos, ignore_index=True)
            #keeping track
            if j % 100 == 0:
                print("PER pair with index {}: {}".format(j, dev_names[j]))
                print("*****")
            i+=1
            j+=1

    #save results
    results.to_csv(path_for_experiment_data+"results/results_expv1_"+model_name+".csv", encoding ='utf-8')
    print("These are the first rows of the saved dataframe:")
    print(results.head())
    print("Results for {} using simple context_sentences are saved!".format(model_name))
    #close file
    dev_name_file.close()

    time_needed = timer() - start
    print("Time needed for experiment:", time_needed)
    sys.stdout.close()

if __name__ == "__main__":
    main()
