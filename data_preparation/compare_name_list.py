import pandas as pd
import numpy as np
import string
import re

path_to_lists = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/"

# BERT training data
bert_train_name_list = pd.read_csv(
    path_to_lists + "CoNLL_2003/conll2003_person_name_list/UNIQUE_person_name_list_train_data.txt")
bert_train_name_list = list(bert_train_name_list)
print("Pre-cleaning there are {} names in BERT training data list".format(len(bert_train_name_list)))

# Wikidata
wikidata_name_list = pd.read_csv(path_to_lists + "wikidata/list_of_all_persons_eng_CLEANED.csv", encoding="utf-8")
print(wikidata_name_list.head())
wikidata_name_list = pd.Series(wikidata_name_list['?personLabel']).unique().tolist()
print("Pre-cleaning there are {} names in Wikidata list".format(len(wikidata_name_list)))
print("***"*10)

###Cleaning###
"""
def clean_unicode(name):
    unwanted = [str("\u202a"), str("\u202c"), str("\u200b"), str("\u200e"), str("\u200f")]
    for character in unwanted:
        name = name.replace(character, '')
    return name
"""
def remove_brackets(name):
    name = re.sub("\(.*?\)", "", name)
    return name

def only_Ascii(name_list):
    # all ascii characters
    regexp = re.compile(r'[^\x00-\xff]')
    cleaned_list = []
    counter = 0
    for name in name_list:
        if not regexp.search(name):
            cleaned_list.append(name)
        else:
            counter += 1

    print("There were {} name(s) with non-ascii characters".format(counter))
    return cleaned_list

def cleaning_name_list(names):
    # convert all elements to strings
    names = [str(r) for r in names]
    # convert to lowercase
    #names = [x.lower() for x in names]
    #remove bracket text
    names = [remove_brackets(n) for n in names]
    # remove punctuation at beginning or end of string
    names = [n.lstrip(string.punctuation) for n in names]
    names = [n.rstrip(string.punctuation) for n in names]
    #remove strings with digits
    names = [x for x in names if not any(c.isdigit() for c in x)]
    #only include names with ascii characters
    names = only_Ascii(names)

    return names


# convert list of names to lowercase + string
bert_train_name_list = cleaning_name_list(bert_train_name_list)
wikidata_name_list = cleaning_name_list(wikidata_name_list)

print("Post-cleaning there are {} names in BERT training data list".format(len(bert_train_name_list)))
print("Post-cleaning there are {} names in Wikidata list".format(len(wikidata_name_list)))
print("***"*10)

# consider only names with min. 2 tokens
def contains_whitespace(string):
    if string.count(" ") >= 1:
        return string


def multiple_token_name(names):
    multiple_token_name_list = []
    for n in names:
        n = n.lstrip(" ")
        n = n.rstrip(" ")
        n = contains_whitespace(n)
        if n is not None:
            multiple_token_name_list.append(n)

    return multiple_token_name_list


bert_train_name_list = multiple_token_name(bert_train_name_list)
wikidata_name_list = multiple_token_name(wikidata_name_list)

print("Comparison of person name lists: ")
print("Number of multi-token names in BERT training data: ", len(bert_train_name_list))
print("Number of multi-token names in Wikidata (English): ", len(wikidata_name_list))

print("***"*10)


def common(train, wiki):
    match_in_both = [name for name in train if name in wiki]
    return match_in_both


common_names = common(bert_train_name_list, wikidata_name_list)
print(common_names)
print(len(common_names))

# names that are NOT in the common name list
negative_bert_train = list(np.setdiff1d(bert_train_name_list, common_names))
print("Number of names only in BERT Train: ", len(negative_bert_train))
negative_wikidata = list(np.setdiff1d(wikidata_name_list, common_names))
print("Number of names only in Wikidata: ", len(negative_wikidata))

# test
print(type(common_names), type(negative_bert_train), type(negative_wikidata))

# save name lists
path_for_experiment_data = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/"

# positive (names in BERT train + Wikidata)
with open(path_for_experiment_data + "positive_all.txt", 'w', encoding ='utf-8') as list_file1:
    list_file1.writelines("%s\n" % common_names)

# negative A (only in Wikidata)
with open(path_for_experiment_data + "negative_wikidata.txt", 'w', encoding ='utf-8') as list_file2:
    list_file2.writelines("%s\n" % negative_wikidata)

# negative B (only BERT train)
with open(path_for_experiment_data + "negative_BERT_train.txt", "w", encoding ='utf-8') as list_file3:
    list_file3.writelines("%s\n" % negative_bert_train)
