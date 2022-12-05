import pandas as pd
import random
import pickle


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

#set random seed
random.seed(0)

# load preprocessed list of PER names
# positive: names in BERT training data and wikidata
positive = load_preprocess("positive_all_UL.txt")
# negative: names only in wikidata
negative = load_preprocess("negative_wikidata_UL.txt")

print("There are {} unique, positive samples and {} unique, negative samples.".format(len(positive), len(negative)))

#create neg datataset
k = len(positive) #1665
neg_balanced = random.sample(negative, k)

#create k-pairing dataset
#keys: [0] negative sample / [1] positive sample
#values: names
names_balanced = []
for (name_neg, name_pos) in zip(neg_balanced, positive):
    tempDic = {0 : str(name_neg), 1 : str(name_pos)}
    names_balanced.append(tempDic)

print(names_balanced)

#split into dev and test set
j = round(k/2)
random.shuffle(names_balanced)

devDict = names_balanced[j:]
testDict = names_balanced[:j]

#save data
path_for_experiment_data = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/dev_test/"
devFile = open(path_for_experiment_data+"dev_balanced_UL", "wb")
testFile = open(path_for_experiment_data+"test_balanced_UL", "wb")

pickle.dump(devDict, devFile)
devFile.close()
pickle.dump(testDict, testFile)
testFile.close()