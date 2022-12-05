import pandas as pd
import string

path_deep_ai = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/CoNLL_2003/conll2003_deepai/"
train_data = pd.read_csv(path_deep_ai + 'train.txt', sep=" ", header=None)
# delete first row (no data)
train_data = train_data.iloc[1:, :]
# check unique NER tags
print(train_data[3].unique())
print(train_data.head())
print(len(train_data))

# select only PER instances (I-PER, B-PER)
train_per = train_data.loc[train_data[3].isin(["B-PER", "I-PER"])]
print(train_per.head())
# how many PER tokens
print(len(train_per))
# how many B-ER tokens (names)
print(len(train_per.loc[train_per[3].isin(["B-PER"])]))

#################################################################################
###Step 1###
# at the moment only 5-part names are collected in name_list
name_list = []
chosen_df = train_per
for index in chosen_df.index:
    row = chosen_df.loc[index]
    # print(row)
    # print(row[3])
    if row[3] == "B-PER":
        name1 = row[0]
        new_index = index + 1
        if new_index in chosen_df.index and chosen_df.loc[new_index][3] == "I-PER":
            name2 = chosen_df.loc[new_index][0]
            new_index = new_index + 1
            if new_index in chosen_df.index and chosen_df.loc[new_index][3] == "I-PER":
                name3 = chosen_df.loc[new_index][0]
                new_index = new_index + 1
                if new_index in chosen_df.index and chosen_df.loc[new_index][3] == "I-PER":
                    name4 = chosen_df.loc[new_index][0]
                    new_index = new_index + 1
                    if new_index in chosen_df.index and chosen_df.loc[new_index][3] == "I-PER":
                        name5 = chosen_df.loc[new_index][0]
                        name = name1 + " " + name2 + " " + name3 + " " + name4 + " " + name5
                        name_list.append(name)
                    else:
                        name = name1 + " " + name2 + " " + name3 + " " + name4
                        name_list.append(name)
                else:
                    name = name1 + " " + name2 + " " + name3
                    name_list.append(name)
            else:
                name = name1 + " " + name2
                name_list.append(name)
        else:
            name_list.append(name1)
    else:
        continue

#################################################################################
###Step 2###
# print(name_list)
# print(len(name_list))

# save all PER tokens found in training data
textfile = open(
    "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/CoNLL_2003//conll2003_person_name_list/FULL_person_name_list_train_data.txt",
    "w")
for element in name_list:
    textfile.write(element)
textfile.close()

#################################################################################
###Step 3###

# delete duplicates in name list
unique_names = pd.Series(name_list).unique().tolist()
print("There are {} unique token names in list".format(len(unique_names)))


# consider only names with min. 2 tokens
def contains_whitespace(string):
    if string.count(" ") >= 1:
        return string


multiple_token_name_list = []
for name in unique_names:
    name = contains_whitespace(name)
    if name is not None:
        multiple_token_name_list.append(name)

print("There are {} unique, multiple token names in list".format(len(multiple_token_name_list)))
print(multiple_token_name_list)

# save all PER tokens found in training data
textfile = open(
    "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/CoNLL_2003//conll2003_person_name_list/UNIQUE_person_name_list_train_data.txt",
    "w")
for element in multiple_token_name_list:
    textfile.write(element+",")
textfile.close()
