import pandas as pd
import os

os.chdir("C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/wikidata/")

def clean_person_names(column):
    column = column.split("@")[0] #@[a-z]{2} matches @ and two random letters
    column = column.replace("\"", '')
    return column

def main():
    #load English WikiData person list
    print("***ENGLISH***")
    person_list_eng = pd.read_csv("list_of_all_persons_eng.csv", encoding ='cp1252')
    print(person_list_eng.head())

    if len(person_list_eng) == len(person_list_eng["?personLabel"].unique()):
        print("There are {} unique person names.".format(len(person_list_eng)))
    else:
        print("Number of persons in dataset: ", len(person_list_eng))
        print("Number of unique person names: ", len(person_list_eng["?personLabel"].unique()))

    #convert string of person name
    person_list_eng['?personLabel'] = person_list_eng['?personLabel'].apply(clean_person_names)
    print(person_list_eng.head())

    print("---"*10)
    """
    #load German Wikidata person list
    print("***GERMAN***")
    person_list_ger= pd.read_csv("list_of_all_persons_ger.csv", encoding ='utf-8')
    print(person_list_ger.head())

    if len(person_list_ger) == len(person_list_ger["?personLabel"].unique()):
        print("There are {} unique person names.".format(len(person_list_ger)))
    else:
        print("Number of persons in dataset: ", len(person_list_ger))
        print("Number of unique person names: ", len(person_list_ger["?personLabel"].unique()))

    #convert string of person name
    person_list_ger['?personLabel'] = person_list_ger['?personLabel'].apply(clean_person_names)
    print(person_list_ger.head())
    
    #merge eng and ger dataset
    person_list = pd.concat([person_list_eng, person_list_ger])
    print("Number of persons in merged list: ", len(person_list))
    person_list = person_list.drop_duplicates()
    print("Without duplicates ", len(person_list))
    """

    #save list
    person_list_eng.to_csv("list_of_all_persons_eng_CLEANED.csv", encoding='utf-8')

if __name__ == "__main__":
    main()