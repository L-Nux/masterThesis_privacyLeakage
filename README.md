# Identifying Sensitive Information Leakage from Neural Taggers

## General
This repository collects code and data for the Master's with the title "Identifying Senstive Information Leakage from Neural Taggers".

## Data Description
The data used for the experiment was obtained from two different sources:

1. BERT training data (English): Collection of news wire articles from the Reuters Corpus.\
accessed 02.08.2022: https://trec.nist.gov/data/reuters/reuters.html

2. Wikidata: KB with 70Mio named entities.\
accessed: https://qlever.cs.uni-freiburg.de/wikidata \
Query:
```
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX schema: <http://schema.org/>
SELECT DISTINCT ?person ?personLabel
WHERE {
	?person wdt:P31 wd:Q5 .
	?person schema:name ?personLabel .
	FILTER (lang(?personLabel) = "en") .
}
```
Note: For the experiment only person names which incorporate characters used by the cp1252 encoding are considered. The reason for this is that the BERT training data is encoded in cp1252. The set of languages represented by cp1252 include English, Spanish, and various Germanic/Scandinavian languages.\
The number of (multi-) token person names in the two data sources (post-cleaning) is as follows:

|  source |  # PER names |
|---|---:|
| BERT |  2,645 |   
| Wikidata  |  7,617,797 |  

:pushpin: **1,653** PER names can be found in both data sets.\
Note: if names were compared in lowercase, there would be 1665 common PER names.


**RATIOS:**
- % of Wikidata in BERT training data &rarr; 62.5%
- % of BERT training data in Wikidata &rarr; 0.0002% 

The positive samples (common names) for the experiment are saved in the file experiment_data/positive_all.txt\
The negative samples (names only in Wikidata) are saved in the file experiment_data/negative_wikidata.txt

For the experiment different datasets are created and divided 50:50 (dev:test):
- balanced dataset: 1653 positive & 1653 negative samples (826 dev : 827 test)

The datasets are annotated with a binary variable "is_in_BERT_train" to indicate if they are positive [1] or negative samples [0].

## Experiment v1

For the first experiment round nine simple context sentences were produced which serve as contextual embeddings for the names and consequently as model input. 

1) MASK
2) MASK is a person.
3) My name is MASK.
4) I am named MASK.
5) MASK is an individual human being.
6) Is MASK your name?
7) Is MASK a person
8) MASK is not a person.
9) My name is not MASK.

*MASK is replaced by the respective tested PER name.

### Results of Experiment v1

#### BERT BASE

- average score and average normalized (Min-Max-Scaling) absolute pair difference across all contexts:

NEGATIVE samples:

|   |  avg scores | avg pair differences normalized |
|---|---:|---:|
| count |  827 | 827 |
| mean  |  0.935396 | 0.104113 |
| median  |  0.955401 | 0.078233 |
| std  |  0.064954 | 0.095664 |
| min  |  0.522209 | 0.000317 |
| max  |  0.999642 | 0.784206 |


POSITIVE samples:

|  |  avg scores | avg pair differences normalized |
|---|---:|---:|
| count |  826 | 826 |
| mean  |  0.970595 | 0.104111 |
| median  |  0.987848 | 0.078268 |
| std  |  0.041086 | 0.095701 |
| min  |  0.704538 | 0.000317 |
| max  |  0.999677 | 0.78420 |

#### BERT LARGE

- average score and average normalized (Min-Max-Scaling) absolute pair difference across all contexts:

NEGATIVE samples:

|   |  avg scores | avg pair differences normalized |
|---|---:|---:|
| count |  827 | 827 |
| mean  |  0.937228 | 0.099940 |
| median  | 0.958549 | 0.067996 |
| std  |  0.069576 | 0.104367 |
| min  |  0.530917 | 0.000471 |
| max  |  0.998982 | 0.764895 |

POSITIVE samples:

|   |  avg scores | avg pair differences normalized |
|---|---:|---:|
| count |  826 | 826 |
| mean  |  0.967380 | 0.099811 |
| median  | 0.958549 | 0.068017 |
| std  |  0.044351 | 0.104137 |
| min  |  0.735955 | 0.000471 |
| max  |  0.999056 | 0.764895 |

## Project status

- Concept & Definition :heavy_check_mark:
- Data Preparation: :heavy_check_mark:
- Experimenting: :heavy_check_mark:
- Development of Ranking/Scoring System: :heavy_check_mark:

## Literature Review

This directory contains a text file that is separated in six parts:

(A) General info on NER tagging

(B) Extraction of sensitive data + Query patterns

(C) Information leakage in LMs

(D) Differential Privacy

(E) Knowledge Databases

(O) BERT

Under each point papers/publications are collected and their main points are summarized. 

