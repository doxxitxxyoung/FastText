import nltk
import os
import pickle
import csv
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

agdir = "/home/doyeong/fasttext/data/ag_news_csv/"
agtraindir = agdir+"train.csv"
agtestdir = agdir+"test.csv"

agtraindata = []
agtestdata = []

tokenizer = RegexpTokenizer("[a-zA-Z'`]+")
with open(agtraindir, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        row = [int(row[0]), row[1] + " " + row[2]]
        row[1] = row[1].lower()
        row[1] = tokenizer.tokenize(row[1])
        row[1] = [word for word in row[1] if word not in stopwords.words('english')]
        bigram_tuples = ngrams(row[1], 2)
        bigrams = [' '.join(grams) for grams in bigram_tuples]
        row[1] = row[1] + bigrams
        agtraindata.append(row)

with open(agtestdir, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        row = [int(row[0]), row[1] + " " + row[2]]
        row[1] = row[1].lower()
        row[1] = tokenizer.tokenize(row[1])
        row[1] = [word for word in row[1] if word not in stopwords.words('english')]
        bigram_tuples = ngrams(row[1], 2)
        bigrams = [' '.join(grams) for grams in bigram_tuples]
        row[1] = row[1] + bigrams
        agtestdata.append(row)

with open(agdir+"train.pickle", 'wb') as f:
    pickle.dump(agtraindata, f)

with open(agdir+"test.pickle", 'wb') as f:
    pickle.dump(agtestdata, f)
