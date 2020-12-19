import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import upsetplot

result = pd.DataFrame()
words = ["has", "hasn", "hasn't", "hasnt", "have", "haven", "haven't", "havent"]
file_path = "stopwords/en/"
for stop_word_file in os.listdir(file_path):
    # _none.txt doesn't contain any stop word
    if stop_word_file == "_none.txt":
        continue
    # these files contain ngrams, not just single words
    if stop_word_file in ["galago_structured.txt", "gilner_morales.txt"]:
        continue
    input_file = open(file_path + stop_word_file, encoding='UTF-8')
    stop_words = input_file.readlines()
    input_file.close()
    stop_words = list(map(lambda x:x.strip(), stop_words))
    # only keep English words
    #stop_words = list(map(lambda x:re.sub("[^a-z]+", "", x), stop_words))
    stop_words = list(filter(lambda x:len(x) > 0, stop_words))
    # there aresome duplicate stop words
    stop_words = list(set(stop_words))
    for word in words:
        if word in stop_words:
            result.loc[stop_word_file, word] = 1
        else:
            result.loc[stop_word_file, word] = 0

