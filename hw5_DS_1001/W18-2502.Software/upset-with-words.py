import argparse
import os

import pandas as pd

DIR = "stopwords/en/"

def get_words(path):
    input_file = open(path, encoding='UTF-8')
    stop_words = input_file.readlines()
    input_file.close()
    stop_words = list(map(lambda x:x.strip(), stop_words))
    # only keep English words
    #stop_words = list(map(lambda x:re.sub("[^a-z]+", "", x), stop_words))
    stop_words = list(filter(lambda x:len(x) > 0, stop_words))
    # there aresome duplicate stop words
    stop_words = list(set(stop_words))
    return stop_words


ap = argparse.ArgumentParser()
ap.add_argument('queries', nargs='+')
ap.add_argument('-o', '--out', default=None)
args = ap.parse_args()

result = {}
for stop_word_file in os.listdir(DIR):
    # _none.txt doesn't contain any stop word
    if stop_word_file == "_none.txt":
        continue
    # these files contain ngrams, not just single words
    if stop_word_file in ["galago_structured.txt", "gilner_morales.txt"]:
        continue
    # This is meant to augment another list, and so is confusing here
    if stop_word_file == "galago_forumstop.txt":
        continue
    words = set(get_words(os.path.join(DIR, stop_word_file)))
    result[stop_word_file[:-4]] = {q: q in words for q in args.queries}

result = pd.DataFrame(result).transpose()

import upsetplot
import matplotlib.pyplot as plt

aggregates = result.astype(bool).reset_index().groupby(args.queries[::-1]).count()
upsetplot.plot(aggregates['index'].copy(), sort_sets_by=None,
               sort_by='cardinality',
               intersection_plot_elements=4)


if args.out is None:
    plt.show()
else:
    plt.savefig(args.out)
