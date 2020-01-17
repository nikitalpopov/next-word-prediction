# Python library imports:
import nltk
import pandas as pd
import re
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

N = 100

sums = pd.Series.from_csv(f"{N}/sums.csv", index_col=0, header=None)
# print(sums.head())
# sums.fillna('nan', inplace=True) # because string 'nan' is considered as NaN by pandas
sums.dropna(inplace=True)
ngrams = list(sums.index.values)
# print(ngrams[0:5])
bf = pd.Series.from_csv(f"{N}/base_freq.csv", index_col=0, header=None)
print(bf.head())

# The function below finds any n-grams that are completions of a given prefix phrase with a specified number (could be zero) of words 'chopped' off the beginning.
# For each, it calculates the count ratio of the completion to the (chopped) prefix, tabulating them in a series to be returned by the function.
# If the number of chops equals the number of words in the prefix (i.e. all prefix words are chopped), the 1-gram base frequencies are returned.
def find_completion_scores(prefix, chops=0, factor=0.4):
    cs = pd.Series()
    prefix_split = prefix.split(" ")
    l = len(prefix_split)
    prefix_split_chopped = prefix_split[chops:l]
    new_l = l - chops

    if new_l == 0:
        return factor**chops * bf

    prefix_chopped = ' '.join(prefix_split_chopped)

    for ng in ngrams:
        try: # because string 'nan' is considered as NaN by pandas
            ng_split = ng.split(" ")
            if (len(ng_split) == new_l + 1) and (ng_split[0:new_l] == prefix_split_chopped):
                cs[ng_split[-1]] = factor**chops * sums[ng] / sums[prefix_chopped]
        except:
            continue

    return cs

wn.ensure_loaded()
lemmatizer = WordNetLemmatizer()

#########################################
#
#             SERVER PART
#
#########################################

from flask import Flask, escape, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PREFIX_LENGTH = 4
SUGGESTIONS_LENGTH = 5

@app.route('/')
def index():
    print()
    input_string = request.args.get('input', '')
    print(f"INPUT: '{input_string}'")

    prefix = " ".join(input_string.replace('  ', ' ').split(' ')[-PREFIX_LENGTH:])
    lemmas = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(prefix)]
    lemmatized_prefix = ' '.join(lemmas)
    print(f"\tused prefix: '{prefix}'", f"\tlemmatized to: '{lemmatized_prefix}'", sep='\n')
    best_next_words = find_completion_scores(lemmatized_prefix).to_frame(name="probability")
    best_next_words = best_next_words.sort_values(by='probability', ascending=False)
    ans = { "suggestions": best_next_words.head(SUGGESTIONS_LENGTH).index.tolist() }

    print(f"OUTPUT: {ans}")

    return ans
