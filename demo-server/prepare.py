import re
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# nltk.download('wordnet')

np.random.seed(seed=234)

N = 100

train = pd.read_csv("articles2.csv")

with open('input.txt', 'w') as f:
    f.write(train['content'].str.cat(sep='\n'))

with open("input.txt") as input_file:
    articles = [next(input_file) for x in range(N)]

joined_articles = [" ".join(articles)]

# The below takes out anything that's not a letter, replacing it with a space, as well as any single letter that is not the pronoun "I" or the article "a."
lemmatizer = WordNetLemmatizer()

def clean_article(article):
    art1 = re.sub(r"[^A-Za-z]", ' ', article)
    art2 = re.sub(r"\s[B-HJ-Zb-hj-z]\s", ' ', art1)
    art3 = re.sub(r"^[B-HJ-Zb-hj-z]\s", ' ', art2)
    art4 = re.sub(r"\s[B-HJ-Zb-hj-z]$", ' ', art3)
    art5 = re.sub(r'\s+', ' ', art4).strip()
    sentence = art5.lower()
    lemmas = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence)]
    lemmatized_sentence = ' '.join(lemmas)

    return lemmatized_sentence

# cleaned_text = [clean_article(article) for article in articles]

# with open('cleaned.txt', 'w') as output_file:
#     for item in cleaned_text:
#         output_file.write("%s\n" % item)

# The below breaks up the words into n-grams of length 1 to 5 and puts their counts into a Pandas dataframe with the n-grams as column names.
# The maximum number of n-grams can be specified if a large corpus is being used.
ngram_bow = CountVectorizer(
    stop_words=None, preprocessor=clean_article, tokenizer=WhitespaceTokenizer().tokenize,
    ngram_range=(1,5), max_features=None, max_df=1.0, min_df=1, binary=False
)
ngram_count_sparse = ngram_bow.fit_transform(joined_articles)
ngram_count = pd.DataFrame(ngram_count_sparse.toarray())
ngram_count.columns = ngram_bow.get_feature_names()

ngram_count.to_csv("ngram_count.csv")

# The below turns the n-gram-count dataframe into a Pandas series with the n-grams as indices for ease of working with the counts.
# The second line can be used to limit the n-grams used to those with a count over a cutoff value.
sums = ngram_count.sum(axis=0)
sums = sums[sums > 0]
sums.to_csv("sums.csv", header=False)

ngrams = list(sums.index.values)

# The function below gives the total number of occurrences of 1-grams in order to calculate 1-gram frequencies
def number_of_onegrams(sums):
    onegrams = 0

    for ng in ngrams:
        ng_split = ng.split(" ")
        if len(ng_split) == 1:
            onegrams += sums[ng]

    return onegrams

# The function below makes a series of 1-gram frequencies.
# This is the last resort of the back-off algorithm if the n-gram completion does not occur in the corpus with any of the prefix words.
def base_freq(og):
    freqs = pd.Series()

    for ng in ngrams:
        ng_split = ng.split(" ")
        if len(ng_split) == 1:
            freqs[ng] = sums[ng] / og

    return freqs

# For use in later functions so as not to re-calculate multiple times:
bf = base_freq(number_of_onegrams(sums))
bf.to_csv("base_freq.csv", header=False)
