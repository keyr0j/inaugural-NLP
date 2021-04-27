import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


def lemmatize_stemming(text):
    return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

sentenceList = []

for sentence in open("speeches/trump.txt", 'r', encoding='utf-8'):
    sentenceList.append(sentence)

temp = []

for sentence in sentenceList:
    if (sentence.rstrip() != ""):
        temp.append(sentence.rstrip())

sentenceList = temp

print(sentenceList)

# Preview
'''
sample = sentenceList[0]
print('original sentence: ')
words = []
for word in sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized sentence: ')
print(preprocess(sample))
'''
processed_sentences = list(map(preprocess, sentenceList))
print(processed_sentences)

dictionary = gensim.corpora.Dictionary(processed_sentences)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]
print(bow_corpus)