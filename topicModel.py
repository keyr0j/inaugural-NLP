import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
#nltk.download('wordnet')

if __name__ == '__main__':

    # Runs the snowball stemmer and word net lemmatizer
    def lemmatize_stemming(text):
        return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    # Tokenizes and removes stopwords
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    # Opens the speech
    text = open("speeches/biden.txt", 'r', encoding='utf-8')

    # Tokenizes the speech into sentences
    sentenceList = nltk.sent_tokenize(text.read())

    # Removes any whitespace
    temp = []
    for sentence in sentenceList:
        if (sentence.rstrip() != ""):
            temp.append(sentence.rstrip())

    sentenceList = temp

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
    # Pre-processes each sentence
    processed_sentences = list(map(preprocess, sentenceList))

    # Creates dictionary using the prepreocessed sentences
    dictionary = gensim.corpora.Dictionary(processed_sentences)

    # Prints the first 10 items in the dictionary
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break

    # Filters out tokens that appear:
    # Less than 2 sentences OR
    # More than 0.5 sentences
    # After the first 2, keep only the first 100,000 most frequent tokens
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)

    # For each document we create a dictionary reporting how many words and how many times those words appear
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]


    from gensim import corpora, models

    # Train model using Bag of Words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    print("-----Bag-of-Words-----")
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # Create tf-idf model
    tfidf = models.TfidfModel(bow_corpus)

    # Apply to entire corpus
    corpus_tfidf = tfidf[bow_corpus]

    # Train model using TF-IDF
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    print("-----TF-IDF-----")
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))