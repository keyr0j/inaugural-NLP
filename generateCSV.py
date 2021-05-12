# Imports
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import re
from nltk.corpus import stopwords

# data
presidents = [('arthur', 1881), ('biden', 2021), ('bush', 2001), ('carter', 1977),
             ('clinton', 1993), ('fdr', 1933), ('jackson',1829), ('jefferson', 1801),
             ('kennedy', 1961), ('lincoln', 1861), ('nixon', 1969), ('obama', 2008),
             ('polk', 1845), ('roosevelt', 1905), ('truman', 1949), ('trump', 2016),
             ('washington', 1789), ('wilson', 1913)]


# Prints all of the scores of the sentences using Sentiment Intensity Analyzer
def printVADERScores(sentence):
  scores  = sia.polarity_scores(sentence)
  print("positive:  ", scores["pos"])
  print("negative:  ", scores["neg"])
  print("neutral:   ", scores["neu"])
  
# Prints all of the scores of the sentences using TextBlob
def printBlobScores(sentence):
  scores  = sentence.sentiment
  print("polarity:      ", scores[0])
  print("subjectivity:  ", scores[1])


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


allPositiveVADER = []
allNegativeVADER = []
allNeutralVADER = []
allPositiveBlob = []
allNegativeBlob = []
allNeutralBlob = []
allTopicLists = []
allTop10Words = []
allSentences = []

if __name__ == '__main__':

  for president in presidents:
    print(president)
    data = []
    file = "speeches/" + president[0] + ".txt";
    # Input speech
    text = open(file, 'r', encoding='utf-8')
    data = nltk.sent_tokenize(text.read())
    filtered_data = []
    for sentence in data:
      if (sentence.rstrip() != ""):
          filtered_data.append(sentence.rstrip())
    
    #----------VADER----------      
    sia = SentimentIntensityAnalyzer()
    statsVADER = {
      "positive": [],
      "negative": [],
      "neutral": [],
    }
    
    for i in filtered_data:
      scores = sia.polarity_scores(i)
      # socres = ["compound", "pos", "neg", "neu"]
      if(scores["compound"] >= 0.05):
        print("Result: positive")
        print("-----------------")
        printVADERScores(i)
        statsVADER["positive"].append((i, "pos"))
      if(scores["compound"] <= -0.05):
        print("Result: negative")
        print("-----------------")
        printVADERScores(i)
        statsVADER["negative"].append((i, "neg"))
      if(scores["compound"] > -0.05 and scores["compound"] < 0.05):
        print("Result: neutral")
        print("-----------------")
        printVADERScores(i)
        statsVADER["neutral"].append((i, "neu"))
      print("Sentence:", i, "\n")

    # Prints sentiment statsVADER
    print("\nSpeech Stats: ")
    print("Number of positive   -   ", len(statsVADER["positive"]))
    print("Number of negative   -   ", len(statsVADER["negative"]))
    print("Number of neutral    -   ", len(statsVADER["neutral"]))
    
    allPositiveVADER.append(len(statsVADER["positive"]))
    allNegativeVADER.append(len(statsVADER["negative"]))
    allNeutralVADER.append(len(statsVADER["neutral"]))
    
    #----------TextBlob----------
    statsBlob = {
      "positive": [],
      "negative": [],
      "neutral": [],
    }
      
    for i in filtered_data:
      sentence = TextBlob(i)
      if(sentence.polarity > 0):
        print("Result: positive")
        print("-----------------")
        printBlobScores(sentence)
        statsBlob["positive"].append((sentence, "pos"))
      if(sentence.polarity < 0):
        print("Result: negative")
        print("-----------------")
        printBlobScores(sentence)
        statsBlob["negative"].append((sentence, "neg"))
      if(sentence.polarity == 0):
        print("Result: neutral")
        print("-----------------")
        printBlobScores(sentence)
        statsBlob["neutral"].append((sentence, "neu"))
      print("Sentence:", i, "\n")
    
    
    # Prints sentiment statsVADER
    print("\nSpeech Stats: ")
    print("Number of positive   -   ", len(statsBlob["positive"]))
    print("Number of negative   -   ", len(statsBlob["negative"]))
    print("Number of neutral    -   ", len(statsBlob["neutral"]))
    
    allPositiveBlob.append(len(statsBlob["positive"]))
    allNegativeBlob.append(len(statsBlob["negative"]))
    allNeutralBlob.append(len(statsBlob["neutral"]))

    #----------Topic Model----------
    # Pre-processes each sentence
    processed_sentences = list(map(preprocess, filtered_data))
    allSentences += processed_sentences

    # Creates dictionary using the preprocessed sentences
    dictionary = gensim.corpora.Dictionary(processed_sentences)

    # Filters out tokens that appear:
    # Less than 2 sentences OR
    # More than 0.5 sentences
    # After the first 2, keep only the first 100,000 most frequent tokens
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)

    # For each document we create a dictionary reporting how many words and how many times those words appear
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]

    # Train model using Bag of Words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=2, workers=2)
    
    topicList = []
    for idx, topic in lda_model.print_topics(-1):
      topicList.append(topic)
    
    allTopicLists.append(topicList)
    
    wordsInSpeech = []
    stop_words = list(set(nltk.corpus.stopwords.words("english")))
    for i in filtered_data:
      cleaned_words = re.sub(r'[^(a-zA-Z)\s]','', i)
      tokenized_words = nltk.word_tokenize(cleaned_words)
      stopped_words = [w for w in tokenized_words if not w in stop_words]
      pos = nltk.pos_tag(stopped_words)
      for w in pos:
        wordsInSpeech.append(w[0].lower())
      
    wordsInSpeech = nltk.FreqDist(wordsInSpeech)
    speechTop10 = list(wordsInSpeech.keys())[:10]
    allTop10Words.append(speechTop10)


  # Creates dictionary using ALL the preprocessed sentences in the entire corpus
  dictionary = gensim.corpora.Dictionary(allSentences)

  # Filters out tokens that appear:
  # Less than 2 sentences OR
  # More than 0.5 sentences
  # After the first 2, keep only the first 100,000 most frequent tokens
  dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)

  # For each document we create a dictionary reporting how many words and how many times those words appear
  bow_corpus = [dictionary.doc2bow(sentence) for sentence in allSentences]

  # Train model using Bag of Words
  lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=2, workers=2)

  for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} Words: {}'.format(idx, topic))



  #----------CSV----------
  csv_data = []
  count  = 0
  for i in presidents:
    csv_dict = {
      "President": i[0],
      "Year": i[1],
      "vaderPositive": allPositiveVADER[count],
      "vaderNegative": allNegativeVADER[count],
      "vaderNeutral": allNeutralVADER[count],
      "blobPositive": allPositiveBlob[count],
      "blobNegative": allNegativeBlob[count],
      "blobNeutral": allNeutralBlob[count],
      "topic0": allTopicLists[count][0],
      "topic1": allTopicLists[count][1],
      "topic2": allTopicLists[count][2],
      "topic3": allTopicLists[count][3],
      "top10": allTop10Words[count],
    }
    count = count + 1
    csv_data.append(csv_dict)
    
    
  f = open('generated.csv', 'w', encoding='utf-8')
  writer = csv.writer(f)
  cols = [ 'President', 'Year', 'vaderPositive', 'vaderNegative', 'vaderNeutral', 'blobPositive', 'blobNegative', 'blobNeutral', 'topic0', 'topic1', 'topic2', 'topic3', 'top10']
  writer.writerow(cols)
  for i in csv_data:
      writer.writerow(i.values())

  print("Finished.\n")