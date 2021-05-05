# Imports
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


# data
presidents = ['arthur', 'biden', 'bush', 'carter', 
             'clinton', 'fdr', 'jackson', 'jefferson', 
             'kennedy', 'lincoln', 'nixon', 'obama', 
             'polk', 'roosevelt', 'truman', 'trump',
             'washington', 'wilson']


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


allPositiveVADER = []
allNegativeVADER = []
allNeutralVADER = []
allPositiveBlob = []
allNegativeBlob = []
allNeutralBlob = []

for president in presidents:
  print(president)
  data = []
  file = "speeches/" + president + ".txt";
  # Input speech
  text = open(file, 'r', encoding='utf-8')
  data = nltk.sent_tokenize(text.read())
  filtered_data = []
  for sentence in data:
    if (sentence.rstrip() != ""):
        filtered_data.append(sentence.rstrip())
        
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


csv_data = []
count  = 0
for i in presidents:
  csv_dict = {
    "President": i,
    "Year": "N/A",
    "vaderPositive": allPositiveVADER[count],
    "vaderNegative": allNegativeVADER[count],
    "vaderNeutral": allNeutralVADER[count],
    "blobPositive": allPositiveBlob[count],
    "blobNegative": allNegativeBlob[count],
    "blobNeutral": allNeutralBlob[count],
    "topics": "N/A",
    "top10": "N/A",
  }
  count = count + 1
  csv_data.append(csv_dict)
  
  
f = open('generated.csv', 'w', encoding='utf-8')
writer = csv.writer(f)
cols = [ 'President', 'Year', 'vaderPositive', 'vaderNegative', 'vaderNeutral', 'blobPositive', 'blobNegative', 'blobNeutral', 'topics', 'top10']
writer.writerow(cols)
for i in csv_data:
    writer.writerow(i.values())

print("Finished.\n")