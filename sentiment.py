from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Input speech
data = []
for sentence in open("speeches/trump.txt", 'r', encoding='utf-8'):
  data.append(sentence)

filtered_data = []
for sentence in data:
    if (sentence.rstrip() != ""):
        filtered_data.append(sentence.rstrip())

sia = SentimentIntensityAnalyzer()
stats = {
  "positive": [],
  "negative": [],
}

# Prints all of the scores of the sentences using Sentiment Intensity Analyzer
def printAllScores(sentence):
  scores  = sia.polarity_scores(sentence)
  print("positive:  ", scores["pos"])
  print("negative:  ", scores["neg"])
  print("neutral:   ", scores["neu"])


for i in filtered_data:
  scores = sia.polarity_scores(i)
  # socres = ["compound", "pos", "neg", "neu"]
  if(scores["compound"] > 0):
    print("Result: positive")
    print("-----------------")
    printAllScores(i)
    stats["positive"].append((i, "pos"))
  if(scores["compound"] < 0):
    print("Result: negative")
    print("-----------------")
    printAllScores(i)
    stats["negative"].append((i, "neg"))
  
  print("Sentence:", i, "\n")
  

print("\nSpeech Stats: ")
print("Number of positive - ", len(stats["positive"]))
print("Number of negative - ", len(stats["negative"]))