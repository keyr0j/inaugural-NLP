from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Input speech
data = []
text = open("speeches/biden.txt", 'r', encoding='utf-8')
data = nltk.sent_tokenize(text.read())

filtered_data = []
for sentence in data:
    if (sentence.rstrip() != ""):
        filtered_data.append(sentence.rstrip())


sia = SentimentIntensityAnalyzer()
stats = {
  "positive": [],
  "negative": [],
  "neutral": [],
}


# Prints all of the scores of the sentences using Sentiment Intensity Analyzer
def printAllScores(sentence):
  scores  = sia.polarity_scores(sentence)
  print("positive:  ", scores["pos"])
  print("negative:  ", scores["neg"])
  print("neutral:   ", scores["neu"])

# Print results
for i in filtered_data:
  scores = sia.polarity_scores(i)
  # socres = ["compound", "pos", "neg", "neu"]
  if(scores["compound"] >= 0.05):
    print("Result: positive")
    print("-----------------")
    printAllScores(i)
    stats["positive"].append((i, "pos"))
  if(scores["compound"] <= -0.05):
    print("Result: negative")
    print("-----------------")
    printAllScores(i)
    stats["negative"].append((i, "neg"))
  if(scores["compound"] > -0.05 and scores["compound"] < 0.05):
    print("Result: neutral")
    print("-----------------")
    printAllScores(i)
    stats["neutral"].append((i, "neu"))
  print("Sentence:", i, "\n")
  
# Prints sentiment stats
print("\nSpeech Stats: ")
print("Number of positive   -   ", len(stats["positive"]))
print("Number of negative   -   ", len(stats["negative"]))
print("Number of neutral    -   ", len(stats["neutral"]))