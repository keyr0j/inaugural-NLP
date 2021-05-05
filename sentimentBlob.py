from textblob import TextBlob
import nltk

# Input speech
text = open("speeches/biden.txt", 'r', encoding='utf-8')
data = nltk.sent_tokenize(text.read())

filtered_data = []
for sentence in data:
    if (sentence.rstrip() != ""):
        filtered_data.append(sentence.rstrip())

stats = {
  "positive": [],
  "negative": [],
  "neutral": [],
}

# Prints all of the scores of the sentences using TextBlob
def printAllScores(sentence):
  scores  = sentence.sentiment
  print("polarity:      ", scores[0])
  print("subjectivity:  ", scores[1])

# Print results
for i in filtered_data:
  sentence = TextBlob(i)
  if(sentence.polarity > 0):
    print("Result: positive")
    print("-----------------")
    printAllScores(sentence)
    stats["positive"].append((sentence, "pos"))
  if(sentence.polarity < 0):
    print("Result: negative")
    print("-----------------")
    printAllScores(sentence)
    stats["negative"].append((sentence, "neg"))
  if(sentence.polarity == 0):
    print("Result: neutral")
    print("-----------------")
    printAllScores(sentence)
    stats["neutral"].append((sentence, "neu"))
  print("Sentence:", i, "\n")

# Prints sentiment stats
print("\nSpeech Stats: ")
print("Number of positive   -   ", len(stats["positive"]))
print("Number of negative   -   ", len(stats["negative"]))
print("Number of neutral    -   ", len(stats["neutral"]))