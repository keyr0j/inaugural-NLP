from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt 
import nltk

presidents = [('arthur', 1881), ('biden', 2021), ('bush', 2001), ('carter', 1977),
             ('clinton', 1993), ('fdr', 1933), ('jackson',1829), ('jefferson', 1801),
             ('kennedy', 1961), ('lincoln', 1861), ('nixon', 1969), ('obama', 2008),
             ('polk', 1845), ('roosevelt', 1905), ('truman', 1949), ('trump', 2016),
             ('washington', 1789), ('wilson', 1913)]


words = ''

for president in presidents:
    data = []
    file = "speeches/" + president[0] + ".txt";
    # Input speech
    text = open(file, 'r', encoding='utf-8')
    data = nltk.sent_tokenize(text.read())
    filtered_data = []
    for sentence in data:
      if (sentence.rstrip() != ""):
          filtered_data.append(sentence.rstrip())
          words += sentence.rstrip()

stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(words)


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()