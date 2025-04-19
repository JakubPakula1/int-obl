import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import string

# Remove punctuation from the text

nltk.download('wordnet')
nltk.download('omw-1.4')
# Wczytanie tekstu
with open("artykul.txt", "r", encoding="utf-8") as fs:
    text = fs.read()

text_no_punctuation = text.translate(str.maketrans("", "", string.punctuation + "-"))

# Tokenize the cleaned text
tokens = nltk.word_tokenize(text_no_punctuation)
print(len(tokens))
# 2308
# Usunięcie standardowych stop-words
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(len(filtered_tokens))
# 1315
# Ręczne dodanie dodatkowych słów do stop-words
additional_stopwords = ["could", "would", "also", "however", "may", "might", "one", "two"]
stop_words.update(additional_stopwords)

# Ponowne filtrowanie
filtered_tokens = [word for word in filtered_tokens if word.lower() not in additional_stopwords]

# Wyświetlenie liczby słów po usunięciu dodatkowych stop-words
# print(len(filtered_tokens))
# 1279

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize tokens
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Display the number of words after lemmatization
print(len(lemmatized_tokens))
# 1279
# Create a word count vector
word_counts = Counter(lemmatized_tokens)

# Get the 10 most common words
most_common_words = word_counts.most_common(10)

# Separate words and their counts for plotting
words, counts = zip(*most_common_words)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Generowanie chmury tagów
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Wyświetlenie chmury tagów
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Ukrycie osi
plt.title('Word Cloud')
plt.show()