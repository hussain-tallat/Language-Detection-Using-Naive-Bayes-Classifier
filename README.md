# Language-Detection-Using-Naive-Bayes-Classifier
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv('/content/language .csv')
data.isnull().sum()
data['language'].value_counts()
data.dtypes
x = np.array(data['Text'])
y = np.array(data['language'])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model Accuracy:", model.score(X_test, y_test))
# graph 
plot_language_distribution()
def plot_prediction_probabilities(probabilities, classes):
    plt.figure(figsize=(8, 5))
    plt.bar(classes, probabilities[0], color='lightgreen')
    plt.title("Prediction Probabilities")
    plt.xlabel("Language")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
# first functionality ( short language detection
if len(user.strip()) < 3:
    print("Text too short to detect language. Please enter a longer sentence.")

# secand functionality ( save all in file )

with open("prediction_log.txt", "a", encoding="utf-8") as file:
        file.write(f"Text: {user}\nPredicted: {output[0]}\nConfidence: {confidence:.2f}%\n\n"
