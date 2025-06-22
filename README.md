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
def plot_language_distribution():
    lang_counts = data['language'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(lang_counts.index, lang_counts.values, color='skyblue')
    plt.title("Language Distribution in Dataset")
    plt.xlabel("Language")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
