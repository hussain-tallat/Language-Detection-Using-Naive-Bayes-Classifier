# Language-Detection-Using-Naive-Bayes-Classifier

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
    plt.show()def detect_mixed_languages(text):
    if re.search('[а-яА-Я]', text) and re.search('[a-zA-Z]', text):
        return True
    return False
user = input("Enter a text: ")
if len(user.strip()) < 3:
    print("Text too short to detect language. Please enter a longer sentence.")
else:
    # ⚠ Warn if mixed languages detected
    if detect_mixed_languages(user):
        print("⚠ Warning: Your input contains multiple language scripts. Prediction might be inaccurate.")
 user_data = cv.transform([user]).toarray()
 user_data = cv.transform([user]).toarray()
output = model.predict(user_data)
probabilities = model.predict_proba(user_data)
confidence = np.max(probabilities) * 100print(f"\nPredicted Language: {output[0]}")
print(f"Confidence: {confidence:.2f}%")
 plot_prediction_probabilities(probabilities, model.classes_)
with open("prediction_log.txt", "a", encoding="utf-8") as file:
        file.write(f"Text: {user}\nPredicted: {output[0]}\nConfidence: {confidence:.2f}%\n\n")
