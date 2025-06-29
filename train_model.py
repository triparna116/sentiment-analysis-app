import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
texts = [
    "I love this product!",
    "This is terrible.",
    "Absolutely fantastic experience.",
    "Worst thing I ever bought.",
    "Very happy with the purchase.",
    "I hate it."
]

labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train a classifier
model = MultinomialNB()
model.fit(X, labels)

# Save the model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training complete. Files saved.")

