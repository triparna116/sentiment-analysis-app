from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        text = request.form["text"]
        data = vectorizer.transform([text])
        prediction = model.predict(data)[0]
    return render_template("index.html", prediction=prediction)

@app.route("/bulk-upload", methods=["GET", "POST"])
def bulk_upload():
    predictions = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            df = pd.read_csv(file)
            texts = df["Text"]
            data = vectorizer.transform(texts)
            preds = model.predict(data)
            df["Prediction"] = preds
            predictions = df.to_dict(orient="records")
    return render_template("bulk_upload.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
