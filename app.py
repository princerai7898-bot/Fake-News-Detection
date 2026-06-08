from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    data = vectorizer.transform([news])

    prediction = model.predict(data)

    if prediction[0] == 0:
        result = "❌ FAKE NEWS"
    else:
        result = "✅ REAL NEWS"

    return render_template(
        'index.html',
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)