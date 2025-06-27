# app.py
from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'review' not in data:
        return jsonify({'error': 'Please provide a "review" field in JSON body.'}), 400

    review = data['review']
    pred_proba = model.predict_proba([review])[0]
    pred_label = model.predict([review])[0]

    sentiment = 'positive' if pred_label == 1 else 'negative'
    confidence = pred_proba[pred_label]

    return jsonify({
        'sentiment': sentiment,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
