# predict.py
import sys
import joblib

# Load the trained model
model = joblib.load('model.joblib')

if len(sys.argv) < 2:
    print("Usage: python predict.py \"Your movie review text here.\"")
    sys.exit(1)

review = sys.argv[1]

# Predict the sentiment
proba = model.predict_proba([review])[0]
label = model.predict([review])[0]
sentiment = 'positive' if label == 1 else 'negative'
confidence = proba[label]

print(f"Prediction: {sentiment} ({confidence * 100:.2f}% confidence)")
