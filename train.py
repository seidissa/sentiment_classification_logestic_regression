# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load the IMDb dataset
data_path = "IMDB_Dataset.csv"
df = pd.read_csv(data_path)

# Map sentiment to binary labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

# Build the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.95)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, 'model.joblib')
print("✅ Model saved to model.joblib")
