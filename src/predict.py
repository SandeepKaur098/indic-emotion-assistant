# predict.py

import joblib
import numpy as np
from src.features import extract_features

def predict_emotion(audio_path, model_path="models/telugu_emotion_model.pkl"):
    # Load model
    model = joblib.load(model_path)

    # Extract features
    features = extract_features(audio_path)

    # Reshape for prediction
    features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]

    return prediction

# Example usage (only for testing locally)
if __name__ == "__main__":
    result = predict_emotion("data/sample_telugu.wav")
    print("Predicted Emotion:", result)
