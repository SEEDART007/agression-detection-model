from flask import Flask, request, jsonify
import torch
import librosa
import numpy as np
from model import EmotionModel
import joblib
import os

app = Flask(__name__)

# Load model and label encoder
model = EmotionModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

le = joblib.load("label_encoder.pkl")

def extract_features(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc.mean(axis=1)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    try:
        features = extract_features(filepath)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(features)
            pred_class = torch.argmax(output, dim=1).item()
            emotion = le.inverse_transform([pred_class])[0]

        os.remove(filepath)
        return jsonify({"emotion": emotion})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run it
if __name__ == "__main__":
    app.run(debug=True, port=5000)
