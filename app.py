import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("haleine_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    mq3 = data["mq3"]
    mq135 = data["mq135"]
    temp = data["temp"]
    humidity = data["humidity"]

    X = np.array([[mq3, mq135, temp, humidity]])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    return jsonify({
        "prediction": str(prediction),
        "probability": float(probability * 100),
        "healthScore": int(probability * 100)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)