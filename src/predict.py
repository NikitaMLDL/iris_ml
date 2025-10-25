from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["features"]).reshape(1, -1)
    pred = model.predict(data)[0]
    return jsonify({"prediction": int(pred)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
