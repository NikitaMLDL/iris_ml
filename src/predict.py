from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Загружаем модель
model = joblib.load("models/model.pkl")

# HTML-шаблон для главной страницы
HTML_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <title>Iris Predictor</title>
    </head>
    <body>
        <h1>Iris Predictor</h1>
        <form method="post" action="/predict_form">
            Sepal length: <input type="text" name="sepal_length"><br>
            Sepal width: <input type="text" name="sepal_width"><br>
            Petal length: <input type="text" name="petal_length"><br>
            Petal width: <input type="text" name="petal_width"><br>
            <input type="submit" value="Predict">
        </form>
        {% if prediction is not none %}
            <h2>Prediction: {{ prediction }}</h2>
        {% endif %}
    </body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]
        data = np.array(features).reshape(1, -1)
        pred = model.predict(data)[0]
        return render_template_string(HTML_TEMPLATE, prediction=int(pred))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {e}")

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        features = request.json.get("features")
        if features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400
        data = np.array(features).reshape(1, -1)
        pred = model.predict(data)[0]
        return jsonify({"prediction": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
