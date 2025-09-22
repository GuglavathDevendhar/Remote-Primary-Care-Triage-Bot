from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(name)

# -------------------------------
# Paths to model and scaler
# -------------------------------
PROJECT_ROOT = Path(file).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_rf_lr.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

# -------------------------------
# Load trained model
# -------------------------------
diabetes_model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# Load saved scaler
# -------------------------------
scaler = joblib.load(SCALER_PATH)
print("✅ Scaler loaded successfully!")

# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html", prediction=None)

# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read input from form
        features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["diabetespedigreefunction"]),
            float(request.form["age"])
        ]

        # Convert to 2D array
        input_array = np.array(features).reshape(1, -1)

        # Scale input using saved scaler
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = diabetes_model.predict(input_scaled)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"❌ Error in prediction: {e}"

# -------------------------------
# Run Flask App
# -------------------------------
if name == "main":
    app.run(debug=True)