# src/test_model.py

import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Step 1: Paths
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_rf_lr.pkl"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preventive_care_preprocessed.csv"  # Or a separate test CSV

# -------------------------------
# Step 2: Load the Saved Model
# -------------------------------
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# Step 3: Load Test Data
# -------------------------------
df = pd.read_csv(TEST_DATA_PATH)
print("✅ Test data loaded successfully!")
print(df.head())

# -------------------------------
# Step 4: Prepare Features and Target
# -------------------------------
target_column = "outcome"
X_test = df.drop(target_column, axis=1)
y_test = df[target_column].apply(lambda x: 1 if x > 0 else 0)

# -------------------------------
# Step 5: Scale Features
# -------------------------------
# IMPORTANT: Use the same scaling as training
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # Replace with loaded scaler if saved during training

# -------------------------------
# Step 6: Make Predictions
# -------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy on test set: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Step 8: Save Predictions (Optional)
# -------------------------------
df['predicted'] = y_pred
df['probability'] = y_prob
output_path = PROJECT_ROOT / "data" / "processed" / "predictions.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Predictions saved to: {output_path}")
