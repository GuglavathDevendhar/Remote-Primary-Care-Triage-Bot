import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preventive_care_preprocessed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded successfully!")

# -------------------------------
# Split Features and Target
# -------------------------------
target_column = "outcome"
X = df.drop(target_column, axis=1).copy()  # raw features
y = df[target_column].copy()
y = y.apply(lambda x: 1 if x > 0 else 0)  # binary 0/1

# -------------------------------
# Feature Scaling
# -------------------------------
num_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # scale features for training
X = pd.DataFrame(X_scaled, columns=num_cols)

# Save scaler
SCALER_PATH = MODELS_DIR / "scaler.pkl"
joblib.dump(scaler, SCALER_PATH, compress=3)
print(f"🎯 Scaler saved to: {SCALER_PATH}")

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Define Models
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# -------------------------------
# Voting Ensemble
# -------------------------------
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr)],
    voting='soft'
)

# -------------------------------
# Train Ensemble
# -------------------------------
ensemble.fit(X_train, y_train)
print("✅ Ensemble model trained successfully!")

# -------------------------------
# Evaluate Model
# -------------------------------
y_pred = ensemble.predict(X_test)
print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f}")

# -------------------------------
# Save Model
# -------------------------------
MODEL_PATH = MODELS_DIR / "ensemble_rf_lr.pkl"
joblib.dump(ensemble, MODEL_PATH, compress=3)
print(f"🎯 Model saved to: {MODEL_PATH}")
