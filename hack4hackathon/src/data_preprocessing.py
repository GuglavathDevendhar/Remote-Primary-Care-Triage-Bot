# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# -------------------------------
# Step 1: Define paths using Pathlib
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILE = "preventive_care.csv"
RAW_DATA_PATH = RAW_DIR / EXPECTED_FILE
PROCESSED_DATA_PATH = PROCESSED_DIR / "preventive_care_preprocessed.csv"

# -------------------------------
# Step 2: Handle file selection
# -------------------------------
if not RAW_DATA_PATH.exists():
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    print("Found the following CSV files in data/raw:")
    for i, f in enumerate(csv_files, 1):
        print(f"{i}. {f.name}")
    choice = input(f"Enter the number of the file to use [1-{len(csv_files)}]: ")
    RAW_DATA_PATH = csv_files[int(choice) - 1]

print(f"📂 Using file: {RAW_DATA_PATH.name}")

# -------------------------------
# Step 3: Load Dataset
# -------------------------------
df = pd.read_csv(RAW_DATA_PATH)
print("✅ Raw dataset loaded successfully!")

# -------------------------------
# Step 4: Inspect Dataset
# -------------------------------
print("\nFirst 5 rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# Step 5: Standardize column names
# -------------------------------
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# -------------------------------
# Step 6: Handle missing values
# -------------------------------
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Step 7: Remove duplicates
# -------------------------------
df = df.drop_duplicates()
print(f"\n✅ Dataset shape after removing duplicates: {df.shape}")

# -------------------------------
# Step 8: Encode categorical variables
# -------------------------------
cat_cols = df.select_dtypes(include="object").columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# -------------------------------
# Step 9: Scale numeric columns
# -------------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# Step 10: Save processed data
# -------------------------------
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\n🎯 Preprocessed dataset saved to: {PROCESSED_DATA_PATH}")
