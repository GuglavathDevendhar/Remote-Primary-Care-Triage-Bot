{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d3ff7e8",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 01_data_cleaning.py\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# Step 1: Paths\n",
    "RAW_DATA_PATH = os.path.join('data', 'raw', 'preventive_care.csv')\n",
    "PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'preventive_care_cleaned.csv')\n",
    "\n",
    "# Step 2: Load Dataset\n",
    "df = pd.read_csv(RAW_DATA_PATH)\n",
    "\n",
    "# Step 3: Inspect Dataset\n",
    "print(\"First 5 rows:\")\n",
    "print(df.head())\n",
    "print(\"\\nDataset Info:\")\n",
    "print(df.info())\n",
    "print(\"\\nSummary Statistics:\")\n",
    "print(df.describe())\n",
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Step 4: Handle Missing Values\n",
    "num_cols = df.select_dtypes(include=np.number).columns.tolist()\n",
    "cat_cols = df.select_dtypes(include='object').columns.tolist()\n",
    "\n",
    "# Fill numerical columns with median\n",
    "for col in num_cols:\n",
    "    df[col].fillna(df[col].median(), inplace=True)\n",
    "\n",
    "# Fill categorical columns with mode\n",
    "for col in cat_cols:\n",
    "    df[col].fillna(df[col].mode()[0], inplace=True)\n",
    "\n",
    "# Step 5: Remove duplicates\n",
    "df.drop_duplicates(inplace=True)\n",
    "print(f\"\\nDataset shape after removing duplicates: {df.shape}\")\n",
    "\n",
    "# Step 6: Standardize columns\n",
    "df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]\n",
    "\n",
    "# Step 7: Encode categorical variables\n",
    "df = pd.get_dummies(df, columns=cat_cols, drop_first=True)\n",
    "\n",
    "# Step 8: Save cleaned dataset\n",
    "os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)\n",
    "df.to_csv(PROCESSED_DATA_PATH, index=False)\n",
    "print(f\"\\nCleaned dataset saved to '{PROCESSED_DATA_PATH}'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76753a79",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
