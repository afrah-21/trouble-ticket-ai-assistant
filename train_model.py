import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


CSV_FILE = "trouble_ticket_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    encodings = ["utf-8", "latin1", "cp1252"]

    for enc in encodings:
        try:
            df = pd.read_csv(CSV_FILE, encoding=enc, low_memory=False)
            print(f"CSV loaded using encoding: {enc}")
            return df
        except Exception as e:
            print(f"Failed with {enc}: {e}")

    raise Exception("Could not read CSV file")


def clean_data(df):
    df.columns = df.columns.str.strip().str.upper()

    required_cols = [
        "USERID",
        "FAULTTYPE",
        "SUBFAULTTYPE",
        "SATISFACTIONSCORE",
        "FEEDBACK_COMMENTS",
        "TT_CREATION_TIME"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["TT_CREATION_TIME"] = pd.to_datetime(
        df["TT_CREATION_TIME"],
        errors="coerce",
        dayfirst=True
    )

    df["SATISFACTION_NUMERIC"] = (
        df["SATISFACTIONSCORE"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    df["SATISFACTION_NUMERIC"] = df["SATISFACTION_NUMERIC"].fillna(
        df["SATISFACTION_NUMERIC"].median()
    )

    text_cols = ["USERID", "FAULTTYPE", "SUBFAULTTYPE", "FEEDBACK_COMMENTS"]

    for col in text_cols:
        df[col] = df[col].astype(str).fillna("Unknown").str.strip()

    df = df.dropna(subset=["TT_CREATION_TIME"])
    df = df.sort_values(["USERID", "TT_CREATION_TIME"])

    return df


def create_next_month_target(df):
    """
    Target:
    1 = same user made another complaint within next 30 days
    0 = no complaint within next 30 days
    """

    df["NEXT_TICKET_TIME"] = df.groupby("USERID")["TT_CREATION_TIME"].shift(-1)

    df["DAYS_TO_NEXT_TICKET"] = (
        df["NEXT_TICKET_TIME"] - df["TT_CREATION_TIME"]
    ).dt.days

    df["COMPLAINT_NEXT_MONTH"] = np.where(
        (df["DAYS_TO_NEXT_TICKET"] > 0) &
        (df["DAYS_TO_NEXT_TICKET"] <= 30),
        1,
        0
    )

    return df


def encode_features(df):
    features = [
        "USERID",
        "FAULTTYPE",
        "SUBFAULTTYPE",
        "FEEDBACK_COMMENTS",
        "SATISFACTION_NUMERIC"
    ]

    X = df[features].copy()
    y = df["COMPLAINT_NEXT_MONTH"]

    label_encoders = {}

    categorical_cols = [
        "USERID",
        "FAULTTYPE",
        "SUBFAULTTYPE",
        "FEEDBACK_COMMENTS"
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    return X, y, label_encoders, features


def train_model():
    print("Loading dataset...")
    df = load_dataset()

    print("Cleaning dataset...")
    df = clean_data(df)

    print("Creating prediction target...")
    df = create_next_month_target(df)

    print("Encoding features...")
    X, y, label_encoders, features = encode_features(df)

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining XGBoost model...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_complaint_model.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "model_features.pkl"))

    print("\nModel saved successfully in models/ folder")
    print("Saved files:")
    print("models/xgboost_complaint_model.pkl")
    print("models/label_encoders.pkl")
    print("models/model_features.pkl")


if __name__ == "__main__":
    train_model()