import joblib
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path("data/gender/mfcc_features.pkl")
MODEL_DIR = Path("models/gender")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "gender_svm.pkl"


print("========== GENDER MODEL TRAINING ==========")


# ============================================================
# LOAD MFCC DATA
# ============================================================

print("\nLoading MFCC features...")

data = joblib.load(DATA_FILE)

X = data["X"]
y = data["y"]

print("Features:", X.shape)
print("Labels:", y.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# TRAIN SVM
# ============================================================

print("\n========== TRAINING SVM ==========")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train_scaled, y_train)

print("Training completed!")


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_scaled)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== RESULTS ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Male", "Female"]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nMatrix format:")
print("[[Male predicted Male, Male predicted Female]")
print(" [Female predicted Male, Female predicted Female]]")


# ============================================================
# SAVE MODEL + SCALER
# ============================================================

model_data = {
    "model": model,
    "scaler": scaler
}

joblib.dump(model_data, MODEL_FILE)

print("\n========== SAVED ==========")

print("Model saved at:")
print(MODEL_FILE)

print("============================")

