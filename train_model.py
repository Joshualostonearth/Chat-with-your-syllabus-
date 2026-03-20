import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# --- LOAD NORMAL DATA ONLY ---
X_normal = np.load("data/processed/X_normal.npy")

# --- TRAIN ---
print("Training Isolation Forest...")
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(X_normal)
print("[OK] Model trained")

# --- CHECK IF TRAINED ---
pred = model.predict(X_normal)
normal_correct = (pred == 1).sum()
print(f"Sanity check: {normal_correct} / {len(X_normal)} normal traces correctly identified")

# --- SAVE ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/isolation_forest.pkl")
print("[OK] Model saved to models/isolation_forest.pkl")