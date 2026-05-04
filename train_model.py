"""
Train a Gradient Boosting classifier to predict flight delays.

Output: 
  - delay_model.pkl - trained model
  - model_features.pkl - feature columns and encoders
  - model_metrics.json - test set performance
"""
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

print("📊 Loading flight data...")
df = pd.read_csv('/tmp/flight_chunk.csv')
print(f"   Loaded {len(df):,} flights")

# Drop cancelled flights — we're predicting delays, not cancellations
df = df[df['cancelled'] == 0].copy()
df = df.dropna(subset=['dep_delay'])
print(f"   After dropping cancelled/null: {len(df):,}")

# ─── TARGET: Was the flight delayed >15 minutes? ──────────────────────────────
df['delayed'] = (df['dep_delay'] > 15).astype(int)
print(f"   Delay rate in dataset: {df['delayed'].mean()*100:.1f}%")

# ─── FEATURES ──────────────────────────────────────────────────────────────────
df['dep_hour'] = (df['crs_dep_time'] // 100).astype(int)
df['dep_minute'] = (df['crs_dep_time'] % 100).astype(int)
df['day_of_month'] = df['day_of_month'].astype(int)
df['day_of_week'] = df['day_of_week'].astype(int)

feature_cols_categorical = ['op_unique_carrier', 'origin', 'dest']
feature_cols_numeric = ['month', 'day_of_month', 'day_of_week', 'dep_hour', 'distance']

# Encode categoricals
encoders = {}
for col in feature_cols_categorical:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X_cols = [f'{c}_enc' for c in feature_cols_categorical] + feature_cols_numeric
X = df[X_cols]
y = df['delayed']

print(f"\n🎯 Features: {X_cols}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# ─── TRAIN/TEST SPLIT ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📦 Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─── TRAIN MODEL ───────────────────────────────────────────────────────────────
print("\n🧠 Training Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=20,
    random_state=42,
    verbose=0,
)
model.fit(X_train, y_train)
print("   ✓ Model trained")

# ─── EVALUATE ──────────────────────────────────────────────────────────────────
print("\n📈 Evaluating on test set...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"   Accuracy: {acc:.3f}")
print(f"   AUC-ROC:  {auc:.3f}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))

cm = confusion_matrix(y_test, y_pred)
print("   Confusion Matrix:")
print(f"            Predicted")
print(f"            On-Time  Delayed")
print(f"   Actual  On-Time   {cm[0][0]:>6}  {cm[0][1]:>6}")
print(f"            Delayed   {cm[1][0]:>6}  {cm[1][1]:>6}")

# ─── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
importance = pd.DataFrame({
    'feature': X_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🏆 Feature Importance:")
for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"   {row['feature']:25} {row['importance']:.3f} {bar}")

# ─── SAVE MODEL ARTIFACTS ──────────────────────────────────────────────────────
print("\n💾 Saving model...")
with open('/tmp/FlightRiskAI/delay_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('/tmp/FlightRiskAI/model_features.pkl', 'wb') as f:
    pickle.dump({
        'feature_cols': X_cols,
        'categorical_cols': feature_cols_categorical,
        'numeric_cols': feature_cols_numeric,
        'encoders': encoders,
    }, f)

metrics = {
    'accuracy': round(acc, 3),
    'auc_roc': round(auc, 3),
    'training_size': len(X_train),
    'test_size': len(X_test),
    'overall_delay_rate': round(float(y.mean()), 3),
    'feature_importance': importance.set_index('feature')['importance'].round(3).to_dict(),
    'model_type': 'GradientBoostingClassifier',
    'n_estimators': 200,
    'max_depth': 5,
}
with open('/tmp/FlightRiskAI/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("   ✓ delay_model.pkl")
print("   ✓ model_features.pkl")
print("   ✓ model_metrics.json")
print("\n✅ Done! Model ready for app.")
