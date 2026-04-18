import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("data/employee_data.csv")

# Features and target
X = df.drop("performance", axis=1)
y = df["performance"]

# Convert labels → numbers
y = y.map({"Low": 0, "Medium": 1, "High": 2})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/model.pkl")

print("\nModel saved successfully!")