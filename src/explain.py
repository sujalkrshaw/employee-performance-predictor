import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/model.pkl")

# Load data
df = pd.read_csv("data/employee_data.csv")
X = df.drop("performance", axis=1)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Create plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)

# Save plot
plt.savefig("outputs/shap_summary.png")

print("SHAP plot saved in outputs folder")
