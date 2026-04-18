import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "experience": np.random.randint(1, 20, n),
    "projects": np.random.randint(1, 10, n),
    "training_hours": np.random.randint(5, 100, n),
    "attendance": np.random.uniform(0.6, 1.0, n),
    "feedback_score": np.random.uniform(1, 5, n),
    "salary": np.random.randint(20000, 120000, n),
})

# Performance calculation
score = (
    df["experience"] * 0.4 +
    df["projects"] * 0.3 +
    df["attendance"] * 10 +
    df["feedback_score"] * 2
)

df["performance"] = pd.cut(
    score,
    bins=[0, 15, 25, 50],
    labels=["Low", "Medium", "High"]
)

# 🔥 IMPORTANT LINE
df.to_csv("data/employee_data.csv", index=False)

print("Dataset generated successfully!")