import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("models/model.pkl")
df = pd.read_csv("data/employee_data.csv")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🚀 Employee Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI-powered HR Analytics Dashboard</h4>", unsafe_allow_html=True)

st.write("---")

# ---------------- KPI ----------------
performance_map = {"Low": 0, "Medium": 1, "High": 2}
df["performance_numeric"] = df["performance"].map(performance_map)

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Employees", len(df))
col2.metric("📊 Avg Experience", round(df["experience"].mean(), 2))
col3.metric("⭐ Performance Score", round(df["performance_numeric"].mean(), 2))
col4.metric("🌟 High Performers", (df["performance"] == "High").sum())

st.write("---")

# ---------------- CHARTS ----------------
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="performance",
        color="performance",
        title="Performance Distribution",
        color_discrete_map={
            "Low": "red",
            "Medium": "blue",
            "High": "green"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        df,
        x="experience",
        y="performance_numeric",
        color="performance",
        title="Experience vs Performance",
        color_discrete_map={
            "Low": "red",
            "Medium": "blue",
            "High": "green"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

st.write("---")

# ---------------- PREDICTION ----------------
st.subheader("🔮 Predict Employee Performance")

col1, col2, col3 = st.columns(3)

experience = col1.slider("Experience (Years)", 0, 20, 5)
training = col2.slider("Training Hours", 0, 100, 20)
attendance = col3.slider("Attendance (%)", 50, 100, 80) / 100

input_data = pd.DataFrame({
    "experience": [experience],
    "projects": [5],
    "training_hours": [training],
    "attendance": [attendance],
    "feedback_score": [3],
    "salary": [50000]
})

if st.button("🚀 Predict Now"):
    pred = model.predict(input_data)[0]
    labels = {0: "Low", 1: "Medium", 2: "High"}

    if pred == 2:
        st.success(f"🌟 High Performer")
    elif pred == 1:
        st.warning(f"⚡ Medium Performer")
    else:
        st.error(f"⚠️ Low Performer")

st.write("---")

# ---------------- DATA ----------------
with st.expander("📂 View Dataset"):
    st.dataframe(df.head())

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;'>Developed by <b>Sujal Kumar Shaw</b> 🚀</p>
""", unsafe_allow_html=True)
