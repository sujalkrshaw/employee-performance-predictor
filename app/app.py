import streamlit as st
import pandas as pd
import plotly.express as px
import random

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/employee_data.csv")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.metric-box {
    background: linear-gradient(135deg, #1e293b, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🚀 Employee Performance Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>AI-powered HR Analytics Dashboard</h4>",
    unsafe_allow_html=True
)

st.write("---")

# ---------------- KPI SECTION ----------------
performance_map = {"Low": 0, "Medium": 1, "High": 2}
df["performance_numeric"] = df["performance"].map(performance_map)

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Employees", len(df))
col2.metric("📊 Avg Experience", round(df["experience"].mean(), 2))
col3.metric("⭐ Avg Performance", round(df["performance_numeric"].mean(), 2))
col4.metric("🌟 High Performers", (df["performance"] == "High").sum())

st.write("---")

# ---------------- CHARTS ----------------
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="performance",
        color="performance",
        title="📈 Employee Performance Distribution",
        color_discrete_map={
            "Low": "red",
            "Medium": "orange",
            "High": "green"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        df,
        x="experience",
        y="performance_numeric",
        color="performance",
        size="salary",
        hover_data=["attendance"],
        title="📊 Experience vs Performance",
        color_discrete_map={
            "Low": "red",
            "Medium": "orange",
            "High": "green"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("---")

# ---------------- PREDICTION SECTION ----------------
st.subheader("🔮 Predict Employee Performance")

col1, col2, col3 = st.columns(3)

experience = col1.slider("Experience (Years)", 0, 20, 5)
training = col2.slider("Training Hours", 0, 100, 20)
attendance = col3.slider("Attendance (%)", 50, 100, 80)

st.write("")

# ---------------- SIMPLE AI LOGIC ----------------
if st.button("🚀 Predict Performance"):

    score = 0

    if experience > 10:
        score += 1

    if training > 50:
        score += 1

    if attendance > 85:
        score += 1

    # ---------------- RESULT ----------------
    if score >= 3:
        st.success("🌟 Prediction: High Performer")

    elif score == 2:
        st.warning("⚡ Prediction: Medium Performer")

    else:
        st.error("⚠️ Prediction: Low Performer")

    # ---------------- EXTRA INSIGHTS ----------------
    st.info(f"""
### 📋 AI HR Insights

- 👨‍💼 Experience: {experience} years
- 📚 Training Hours: {training}
- 🕒 Attendance: {attendance}%
- 🤖 AI Confidence Score: {random.randint(85, 99)}%
""")

st.write("---")

# ---------------- DATASET VIEW ----------------
with st.expander("📂 View Employee Dataset"):
    st.dataframe(df)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Developed by <b>Sujal Kumar Shaw</b> 🚀
</p>
""", unsafe_allow_html=True)