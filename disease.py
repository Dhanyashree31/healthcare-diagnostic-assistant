import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load existing dataset
df = pd.read_csv("health.csv")

# Generate 50 synthetic patients
np.random.seed(42)
new_data = pd.DataFrame({
    "Age": np.random.randint(18, 90, 50),
    "BloodPressure": np.random.randint(80, 200, 50),
    "Cholesterol": np.random.randint(150, 300, 50),
    "BMI": np.round(np.random.uniform(15, 40, 50), 1),
    "Glucose": np.random.randint(70, 200, 50),
    "HeartRate": np.random.randint(50, 120, 50),
    "DiseaseRisk": np.random.choice(["Low", "Medium", "High"], 50)
})

# Append to existing dataset
df = pd.concat([df, new_data], ignore_index=True)

# Save back to CSV
df.to_csv("health.csv", index=False)

print("✅ Added 50 new synthetic patient records!")
# Background CSS with Unsplash image + styled heading
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?q=80&w=1332&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.95);
    border-right: 1px solid #ccc;
}

/* Sidebar text color */
[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Radio button labels */
div[role="radiogroup"] > label > div {
    color: #000000 !important;
    font-weight: 600;
}

/* Selected radio button highlight */
div[role="radiogroup"] > label[data-checked="true"] > div {
    color: #ffffff !important;
    background-color: #000000 !important;
    border-radius: 5px;
    padding: 4px 8px;
}

/* Heading banner */
.custom-heading {
    background: linear-gradient(90deg, #00416A, #E4E5E6);
    padding: 20px;
    border-radius: 8px;
    color: white;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
    font-size: 28px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Heading with gradient banner
st.markdown(
    "<div class='custom-heading'>🩺 AI-Powered Healthcare Diagnostic Assistant</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px; color:#333;'>Predict disease risk and receive lifestyle recommendations instantly.</p>",
    unsafe_allow_html=True
)

# Load dataset
df = pd.read_csv("health.csv")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Overview", "📊 Analysis", "🤖 Predictions", "➕ Add Data", "ℹ️ About"])

# Overview
if page == "🏠 Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Age", round(df['Age'].mean(),1))
    col2.metric("Avg BMI", round(df['BMI'].mean(),1))
    col3.metric("High Risk Patients", df[df['DiseaseRisk']=="High"].shape[0])

# Analysis
elif page == "📊 Analysis":
    st.header("Data Analysis")
    feature = st.selectbox("Choose a numeric column:", df.select_dtypes(include=['int64','float64']).columns)
    fig = px.histogram(df, x=feature, nbins=10, title=f"Distribution of {feature}", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.header("Correlation Heatmap")
    corr = df[['Age','BloodPressure','Cholesterol','BMI','Glucose','HeartRate']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Feature Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)

# Predictions
elif page == "🤖 Predictions":
    st.header("Disease Risk Prediction")
    X = df[['Age','BloodPressure','Cholesterol','BMI','Glucose','HeartRate']]
    y = df['DiseaseRisk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully with Accuracy: {acc:.2f}")

    st.subheader("Try Your Own Prediction")
    col1, col2 = st.columns(2)
    age = col1.slider("Age", 18, 90, 30)
    bp = col2.slider("Blood Pressure", 80, 200, 120)
    chol = col1.slider("Cholesterol", 150, 300, 200)
    bmi = col2.slider("BMI", 15, 40, 25)
    glucose = col1.slider("Glucose", 70, 200, 100)
    hr = col2.slider("Heart Rate", 50, 120, 75)

    prediction = model.predict([[age, bp, chol, bmi, glucose, hr]])[0]
    st.success(f"🩺 Predicted Disease Risk: {prediction}")

    st.subheader("💡 Suggested Recommendations")
    if prediction == "Low":
        st.info("✅ Maintain a healthy lifestyle with balanced diet and regular exercise.")
    elif prediction == "Medium":
        st.warning("⚠️ Improve diet, increase physical activity, and monitor blood pressure regularly.")
    else:
        st.error("🚨 High risk detected! Consult a doctor, consider medication, and monitor health closely.")

# Add Data
elif page == "➕ Add Data":
    st.header("Add New Patient Data")
    with st.form("add_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=18, max_value=90, step=1)
        bp = st.number_input("Blood Pressure", min_value=80, max_value=200, step=1)
        chol = st.number_input("Cholesterol", min_value=150, max_value=300, step=1)
        bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, step=0.1)
        glucose = st.number_input("Glucose", min_value=70, max_value=200, step=1)
        hr = st.number_input("Heart Rate", min_value=50, max_value=120, step=1)
        risk = st.selectbox("Disease Risk", ["Low","Medium","High"])
        submitted = st.form_submit_button("Add Patient")
        if submitted:
            new_row = {"Age": age, "BloodPressure": bp, "Cholesterol": chol, "BMI": bmi,
                       "Glucose": glucose, "HeartRate": hr, "DiseaseRisk": risk}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv("data/health.csv", index=False)
            st.success("✅ New patient data added successfully!")
            st.dataframe(df.tail())

# About
elif page == "ℹ️ About":
    st.header("About This Project")
    st.markdown("""
    **AI-Powered Healthcare Diagnostic Assistant**  
    Built with Python (Streamlit, Pandas, Plotly, Scikit-learn).  
    Predicts disease risk based on patient health data and provides lifestyle recommendations.  
    Designed to showcase technical + AI skills in a recruiter-ready format with professional UI.  
    """)
