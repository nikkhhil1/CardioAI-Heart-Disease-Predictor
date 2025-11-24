# app.py

# app.py
import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
import sqlite3
from sqlite3 import Connection
from datetime import datetime
import pandas as pd
import plotly.express as px
import hashlib
import os
from io import BytesIO

# -------------------------
# Configuration / Utilities
# -------------------------
DB_PATH = "users_history.db"
MODEL_PATH = "heart_model.pkl"  # <- set to your actual model filename

# IMPORTANT: make FEATURE_NAMES match the exact order and number of features your model was trained on.
# You had 13 earlier; model expects 16. Add the three extra feature names here (replace Extra1..3 with real names if known).
FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain", "BP", "Chol", "FBS",
    "ECG", "Max HR", "ExAng", "ST Depression", "Slope",
    "Vessels", "Thal", "Extra1", "Extra2", "Extra3"
]

# -------- DB helpers ----------
def get_db() -> Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_json TEXT,
            risk_percent REAL,
            category TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str) -> bool:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, hash_password(password), datetime.utcnow().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username: str, password: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return None
    user_id, pw_hash = row
    if pw_hash == hash_password(password):
        return user_id
    return None

def save_history(user_id: int, input_json: str, risk_percent: float, category: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO history (user_id, input_json, risk_percent, category, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, input_json, risk_percent, category, datetime.utcnow().isoformat()))
    conn.commit()

def get_user_history_df(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, input_json, risk_percent, category, created_at FROM history WHERE user_id = ? ORDER BY created_at", (user_id,))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["id", "input_json", "risk_percent", "category", "created_at"])
    return df

# ------------- Init -------------
init_db()

# ------------- Load model -------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please place your trained model there (joblib .pkl).")
    st.stop()

# load model using MODEL_PATH variable
model = joblib.load("heart_model.pkl")
st.write("✅ Loaded model: heart_model.pkl")

# --------- Sidebar : Auth ------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user_id": None, "username": None}

st.sidebar.title("Account")
if not st.session_state.auth["logged_in"]:
    auth_mode = st.sidebar.radio("Login / Register", ["Login", "Register"])
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    if auth_mode == "Register":
        if st.sidebar.button("Create account"):
            if username_input.strip() == "" or password_input.strip() == "":
                st.sidebar.warning("Choose a valid username & password.")
            else:
                ok = create_user(username_input.strip(), password_input.strip())
                if ok:
                    st.sidebar.success("Account created — please login.")
                else:
                    st.sidebar.error("Username taken. Choose another.")
    else:  # Login
        if st.sidebar.button("Login"):
            user_id = authenticate_user(username_input.strip(), password_input.strip())
            if user_id:
                st.session_state.auth["logged_in"] = True
                st.session_state.auth["user_id"] = int(user_id)
                st.session_state.auth["username"] = username_input.strip()
                st.sidebar.success(f"Signed in as {username_input.strip()}")
            else:
                st.sidebar.error("Invalid username or password.")
else:
    st.sidebar.write(f"Signed in: **{st.session_state.auth['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.auth = {"logged_in": False, "user_id": None, "username": None}
        st.experimental_rerun()

# -------- Main App UI ----------
st.title("CardioAI: Heart Disease Predictor ❤️")

st.markdown("""
This app includes:
- Prediction + SHAP explainability  
- Animated / interactive plots of your prediction history (Plotly)  
- User login and saved history (SQLite)  
- Downloadable PDF reports per prediction  
""")

# User Inputs (now 16 inputs matching FEATURE_NAMES)
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=60, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    with col2:
        restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
        slope = st.selectbox("Slope (0–2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    # Extra features required by the model (replace labels with real names if you know them)
    ex1 = st.number_input("Extra Feature 1 (model required)", value=0.0)
    ex2 = st.number_input("Extra Feature 2 (model required)", value=0.0)
    ex3 = st.number_input("Extra Feature 3 (model required)", value=0.0)

    submitted = st.form_submit_button("Predict")

# Build input array in the exact order matching FEATURE_NAMES
input_data = np.array([[age,
                        1 if sex == "Male" else 0,
                        cp,
                        trestbps,
                        chol,
                        fbs,
                        restecg,
                        thalach,
                        exang,
                        oldpeak,
                        slope,
                        ca,
                        thal,
                        ex1,
                        ex2,
                        ex3
                       ]])

# When submitted
if submitted:
    # Compute probability safely
    try:
        # Some classifiers may not implement predict_proba
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1] * 100  # probability of class 1
        else:
            # fallback: use predict and set proba to None or 0/100
            pred = model.predict(input_data)[0]
            proba = None
    except Exception as e:
        st.error("Model predict_proba/predict failed — check model compatibility. Error: " + str(e))
        st.stop()

    # If proba is None (no predict_proba), compute predicted class
    if proba is None:
        pred = model.predict(input_data)[0]
        proba_str = "N/A"
        st.subheader(f"🔍 Prediction: **{int(pred)}** (model has no predict_proba)")
        proba = None
    else:
        st.subheader(f"🔍 Prediction: **{proba:.2f}%** risk")

    # Category & recommendation (if proba available, else use predicted class)
    if proba is not None:
        if proba <= 30:
            category = "Low Risk"
            recommendation = [
                "Maintain a balanced diet",
                "Exercise regularly (150 min/week)",
                "Routine yearly checkups"
            ]
        elif proba <= 60:
            category = "Moderate Risk"
            recommendation = [
                "Reduce salt (<5g/day)",
                "Avoid smoking/alcohol",
                "Exercise 30–45 min daily",
                "Monitor cholesterol & BP every 6 months"
            ]
        else:
            category = "High Risk"
            recommendation = [
                "Immediate medical consultation advised",
                "Follow low-salt & low-fat diet",
                "Avoid stress & heavy exercise without doctor approval",
                "Regular monitoring of BP, glucose & ECG"
            ]
    else:
        # No probability available — use class label
        pred = model.predict(input_data)[0]
        if pred == 1:
            category = "High Risk"
            recommendation = [
                "Immediate medical consultation advised",
            ]
        else:
            category = "Low Risk"
            recommendation = [
                "Maintain a balanced diet",
            ]

    st.markdown(f"**Risk category:** {category}")
    st.markdown("**Recommendations:**")
    for r in recommendation:
        st.write("- " + r)

    # Save history if user logged in
    input_json = str(dict(zip(FEATURE_NAMES, input_data[0].tolist())))
    if st.session_state.auth["logged_in"]:
        save_history(st.session_state.auth["user_id"], input_json, float(proba) if proba is not None else None, category)
        st.success("Saved prediction to your history.")

    # ---- SHAP explainability ----
    st.write("---")
    st.write("### 🔍 Model Explainability (SHAP)")

    try:
        # shap.Explainer works with most model types
        explainer = shap.Explainer(model, feature_names=FEATURE_NAMES)
        shap_values = explainer(input_data)  # returns Explanation object
        # Try a waterfall / bar plot
        try:
            shap.plots.waterfall(shap_values[0], max_display=16)
        except Exception:
            # fallback bar of absolute contributions
            vals = shap_values.values[0] if hasattr(shap_values, "values") else np.array(shap_values)
            df_contrib = pd.DataFrame({"feature": FEATURE_NAMES, "contrib": vals})
            df_contrib = df_contrib.sort_values("contrib", key=lambda x: x.abs(), ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(df_contrib["feature"], df_contrib["contrib"])
            ax.set_xlabel("SHAP value (impact on model output)")
            st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP explainability failed (model type may not be supported or shap not installed). Error: " + str(e))

    # ---- Plotly animated / interactive charts ----
    st.write("---")
    st.write("### 📈 Interactive Dashboard")

    # Get user history (if logged in)
    if st.session_state.auth["logged_in"]:
        df_hist = get_user_history_df(st.session_state.auth["user_id"])
        if df_hist.empty:
            st.info("No past predictions found — your current prediction is shown below.")
            df_display = pd.DataFrame([{
                "created_at": datetime.utcnow().isoformat(),
                "risk_percent": proba if proba is not None else (100 if category == "High Risk" else 0),
                "category": category
            }])
        else:
            df_display = df_hist[["created_at", "risk_percent", "category"]].copy()
            df_display = pd.concat([df_display, pd.DataFrame([{
                "created_at": datetime.utcnow().isoformat(),
                "risk_percent": proba if proba is not None else (100 if category == "High Risk" else 0),
                "category": category
            }])], ignore_index=True)
            df_display["created_at"] = pd.to_datetime(df_display["created_at"])
    else:
        now = datetime.utcnow()
        df_display = pd.DataFrame({
            "created_at": [now],
            "risk_percent": [proba if proba is not None else (100 if category == "High Risk" else 0)],
            "category": [category]
        })
        st.info("Sign up / login to save history and see the animated timeline.")

    try:
        fig = px.line(df_display, x="created_at", y="risk_percent", markers=True, title="Risk % over time")
        fig.update_layout(yaxis_title="Risk (%)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

        df_display_sorted = df_display.sort_values("created_at")
        df_display_sorted["created_at_str"] = df_display_sorted["created_at"].astype(str)
        scatter_fig = px.scatter(df_display_sorted, x="created_at", y="risk_percent",
                                 size="risk_percent", color="category",
                                 animation_frame="created_at_str",
                                 title="Animated Predictions (by timestamp)")
        st.plotly_chart(scatter_fig, use_container_width=True)
    except Exception as e:
        st.warning("Plotly charts failed: " + str(e))

    # ---- PDF Report generation (download) ----
    st.write("---")
    st.write("### 📄 Downloadable Report")

    def create_pdf_bytes(username, input_dict, risk_percent, category, recommendation_list):
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, 800, "Heart Disease Prediction Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 770, f"User: {username if username else 'Anonymous'}")
        c.drawString(50, 750, f"Date (UTC): {datetime.utcnow().isoformat()}")
        c.drawString(50, 730, f"Risk Percentage: {risk_percent:.2f}%" if risk_percent is not None else "Risk Percentage: N/A")
        c.drawString(50, 710, f"Risk Category: {category}")
        c.drawString(50, 690, "Inputs:")
        y = 670
        for k, v in input_dict.items():
            c.drawString(60, y, f"{k}: {v}")
            y -= 16
            if y < 120:
                c.showPage()
                y = 800
        c.drawString(50, y - 10, "Recommendations:")
        y -= 30
        for r in recommendation_list:
            c.drawString(60, y, "- " + r)
            y -= 16
            if y < 120:
                c.showPage()
                y = 800
        c.drawString(50, 80, "Disclaimer: This is informational only and not a medical diagnosis.")
        c.save()
        buffer.seek(0)
        return buffer

    if st.button("Generate & Download PDF Report"):
        username_for_pdf = st.session_state.auth["username"] if st.session_state.auth["logged_in"] else "Anonymous"
        input_dict = dict(zip(FEATURE_NAMES, input_data[0].tolist()))
        pdf_bytes = create_pdf_bytes(username_for_pdf, input_dict, proba if proba is not None else 0.0, category, recommendation)
        st.download_button("Download Report (PDF)", pdf_bytes, file_name="Heart_Disease_Report.pdf", mime="application/pdf")

# ---- If logged in -> show full history table and options ----
if st.session_state.auth["logged_in"]:
    st.write("---")
    st.write("## 📚 Your Prediction History")
    user_id = st.session_state.auth["user_id"]
    df_hist_full = get_user_history_df(user_id)
    if df_hist_full.empty:
        st.info("No saved predictions yet.")
    else:
        st.dataframe(df_hist_full[["created_at", "risk_percent", "category"]].sort_values("created_at", ascending=False))

        if st.button("Clear my history"):
            conn = get_db()
            cur = conn.cursor()
            cur.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
            conn.commit()
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Built for demo and educational purposes. Not a medical device.")
