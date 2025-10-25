import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
import plotly.express as px
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils import preprocess_canvas_image, predict_digit_from_canvas, log_prediction

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("notebooks/models/mnist_cnn.h5")

model = load_model()

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Digit Recognition & Analytics",
    page_icon="✍️",
    layout="wide",
)


# SIDEBAR NAVIGATION
page = st.sidebar.radio("📂 Navigation", ["✍️ Digit Recognition", "📊 Analytics Dashboard"])

# =================================================================
# PAGE 1: DIGIT RECOGNITION
# =================================================================
if page == "✍️ Digit Recognition":
    st.title("✍️ Real-Time Handwritten Digit Recognition")
    st.markdown("Draw or capture a digit (0-9) and watch predictions instantly!")

    col1, col2 = st.columns([1, 1])

    # ---------------- Canvas Drawing ----------------
    with col1:
        st.subheader("🎨 Drawing Canvas")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=15,
            stroke_color="white",
            background_color="black",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
            point_display_radius=0,
        )

        if st.button("🔄 Clear Canvas"):
            st.rerun()

    # ---------------- Webcam Capture ----------------
    with col1:
        st.subheader("📷 Capture Digit via Webcam")
        use_webcam = st.checkbox("Use Webcam")
        img_rgb = None

        if use_webcam:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            capture_btn = st.button("Capture Image")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Cannot access webcam!")
                    break
                frame = cv2.flip(frame, 1)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                if capture_btn:
                    img_rgb = frame.copy()
                    cap.release()
                    cv2.destroyAllWindows()
                    st.success("📸 Image Captured!")
                    break

    # ---------------- Prediction Section ----------------
    with col2:
        st.subheader("⚡ Prediction & Probability")
        image_to_predict = None

        if canvas_result.image_data is not None:
            image_to_predict = canvas_result.image_data
        elif img_rgb is not None:
            image_to_predict = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGBA)

        if image_to_predict is not None:
            img = preprocess_canvas_image(image_to_predict)
            img_input = np.expand_dims(img, axis=0)

            preds = model.predict(img_input, verbose=0)[0]
            digit = int(np.argmax(preds))
            confidence = float(np.max(preds))

            st.success(f"**Predicted Digit:** {digit}")
            st.metric("Confidence", f"{confidence*100:.2f}%")

            # Probability Chart
            fig = px.bar(
                x=list(range(10)),
                y=preds,
                labels={"x": "Digit", "y": "Probability"},
                title="Digit Confidence Scores",
                color=preds,
            )
            fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]), template="plotly_dark", bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)

            # Log prediction
            log_prediction(digit_pred=digit, confidence=confidence, input_type="canvas")

            # Feedback Logging
            with st.expander("Not Correct? Submit Feedback"):
                correct_digit = st.number_input("Enter the correct digit:", 0, 9, step=1)
                if st.button("Submit Correction"):
                    feedback_path = "feedback_logs.csv"
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "digit_predicted": digit,
                        "digit_actual": correct_digit,
                        "confidence": confidence,
                        "correct": int(digit == correct_digit),
                    }
                    df = pd.DataFrame([entry])
                    if os.path.exists(feedback_path):
                        df.to_csv(feedback_path, mode="a", header=False, index=False)
                    else:
                        df.to_csv(feedback_path, index=False)
                    st.success("Feedback saved for future retraining!")
        else:
            st.warning("🖋️ Draw something or capture a digit via webcam!")

    st.markdown("---")
    st.caption("Built with ❤️ by **Rohit Ranjan Kumar** | TensorFlow & Streamlit © 2025")

# =================================================================
# PAGE 2: ANALYTICS DASHBOARD
# =================================================================
elif page == "📊 Analytics Dashboard":
    st.title("📊 Prediction Analytics Dashboard")
    st.markdown("Monitor model accuracy, confusion trends, and user feedback over time.")

    feedback_path = "feedback_logs.csv"

    try:
        logs = pd.read_csv(feedback_path)
        st.success("✅ Feedback logs loaded successfully!")
    except FileNotFoundError:
        st.error("❌ No feedback logs found! Run the main app and submit a few corrections first.")
        st.stop()

    # Most Confused Digits
    st.header("🤔 Most Confused Digits")

    required_cols = {"digit_predicted", "digit_actual", "timestamp"}
    if not required_cols.issubset(logs.columns):
        st.warning("⚠️ Missing required columns in CSV. Please check your logging format.")
    else:
        confused = logs.dropna(subset=["digit_actual"])
        confused_counts = confused[confused["digit_predicted"] != confused["digit_actual"]]
        confusion_stats = (
            confused_counts.groupby(["digit_actual", "digit_predicted"])
            .size()
            .reset_index(name="count")
        )

        if not confusion_stats.empty:
            pivot_df = confusion_stats.pivot(
                index="digit_actual",
                columns="digit_predicted",
                values="count"
            )

            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df.fillna(0), annot=True, fmt=".0f", cmap="coolwarm")
            plt.title("Most Confused Digits")
            st.pyplot(plt)
        else:
            st.info("No confusion data yet. Add some feedback first.")

    # Accuracy Over Time
    st.header("📈 Accuracy Over Time")
    try:
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
        logs["correct"] = logs["digit_predicted"] == logs["digit_actual"]
        accuracy_over_time = logs.groupby(pd.Grouper(key="timestamp", freq="D"))["correct"].mean()

        st.line_chart(accuracy_over_time)
        st.write("Daily Accuracy (%)")
    except Exception as e:
        st.warning(f"Could not calculate accuracy trend: {e}")

    # Summary Stats
    st.header("📋 Summary Statistics")

    total_preds = len(logs)

    if "correct" not in logs.columns and {"digit_predicted", "digit_actual"}.issubset(logs.columns):
        logs["correct"] = logs["digit_predicted"] == logs["digit_actual"]
    elif "correct" not in logs.columns:
        logs["correct"] = False  # default fallback

    if total_preds > 0:
       correct_preds = logs["correct"].sum()
       accuracy = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
       st.metric("Overall Accuracy", f"{accuracy:.2f}%")
       st.metric("Total Predictions Logged", total_preds)
    else:
       st.info("No predictions logged yet.")