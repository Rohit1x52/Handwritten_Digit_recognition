import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import predict_digit_from_canvas

st.title("🖊️ Handwritten Digit Recognition")
st.write("Draw or upload a digit to predict it!")

# Drawing canvas or file uploader
option = st.radio("Choose input method:", [" Draw Digit", " Upload Image"])

if option == " Draw Digit":
    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        stroke_width=st.slider("Brush Size", 10, 25, 15),
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
        point_display_radius=0,
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        if st.button("Predict Digit"):
            digit, confidence = predict_digit_from_canvas(img)
            st.success(f"Predicted Digit: **{digit}** (Confidence: {confidence:.2f})")

elif option == " Upload Image":
    uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded))
        st.image(image, caption="Uploaded Image", width=200)
        if st.button("Predict Digit"):
            digit, confidence = predict_digit_from_canvas(image)
            st.success(f"Predicted Digit: **{digit}** (Confidence: {confidence:.2f})")
