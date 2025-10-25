import numpy as np
import cv2
import tensorflow as tf
import os
import csv
from datetime import datetime

# -----------------------------
# Model Loading
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "digit_model.h5")

@tf.keras.utils.register_keras_serializable()
def custom_activation(x):
    return tf.nn.relu(x)

@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "custom_activation": custom_activation,
            "custom_loss": custom_loss
        }
    )
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
def preprocess_canvas_image(canvas_rgba, target_size=28, pad=10):
    """
    Preprocess RGBA canvas or webcam image for CNN prediction.
    Returns (28,28,1) float32 array normalized [0,1].
    """
    # Convert RGBA -> Grayscale
    img = cv2.cvtColor(canvas_rgba.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Invert if background is light
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Gaussian blur and threshold
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        resized = cv2.resize(th, (target_size, target_size))
        out = resized.astype("float32") / 255.0
        return out.reshape(target_size, target_size, 1)

    # Bounding box around digit
    x_min = min([cv2.boundingRect(c)[0] for c in contours])
    y_min = min([cv2.boundingRect(c)[1] for c in contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

    # Pad and crop
    x_min, y_min = max(int(x_min - pad), 0), max(int(y_min - pad), 0)
    x_max, y_max = min(int(x_max + pad), th.shape[1]), min(int(y_max + pad), th.shape[0])
    roi = th[y_min:y_max, x_min:x_max]

    # Center in square
    h, w = roi.shape
    size = max(h, w)
    square = np.full((size, size), 255, dtype=np.uint8)
    y_offset, x_offset = (size - h) // 2, (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = roi

    # Deskew using moments
    moments = cv2.moments(square)
    if abs(moments["mu02"]) > 1e-2:
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
        square = cv2.warpAffine(square, M, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Resize and normalize
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    out = resized.astype("float32") / 255.0

    # Ensure shape (H,W,1)
    return out.reshape(target_size, target_size, 1)

# -----------------------------
# Prediction Wrapper
# -----------------------------
def predict_digit_from_canvas(canvas_rgba):
    """
    Takes RGBA canvas or webcam image → preprocess → predict digit and confidence.
    Returns (digit:int, confidence:float)
    """
    if model is None:
        return None, 0.0

    # Preprocess
    img = preprocess_canvas_image(canvas_rgba)

    # Ensure proper shape (1,28,28,1) and dtype float32
    img_input = np.expand_dims(img, axis=0).astype(np.float32)

    # Predict
    preds = model.predict(img_input, verbose=0)[0]
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return digit, confidence

def log_prediction(digit_pred, confidence, input_type="canvas", digit_actual=None):
    """
    Logs prediction results for dashboard analytics.
    Ensures consistent column count every time.
    """
    log_file = "feedback_logs.csv"
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "input_type",
                "digit_predicted",
                "confidence",
                "digit_actual",
                "correct"
            ])

        # Normalize values — always write exactly 6 fields
        correct_value = ""
        if digit_actual is not None:
            correct_value = int(digit_pred == digit_actual)

        writer.writerow([
            datetime.now().isoformat(),
            input_type,
            digit_pred,
            round(float(confidence), 4),
            digit_actual if digit_actual is not None else "",
            correct_value
        ])
