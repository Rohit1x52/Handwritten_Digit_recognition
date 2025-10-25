🧠 Handwritten Digit Recognition

🧩 Overview

This project demonstrates an end-to-end deep learning pipeline for recognizing handwritten digits similar to the MNIST dataset.
Built using TensorFlow CNNs and a sleek Streamlit interface, it allows you to:
- Draw digits on a canvas ✍️  
- Capture digits via webcam 📷  
- Get instant predictions with confidence scores ⚡  
- Log feedback for continuous model improvement 🔄  
- View analytics dashboards 📊 (accuracy, confusion matrix)

⚙️ Tech Stack
-Frontend/UI
🖥️ Streamlit — Interactive web app framework
🎨 Streamlit DrawCanvas — Digit drawing interface
-Backend & Model
🧠 PyTorch — Deep learning model training and inference
🧮 NumPy, OpenCV — Image preprocessing and transformations
-Data & Analytics
📊 Pandas — CSV data logging and processing
📈 Seaborn & Matplotlib — Data visualization
🗃️ CSV-based logging system (extendable to SQL or NoSQL DB)

🌟 Features
Real-time Digit Prediction — Draw a digit and get instant prediction with confidence.
Prediction Logging — Every input and prediction is logged with timestamp, confidence, and feedback.
Analytics Dashboard — Visualize model performance and user input trends.
Confusion Analysis — Heatmap showing which digits are most often confused.
Accuracy Monitoring — Track daily accuracy trends from feedback logs.
Lightweight & Portable — No heavy dependencies beyond core ML and visualization libs.

📁 Project Structure

Handwritten_Digit_Recognition/
├── app/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── app.py
│   ├── logger.py
│   ├── main.py
│   └── utils.py
├── notebooks/
│   ├── models/
│   │   └── mnist_cnn.h5
│   └── data_prep.ipynb
├── venv/
├── .gitignore
├── feedback_logs.csv
├── README.md
└── requirements.txt


🧾 Analytics & Logging
Every user interaction is logged in data/feedback_logs.csv with columns:
timestamp	input_type	digit_predicted	confidence	digit_actual
2025-10-23T14:03:12	canvas	7	0.982	7
2025-10-23T14:05:47	canvas	3	0.753	5

## 📈 Dashboard

**Modules Used:**
- `pandas` for loading and grouping data
- `matplotlib` and `seaborn` for plotting
- `streamlit` for interactivity

**Displayed Metrics:**
- ✅ Overall Accuracy
- 🔥 Most Confused Digits (Heatmap)
- 📅 Accuracy Over Time (Line Chart)
- 📊 Prediction Summary Statistics

## 🖼️ Screenshots

### Writing Prediction
![Handwriting Prediction](screenshots/writing-prediction.png)

### Webcam Feature  
![Webcam Digit Recognition](screenshots/Webcam.png)

### Feedback System
![User Feedback](screenshots/Feedback.png)

🧑‍💻 Developer
Rohit Ranjan Kumar
B.Tech Computer Science, Manipal University Jaipur
Passionate about Machine/Deep Learning, AI Systems, Computer Vision, and DSA
📧 Contact Me

Copyright (c) 2024 Rohit