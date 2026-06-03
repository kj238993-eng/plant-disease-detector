# 🌿 Plant Disease Detector & Assistant (CNN + Flask)

An interactive, deep learning-powered web application that classifies potato leaf diseases (Early Blight, Late Blight, or Healthy) and provides detailed care recommendations. It features bilingual (English/Hindi) support, dynamic probability charts, PDF health report exports, local scan history logs, and an integrated plant care chatbot.

---

## 🚀 Key Features

1. **🧠 Optimized Custom CNN**: Hand-coded neural network trained on the PlantVillage dataset (~28k parameters, ~97.7% accuracy) optimized for fast CPU inference.
2. **🌐 Bilingual UI**: Instant English and हिन्दी (Hindi) translations toggle.
3. **📊 Probability Charts**: Interactive probability distribution chart powered by `Chart.js`.
4. **📄 Health Report PDF Generator**: One-click download of a styled PDF report containing the classification result and treatment instructions.
5. **📜 Recent Scans Dashboard**: Stores scan history locally in the browser's `localStorage` for quick retrieval.
6. **💬 Plant Care AI Chatbot**: A friendly assistant to answer agricultural, watering, and fungicide queries in both English and Hindi.
7. **🔌 Offline Demo Mode**: Includes a `plant_detector_offline.html` file that runs all premium features offline (using simulated mock predictions) by simply double-clicking the file.

---

## 📁 Project Directory Layout

```text
├── app.py                      # Flask backend server
├── train.py                    # Custom CNN model training script
├── plant_model.h5              # Trained Keras CNN model weights
├── class_indices.json          # Folder class index mapping file
├── plant_detector_offline.html # Standard double-click offline demo HTML page
├── static/
│   └── uploads/                # Directory for uploaded leaf images
├── templates/
│   └── index.html              # Frontend UI template (serves all 5 features)
└── .gitignore                  # Ignores large datasets and temporary cache files
```

---

## 🛠️ How to Set Up and Run Locally

### Prerequisites
Make sure you have Python installed. The project runs on **Python 3.13**.

### Step 1: Install Dependencies
Run the following command in your terminal to install the required libraries:
```bash
pip install tensorflow flask pillow numpy
```

### Step 2: Start the Flask Application
Run the backend server using:
```bash
python app.py
```
*(If running under Windows python launcher, use: `py -3.13 app.py`)*

### Step 3: Open the App in your Browser
Once the server loads, open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📝 Technologies Used
* **Deep Learning**: TensorFlow, Keras (Custom CNN, global average pooling)
* **Backend Server**: Python, Flask
* **Frontend Design**: Vanilla HTML, CSS, JavaScript (Google Fonts, HSL tailormade colors)
* **Libraries**: Chart.js (Probability charts), html2pdf.js (PDF reports)
* **Dataset Source**: [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 💡 Notes for Developers Cloning this Project

1. **Running the App Directly (Inference)**:
   * You **DO NOT** need to download the dataset to run the app. The pre-trained weights `plant_model.h5` are already included in the repository. Simply run `python app.py` to start using it immediately!
2. **Retraining the Model**:
   * If you wish to retrain the model, download the dataset from the Kaggle link above, extract the potato leaf folders (`Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`) into a directory path `archive/PlantVillage/`, and execute:
     ```bash
     python train.py
     ```
