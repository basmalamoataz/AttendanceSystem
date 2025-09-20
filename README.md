# 📝 Siamese Attendance System

## 📌 Project Overview

This project is an **attendance system** based on **Face Recognition**, combining:

1. **YOLO** for **real-time face detection** 🕵️‍♂️
2. **Siamese Neural Network** for **accurate face recognition** 🤖

It allows **automatic attendance recording** when a person's face is recognized.

---

## ⚙️ Features

* 🕵️‍♂️ **Real-time face detection** using YOLO
* 🤖 **Accurate face recognition** using Siamese Network
* ➕ Easy addition of new faces to the system
* 📊 Automatic attendance logging in a database
* 🔄 Support for **Train / Validation** data split
* 🖥 Simple interface to display results
* 🎯 **Customizable threshold** for face recognition

---

## 🖥 Streamlit Interface

The project includes an optional **Streamlit dashboard** for a user-friendly interface:

* Display **live camera feed** or uploaded images
* Show **YOLO face detection bounding boxes**
* Identify faces using **Siamese Network**
* Log attendance automatically
* Adjust **recognition threshold** in real-time

**How to run Streamlit interface:**

```bash
streamlit run src/app.py
```
![Screenshot 2025-09-20 173921](https://github.com/user-attachments/assets/af7c5460-b79e-4af3-8a77-63c60ddd291c)

---

**How it works:**

1. **Face Detection (YOLO):** Detects faces in images or video.
2. **Face Recognition (Siamese Network):** Compares detected faces with stored embeddings to identify individuals.
3. **Attendance Recording:** Automatically logs attendance for recognized faces.

---

## 🔧 Siamese Network Threshold

The **Siamese Network** computes a similarity score (distance) between two face embeddings:

* **Threshold**: Determines if two faces are the same person.

  * Distance < threshold → **same person**
  * ![WhatsApp Image 2025-09-19 at 11 54 22 PM](https://github.com/user-attachments/assets/553cd42a-8277-41bf-922c-f8523285c59b)

  * Distance > threshold → **different person**
  * ![WhatsApp Image 2025-09-19 at 11 54 23 PM](https://github.com/user-attachments/assets/3bc2c63b-1d47-44a6-a9dd-e83d438adb6a)


**Why it matters:**

* **Lower threshold** → stricter matching → reduces false positives, may increase false negatives
* **Higher threshold** → looser matching → reduces false negatives, may increase false positives

**Tip:** Tune the threshold on a **validation set** for the best balance between accuracy and reliability.

---

## 🛠 Technologies Used

* Python 3.x
* YOLO (Face Detection)
* TensorFlow / Keras (Siamese Network)
* OpenCV
* NumPy, Pandas
* Streamlit (optional, for interface)
* SQLite or other database to store attendance records

---

## 🏗 Project Structure

```plaintext
siamese-attendance-system/
│
├── data/               # Image data (train / val / test)
├── models/             # Trained models (Siamese & YOLO)
├── src/                # Scripts
│   ├── train.py        # Train Siamese model
│   ├── predict.py      # Predict faces
│   ├── detect.py       # YOLO face detection
│   └── app.py          # Streamlit interface
├── requirements.txt    # Libraries
└── README.md
```

---

## 🚀 How to Use

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Organize images per person:

```plaintext
data/train/
    person1/
        img1.jpg
        img2.jpg
    person2/
        img1.jpg
        img2.jpg
```

### 3. Train the Siamese Model

```bash
python src/train.py
```

### 4. Detect and Recognize Faces

```bash
python src/detect.py --source path_to_image_or_video
python src/predict.py --image path_to_image.jpg
```

### 5. Launch Streamlit Interface

```bash
streamlit run src/app.py
```
Streamlit Link : https://2883a9955b9b.ngrok-free.app/
---

## 📊 Example Results

![WhatsApp Image 2025-09-20 at 5 08 09 PM](https://github.com/user-attachments/assets/dc5d29de-b60e-499a-b77d-26a8b0786d1c)
![WhatsApp Image 2025-09-20 at 4 35 35 PM](https://github.com/user-attachments/assets/a1f3bb34-f4da-4544-86be-5f0c5d62aa7b)

---

## 💡 Notes

* High-quality face images improve recognition accuracy
* Adjust the **threshold** for optimal performance
* YOLO ensures **fast and reliable face detection**, improving overall system efficiency
* Streamlit interface allows **real-time monitoring** and easy interaction

---


