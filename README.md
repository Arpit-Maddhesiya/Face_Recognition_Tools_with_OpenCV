# 👤 Face Recognition System using OpenCV & LBPH

A real-time **Face Detection and Recognition System** built using **Python, OpenCV, Haar Cascade Classifier, and LBPH (Local Binary Patterns Histograms)**.

The system captures facial images through a webcam, detects and stores cropped faces, trains an LBPH face-recognition model, and finally recognizes registered people in real time.

---

## 📌 Project Overview

This project implements a complete face-recognition pipeline consisting of three major stages:

1. **Face Data Collection** – Captures faces from a webcam and saves cropped face images with a unique ID.
2. **Model Training** – Uses the collected images to train an LBPH face-recognition model.
3. **Real-Time Recognition** – Uses the trained model to recognize registered faces through a webcam.

The project uses **Haar Cascade** for face detection and **LBPH** for face recognition.

---

## ✨ Features

* 📷 Real-time webcam face detection
* 👤 Automatic face cropping
* 🆔 Unique ID-based face registration
* 💾 Automatic storage of face training images
* 🧠 LBPH-based face recognition
* 🎥 Real-time face recognition using webcam
* 🟩 Face detection bounding boxes
* 🏷️ Displays recognized person's name
* ⚡ Lightweight and suitable for beginners
* 🐍 Completely implemented in Python

---

## 🛠️ Technologies Used

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| **Python**       | Main programming language              |
| **OpenCV**       | Computer vision and webcam processing  |
| **Haar Cascade** | Face detection                         |
| **LBPH**         | Face recognition                       |
| **NumPy**        | Image array processing                 |
| **Pillow (PIL)** | Reading and converting training images |

---

## 🧠 How It Works

The complete system follows this pipeline:

```text
                 ┌─────────────────────┐
                 │      Webcam         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Haar Cascade      │
                 │   Face Detection    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Crop Detected Face  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Save Face Images    │
                 │ user.ID.number.jpg  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    LBPH Training    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    training.yml     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Real-Time Webcam    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ LBPH Prediction     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Display Person Name │
                 └─────────────────────┘
```

---

# 📂 Project Structure

```text
FaceRecognition/
│
├── detector.py
├── trainer.py
├── recognizer.py
│
├── output/
│   ├── user.1.1.jpg
│   ├── user.1.2.jpg
│   ├── user.1.3.jpg
│   ├── user.2.1.jpg
│   └── ...
│
├── trained/
│   └── training.yml
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
└── README.md
```

---

# 🔹 1. Face Data Collection

The first program captures facial images from the webcam.

### Process

```text
Webcam
   ↓
Capture Frame
   ↓
Convert to Grayscale
   ↓
Detect Face using Haar Cascade
   ↓
Crop Face
   ↓
Save Image
```

Each image is saved using the following format:

```text
user.<ID>.<image_number>.jpg
```

For example:

```text
user.1.1.jpg
user.1.2.jpg
user.1.3.jpg
```

Here:

* `1` → Person ID
* `1, 2, 3...` → Image number

To register another person, assign a different ID.

Example:

```python
id = 3
```

The images will then be stored as:

```text
user.3.1.jpg
user.3.2.jpg
user.3.3.jpg
```

---

# 🔹 2. Model Training

After collecting facial images, the second program trains the LBPH recognizer.

The program:

1. Reads all images from the `output` folder.
2. Converts images to grayscale.
3. Extracts the person's ID from the filename.
4. Creates a list of faces.
5. Creates a corresponding list of IDs.
6. Trains the LBPH recognizer.
7. Saves the trained model as:

```text
training.yml
```

### Training Flow

```text
Stored Face Images
        ↓
Read Images
        ↓
Convert to Grayscale
        ↓
Extract IDs
        ↓
LBPH Training
        ↓
training.yml
```

---

# 🔹 3. Real-Time Face Recognition

The third program loads the trained model:

```text
training.yml
```

It then starts the webcam and continuously:

```text
Capture Frame
     ↓
Convert to Grayscale
     ↓
Detect Face
     ↓
Crop Face
     ↓
LBPH Prediction
     ↓
Get ID
     ↓
Map ID → Person Name
     ↓
Display Name
```

For example:

```python
if id == 1:
    id = "anmol"
elif id == 2:
    id = "modi"
else:
    id = "obama"
```

The ID is mapped to the corresponding person's name.

---

# 🧠 What is Haar Cascade?

**Haar Cascade Classifier** is a machine-learning-based object detection method used to detect objects such as human faces.

In this project, the pretrained classifier:

```text
haarcascade_frontalface_default.xml
```

is used to detect frontal faces.

The detector returns the bounding box:

```text
(x, y, width, height)
```

which is then used to crop the face.

---

# 🧠 What is LBPH?

**LBPH** stands for:

> Local Binary Patterns Histograms

It is a traditional computer-vision technique used for face recognition.

LBPH works by analyzing the local texture patterns of an image.

The basic idea is:

```text
Face Image
    ↓
Divide into local regions
    ↓
Calculate Local Binary Patterns
    ↓
Generate Histograms
    ↓
Compare with trained faces
    ↓
Predict ID
```

LBPH is particularly useful for small-scale face-recognition applications because it is relatively lightweight and does not require a large deep-learning model.

---

# ⚙️ Installation

## 1. Install Python

Make sure Python is installed:

```bash
python --version
```

Recommended:

```text
Python 3.x
```

---

## 2. Install Required Libraries

Install OpenCV with the contrib modules:

```bash
pip install opencv-contrib-python
```

Install NumPy:

```bash
pip install numpy
```

Install Pillow:

```bash
pip install pillow
```

Or install everything together:

```bash
pip install opencv-contrib-python numpy pillow
```

### Why `opencv-contrib-python`?

The LBPH recognizer is available through:

```python
cv2.face.LBPHFaceRecognizer_create()
```

The `cv2.face` module is included in the **contrib** version of OpenCV.

---

# 📁 Configure Paths

Update the paths according to your system.

Example:

```python
faceCascade = cv2.CascadeClassifier(
    r"D:\recent\open cv\softs and docs\misc\haarcascade\haarcascade_frontalface_default.xml"
)
```

Training images:

```python
path = r"D:\recent\open cv\softs and docs\output\\"
```

Trained model:

```python
recognizer.save(
    r"D:\recent\open cv\softs and docs\trained\training.yml"
)
```

Using `r` before the path is recommended on Windows because it prevents problems with backslashes.

---

# ▶️ How to Run

The programs should be executed in this order.

## Step 1 — Collect Face Data

Run:

```bash
python detector.py
```

Look into the `output` directory.

You should see files such as:

```text
user.1.1.jpg
user.1.2.jpg
user.1.3.jpg
...
```

---

## Step 2 — Train the Model

Run:

```bash
python trainer.py
```

After successful training:

```text
trained/
└── training.yml
```

will be created.

---

## Step 3 — Start Recognition

Run:

```bash
python recognizer.py
```

The webcam will open and the system will attempt to recognize registered faces.

Press:

```text
SPACE
```

to exit the webcam window.

---

# 🆔 Adding a New Person

Suppose the existing IDs are:

```text
1 → Anmol
2 → Modi
```

For a new person, assign:

```python
id = 3
```

Run the face-data collection program again.

It will generate:

```text
user.3.1.jpg
user.3.2.jpg
user.3.3.jpg
...
```

Then retrain the model:

```bash
python trainer.py
```

Finally run:

```bash
python recognizer.py
```

and add the corresponding name:

```python
if id == 1:
    name = "Anmol"
elif id == 2:
    name = "Modi"
elif id == 3:
    name = "New Person"
```

---

# 📊 Dataset Format

The training dataset uses filenames to store identity information.

```text
user.ID.image_number.jpg
```

Example:

```text
user.1.1.jpg
user.1.2.jpg
user.1.3.jpg

user.2.1.jpg
user.2.2.jpg
user.2.3.jpg
```

The trainer extracts the ID using:

```python
Id = int(
    os.path.split(imagepath)[-1].split('.')[1]
)
```

Therefore, maintaining the correct filename format is important.

---

# 🔍 Prediction

LBPH returns two values:

```python
id, confidence = recognizer.predict(face)
```

Where:

```text
id
```

represents the predicted person's ID.

The second value represents the recognition distance/confidence measure.

The current implementation can be improved by using a threshold to determine whether a face should be considered **unknown**.

For example:

```python
id, confidence = recognizer.predict(face)

if confidence < threshold:
    print("Recognized")
else:
    print("Unknown")
```

The exact threshold should be selected experimentally for the dataset and environment.

---

# ⚠️ Important Limitations

This project is primarily an educational/small-scale computer-vision implementation.

### 1. Lighting

Recognition performance can change significantly with different lighting conditions.

### 2. Face Angle

The Haar Cascade used here is mainly designed for frontal faces.

### 3. Dataset Size

Too few training images may reduce recognition accuracy.

### 4. Expression Changes

Large changes in facial expressions may affect recognition.

### 5. Camera Quality

Low-resolution cameras can reduce detection and recognition performance.

### 6. Multiple Faces

The system can detect multiple faces, but the ID/name mapping and recognition logic should be carefully designed for multiple people.

### 7. Unknown Face Handling

The basic implementation should be improved with an appropriate LBPH distance threshold rather than assuming every unrecognized ID belongs to a known person.

---

# 🚀 Possible Improvements

The project can be extended with:

* 🔐 Face-based attendance system
* 📋 Automatic attendance logging
* 🕒 Date and time tracking
* 📊 Attendance dashboard
* 💾 SQLite/MySQL database
* 👥 Multiple-user management
* ❌ Unknown-face detection
* 📸 Better dataset collection
* 🎯 Confidence thresholding
* 🖥️ GUI using Tkinter/PyQt
* 🌐 Web interface
* 📱 Mobile/web integration
* 📈 Attendance reports
* 📁 CSV/Excel export
* 🔔 Notifications
* 🔒 Access-control system

---

# 🧪 Testing

The system can be tested under different conditions:

| Condition                       | Expected Result                   |
| ------------------------------- | --------------------------------- |
| Same person, good lighting      | High recognition                  |
| Same person, different lighting | May reduce accuracy               |
| Slight face rotation            | May still recognize               |
| Large face rotation             | May fail                          |
| Unknown person                  | Should be rejected with threshold |
| Multiple faces                  | Detects multiple faces            |
| Low-resolution camera           | Reduced performance               |

---

# 🔒 Privacy Considerations

Face data is biometric information and should be handled responsibly.

For real-world deployment:

* Obtain appropriate consent.
* Secure stored face images.
* Restrict access to training data.
* Avoid storing unnecessary biometric information.
* Provide appropriate deletion mechanisms.
* Follow applicable privacy and data-protection requirements.

This project is intended primarily for **learning and experimentation**.

---

# 📚 Learning Outcomes

By building this project, you learn:

* Python programming
* OpenCV
* Computer vision fundamentals
* Webcam processing
* Haar Cascade face detection
* Image preprocessing
* Image cropping
* NumPy arrays
* LBPH face recognition
* Model training and persistence
* Real-time prediction
* Basic biometric-system concepts

---

# 💡 Project Workflow in One View

```text
                 FACE REGISTRATION
                        │
                        ▼
                 Open Webcam
                        │
                        ▼
                Detect Face
                        │
                        ▼
                 Crop Face
                        │
                        ▼
             Save 30–50 Images
                        │
                        ▼
                  output/
                        │
                        ▼
                ┌──────────────┐
                │    TRAIN     │
                │    LBPH      │
                └──────┬───────┘
                       │
                       ▼
                 training.yml
                       │
                       ▼
                Open Webcam
                       │
                       ▼
                Detect Face
                       │
                       ▼
                LBPH Prediction
                       │
                       ▼
                 Predicted ID
                       │
                       ▼
                  Person Name
```

---

# 📌 Project Highlights

> **Face Recognition System using Python & OpenCV**

A real-time face recognition application that uses **Haar Cascade for face detection** and **LBPH for face recognition**. The system captures and stores face datasets, trains a recognition model, and performs real-time identity prediction through a webcam.

### Core Pipeline

```text
Face Detection
      ↓
Face Dataset Creation
      ↓
LBPH Model Training
      ↓
Model Serialization
      ↓
Real-Time Recognition
```

---

<div align="center">
 👨‍💻 Author

<div align="center">

## Arpit Maddhesiya

**Full Stack Developer • MERN Enthusiast • AI Explorer**

</div>

<p align="center">

<a href="https://github.com/Arpit-Maddhesiya">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github">
</a>

<a href="https://www.linkedin.com/in/arpit-maddhesiya/">
<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin">
</a>

<a href="mailto:your-email@example.com">
<img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail">
</a>

</p>



<div align="center">

## ⭐ Give this repository a Star!

It motivates me to build more impactful open-source software.

</div>
Python • OpenCV • Haar Cascade • LBPH • Computer Vision

