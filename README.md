# 🗳️ Face Recognition Voting System  

A secure and automated voting system that uses **CNN + SVM + DeepFace** for real-time face recognition.  
This project ensures **one-person-one-vote** by capturing the voter’s face, verifying it against the database, and preventing duplicate voting with an **alarm alert system**.  

## 🚀 Features  
- CNN (VGG16) for face feature extraction  
- SVM for classification of registered voters  
- DeepFace for enhanced recognition and duplicate detection  
- One-time voting enforcement with real-time duplicate detection  
- Alarm alert when the same person tries to vote again  
- Flask-based web application with **Indian flag UI** and transparent input box  
- Voting interface with **3 parties to choose from**  
- Stored images database with auto-training for new entries  

## 🛠️ Technologies Used  
- **Python, Flask** – Backend and web framework  
- **OpenCV** – Real-time image capture  
- **TensorFlow / Keras (VGG16)** – CNN feature extraction  
- **Scikit-learn (SVM)** – Classification  
- **DeepFace** – Face verification  
- **HTML, CSS** – Frontend (with Indian flag background UI)  

## 📂 Project Structure  
├── app.py # Main Flask application
├── train_model.py # CNN + SVM training and recognition
├── capture_faces.py # Script to capture and store faces
├── templates/ # HTML files (index.html, face.html)
├── static/ # CSS (style.css), background images
├── database/ # Stored face images
└── svm_model.pkl # Trained SVM model


## 🎯 How It Works  
1. User enters their name on the web app.  
2. Face is captured via webcam and stored in the database.  
3. System trains the image using CNN (feature extraction) + SVM (classification).  
4. User is allowed to vote once.  
5. If the same user tries again, system detects it with DeepFace and **rings an alarm**.  
6. After verification, user is shown a **voting poll with 3 parties**.  

---

⚡ **This project demonstrates how AI can make voting systems more secure, transparent, and tamper-proof.**  
