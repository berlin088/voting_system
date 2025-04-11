import os
import cv2
import numpy as np
from sklearn.svm import SVC
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
from deepface import DeepFace

DATA_DIR = 'database'
MODEL_PATH = 'svm_model.pkl'

# Load pre-trained CNN (VGG16 without top layer)
cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            return None
        img = cv2.resize(img, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = cnn_model.predict(img)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

features, labels = [], []

print("📥 Scanning database for training images...")
for file in os.listdir(DATA_DIR):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(DATA_DIR, file)
        label = file.split('_')[0]
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            labels.append(label)

if len(set(labels)) < 1:
    raise ValueError("No sufficient data found to train.")

svm = SVC(kernel='linear', probability=True)
svm.fit(features, labels)
joblib.dump(svm, MODEL_PATH)
print("✅ Model trained successfully.")

def recognize_face(img_path):
    try:
        feat = extract_features(img_path)
        if feat is None:
            return False, 0.0, None, None

        svm = joblib.load(MODEL_PATH)
        pred = svm.predict([feat])[0]
        prob = max(svm.predict_proba([feat])[0])

        for file in os.listdir(DATA_DIR):
            db_img = os.path.join(DATA_DIR, file)
            result = DeepFace.verify(img_path, db_img, enforce_detection=False)
            if result['verified']:
                print(f"✅ DeepFace Verified: {file}")
                return True, prob, pred, file

        return False, prob, pred, None
    except Exception as e:
        print("Recognition error:", e)
        return False, 0.0, None, None
