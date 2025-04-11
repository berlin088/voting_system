from flask import Flask, render_template, request
import cv2
import os
import uuid
import threading
import pygame
from train_model import recognize_face

app = Flask(__name__)

# Alarm setup
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

def play_alarm():
    alarm_sound.play()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    name = request.form['name']
    if not name:
        return "Name is required", 400

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "❌ Failed to capture face", 500

    # Save temporarily for checking
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("static", temp_filename)
    cv2.imwrite(temp_path, frame)

    recognized, confidence, predicted_name, matched_file = recognize_face(temp_path)

    # Remove temp image after checking
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if recognized:
        return render_template("face.html",
                               status=f"✅ {predicted_name} verified! You may vote.",
                               allow_vote=True,
                               filename=matched_file)
    else:
        threading.Thread(target=play_alarm).start()
        return render_template("face.html",
                               status="🚨 Face not recognized! Access denied.",
                               allow_vote=False,
                               filename=None)

@app.route('/help')
def help():
    return "<h1>Help: Contact admin@example.com</h1>"

@app.route('/contact')
def contact():
    return "<h1>Contact us at: voting-system@example.com</h1>"

if __name__ == '__main__':
    app.run(debug=True)
