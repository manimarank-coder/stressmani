"""
Stress Detection Web Application using Flask and OpenCV
This app uses facial expression analysis to detect stress levels
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variables for stress analysis
stress_history = []
current_stress_level = "Normal"

def analyze_facial_features(frame, face):
    """Analyze facial features to determine stress indicators"""
    x, y, w, h = face
    roi_gray = frame[y:y+h, x:x+w]
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    
    # Calculate stress indicators
    stress_score = 0
    
    # Eye detection (fewer eyes detected might indicate squinting/stress)
    if len(eyes) < 2:
        stress_score += 20
    
    # Face region brightness (darker might indicate tiredness/stress)
    brightness = np.mean(roi_gray)
    if brightness < 80:
        stress_score += 15
    
    # Edge detection (more edges might indicate tension)
    edges = cv2.Canny(roi_gray, 50, 150)
    edge_density = np.sum(edges) / (w * h)
    if edge_density > 0.15:
        stress_score += 25
    
    # Forehead region analysis (upper 1/3 of face)
    forehead = roi_gray[0:h//3, :]
    forehead_brightness = np.mean(forehead)
    if forehead_brightness < 70:
        stress_score += 20
    
    # Determine stress level
    if stress_score >= 60:
        level = "High Stress"
    elif stress_score >= 30:
        level = "Moderate Stress"
    else:
        level = "Normal"
    
    return level, stress_score, len(eyes)

def generate_frames():
    """Generate video frames with stress detection"""
    global current_stress_level
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for face in faces:
            x, y, w, h = face
            
            # Analyze stress level
            stress_level, stress_score, eye_count = analyze_facial_features(gray, face)
            current_stress_level = stress_level
            
            # Choose color based on stress level
            if stress_level == "High Stress":
                color = (0, 0, 255)  # Red
            elif stress_level == "Moderate Stress":
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display stress level
            cv2.putText(frame, f"{stress_level}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display stress score
            cv2.putText(frame, f"Score: {stress_score}", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Store in history
            stress_history.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': stress_level,
                'score': stress_score
            })
            
            # Keep only last 50 readings
            if len(stress_history) > 50:
                stress_history.pop(0)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stress_data')
def stress_data():
    """API endpoint for stress data"""
    return jsonify({
        'current_level': current_stress_level,
        'history': stress_history[-10:] if stress_history else []
    })

@app.route('/recommendations')
def recommendations():
    """Get stress relief recommendations"""
    if current_stress_level == "High Stress":
        recs = [
            "Take a 5-minute breathing exercise",
            "Step away from screen for 10 minutes",
            "Try progressive muscle relaxation",
            "Consider talking to a counselor",
            "Practice mindfulness meditation"
        ]
    elif current_stress_level == "Moderate Stress":
        recs = [
            "Take short breaks every hour",
            "Practice deep breathing",
            "Go for a short walk",
            "Listen to calming music",
            "Stretch your body"
        ]
    else:
        recs = [
            "Maintain good posture",
            "Stay hydrated",
            "Keep up regular breaks",
            "Continue healthy habits"
        ]
    
    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
