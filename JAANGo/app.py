import cv2
import time
import datetime
import pygame
import sqlite3
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720
model = YOLO('yolov8n.pt')
pygame.mixer.init()
try:
    warning_sound = pygame.mixer.Sound("warning.mp3")
except:
    warning_sound = None

# Shared State
class SystemState:
    signal = "GREEN"  # GREEN, YELLOW, RED
    pedestrian_on_zebra = False
    is_playing_sound = False

state = SystemState()

# ROIs (Left Zebra, Center Stop)
zebra_roi = Polygon([(380, 0), (600, 0), (600, 720), (380, 720)])
stop_roi = Polygon([(630, 0), (660, 0), (660, 720), (630, 720)])

def generate_frames():
    cap = cv2.VideoCapture(0) # Use 0 for webcam or "path/to/video.mp4"
    yellow_timer = 0
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        results = model.track(frame, persist=True, classes=[0, 2, 3, 5, 7], verbose=False)
        
        state.pedestrian_on_zebra = False
        current_vehicles = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, tid in zip(boxes, cls_ids, track_ids):
                x1, y1, x2, y2 = map(int, box)
                contact_pt = Point((x1 + x2) / 2, y2)
                
                if cls == 0: # Person
                    if zebra_roi.contains(contact_pt):
                        state.pedestrian_on_zebra = True
                    color = (0, 0, 255) if zebra_roi.contains(contact_pt) else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else: # Vehicle
                    current_vehicles.append({'in_stop': stop_roi.contains(contact_pt)})
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- Logic & Sound ---
        if state.pedestrian_on_zebra:
            if state.signal == "GREEN":
                state.signal = "YELLOW"
                yellow_timer = time.time()
            if not state.is_playing_sound and warning_sound:
                warning_sound.play(loops=-1)
                state.is_playing_sound = True
        else:
            if state.is_playing_sound and warning_sound:
                warning_sound.stop()
                state.is_playing_sound = False
            state.signal = "GREEN"

        if state.signal == "YELLOW" and (time.time() - yellow_timer > 3):
            state.signal = "RED"

        # --- Visual Infrastructure ---
        # Draw Zebra
        overlay = frame.copy()
        for i in range(0, 720, 60):
            cv2.rectangle(overlay, (380, i), (600, i+30), (255, 255, 255), -1)
        # Draw Stop Line
        cv2.rectangle(overlay, (630, 0), (660, 720), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify(signal=state.signal)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)