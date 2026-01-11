import cv2
import time
import datetime
import pygame
import sqlite3
import os
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

app = Flask(__name__)

# --- CONFIG & DIRECTORIES ---
WIDTH, HEIGHT = 1280, 720
VIOLATION_DIR = 'static/violations'
if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

model = YOLO('yolov8n.pt')
pygame.mixer.init()

class TrafficSystem:
    def __init__(self):
        self.is_running = False
        self.is_muted = False
        self.signal_state = "GREEN" # GREEN, G_TO_Y, RED, R_TO_Y
        self.ped_present = False
        self.last_state_change = time.time()
        self.is_playing_sound = False
        self.logged_ids = set() # To prevent double logging same vehicle

sys = TrafficSystem()

# ROIs
zebra_roi = Polygon([(380, 0), (600, 0), (600, 720), (380, 720)])
stop_roi = Polygon([(630, 0), (660, 0), (660, 720), (630, 720)])

def log_violation(frame, box, v_type):
    # Save image
    x1, y1, x2, y2 = map(int, box)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"violation_{timestamp}.jpg"
    img_path = os.path.join(VIOLATION_DIR, img_name)
    
    # Crop the vehicle with some padding
    crop = frame[max(0, y1-20):min(HEIGHT, y2+20), max(0, x1-20):min(WIDTH, x2+20)]
    cv2.imwrite(img_path, crop)
    
    # DB Entry
    with sqlite3.connect('traffic_monitor.db') as conn:
        conn.execute("INSERT INTO violations (timestamp, type, image_path) VALUES (?, ?, ?)",
                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, img_name))

def manage_signal():
    now = time.time()
    elapsed = now - sys.last_state_change
    if sys.ped_present:
        if sys.signal_state == "GREEN":
            sys.signal_state = "G_TO_Y"; sys.last_state_change = now
        elif sys.signal_state == "G_TO_Y" and elapsed > 1.5:
            sys.signal_state = "RED"; sys.last_state_change = now
    else:
        if sys.signal_state == "RED":
            sys.signal_state = "R_TO_Y"; sys.last_state_change = now
        elif sys.signal_state == "R_TO_Y" and elapsed > 1.0:
            sys.signal_state = "GREEN"; sys.last_state_change = now

def generate_frames():
    cap = cv2.VideoCapture(0)
    while sys.is_running:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        results = model.track(frame, persist=True, classes=[0, 2, 3, 5, 7], verbose=False)
        
        sys.ped_present = False
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, tid in zip(boxes, cls_ids, track_ids):
                x1, y1, x2, y2 = map(int, box)
                contact_pt = Point((x1 + x2) / 2, y2)
                
                # Pedestrian Logic
                if cls == 0:
                    on_zebra = zebra_roi.contains(contact_pt)
                    if on_zebra: sys.ped_present = True
                    color = (0, 0, 255) if on_zebra else (0, 255, 0)
                    label = f"PEDESTRIAN #{tid}"
                # Vehicle Logic
                else:
                    color = (255, 0, 0)
                    label = f"{model.names[cls].upper()} #{tid}"
                    # Violation check
                    if sys.signal_state == "RED" and stop_roi.contains(contact_pt):
                        if tid not in sys.logged_ids:
                            log_violation(frame, box, f"Red Light: {model.names[cls]}")
                            sys.logged_ids.add(tid)
                            color = (0, 0, 255) # Flash Red on violator

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        manage_signal()

        # Audio Logic
        if sys.ped_present and not sys.is_muted:
            if not sys.is_playing_sound:
                try:
                    pygame.mixer.music.load("warning.mp3")
                    pygame.mixer.music.play(-1)
                    sys.is_playing_sound = True
                except: pass
        else:
            if sys.is_playing_sound:
                pygame.mixer.music.stop()
                sys.is_playing_sound = False

        # ROI Drawing
        overlay = frame.copy()
        for i in range(0, 720, 60): cv2.rectangle(overlay, (380, i), (600, i+30), (255, 255, 255), -1)
        cv2.rectangle(overlay, (630, 0), (660, 720), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not sys.is_running: return Response(status=204)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<action>')
def control(action):
    if action == 'start': sys.is_running = True
    elif action == 'stop': 
        sys.is_running = False
        pygame.mixer.music.stop()
        sys.is_playing_sound = False
    elif action == 'mute': sys.is_muted = not sys.is_muted
    return jsonify(running=sys.is_running, muted=sys.is_muted)

@app.route('/get_status')
def get_status():
    light = "GREEN"
    if "YELLOW" in sys.signal_state or "TO_Y" in sys.signal_state: light = "YELLOW"
    elif sys.signal_state == "RED": light = "RED"
    return jsonify(signal=light, running=sys.is_running)

@app.route('/database')
def database():
    with sqlite3.connect('traffic_monitor.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM violations ORDER BY id DESC")
        data = cursor.fetchall()
    return render_template('database.html', violations=data)

if __name__ == '__main__':
    with sqlite3.connect('traffic_monitor.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS violations (id INTEGER PRIMARY KEY, timestamp TEXT, type TEXT, image_path TEXT)')
    app.run(debug=True, threaded=True)