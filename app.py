import cv2
from fer import FER
from ultralytics import YOLO
import mediapipe as mp
import math
from flask import Flask, render_template_string, Response
import winsound   # ⬅️ Add this

# Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# FER
detector = FER(mtcnn=True)

# HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Weapon + Emotion + Pose Detection</title>
    <style>
        body { background-color: black; text-align: center; }
        h1 { color: aqua; margin-top: 20px; }
        .video-container { margin-top: 30px; }
        img { border: 3px solid aqua; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>Weapon + Emotion + Pose Detection</h1>
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" width="800">
    </div>
    <audio id="alert-sound" src="/static/beep.mp3"></audio>
</body>
</html>
"""

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fight_alert = False

        # -------- Pose detection --------
        results_pose = pose.process(rgb_frame)
        if results_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            lm = results_pose.pose_landmarks.landmark
            left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

            if left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y:
                fight_alert = True
                cv2.putText(frame, "Aggressive Pose!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # -------- Weapon detection --------
        results_yolo = model(frame, verbose=False)
        for r in results_yolo[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            label = model.names[cls]
            if label.lower() in ["knife", "gun", "pistol"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                fight_alert = True

        # -------- Emotion detection --------
        emotions = detector.detect_emotions(frame)
        for e in emotions:
            (x, y, w, h) = e["box"]
            emotion = max(e["emotions"], key=e["emotions"].get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # -------- Fight alert --------
        if fight_alert:
            cv2.putText(frame, "FIGHT ALERT!", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 🔊 Beep sound on server
            winsound.Beep(1000, 500)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
