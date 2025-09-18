import cv2
from fer import FER
from ultralytics import YOLO
import mediapipe as mp
import math
import winsound  # For beep sound

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  

# FER for emotion detection
detector = FER(mtcnn=True)

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fight_alert = False

    # --------- Pose Detection ----------
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results_pose.pose_landmarks.landmark
        left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Simple aggressive pose detection
        if left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y:
            fight_alert = True
            cv2.putText(frame, "Aggressive Pose Detected!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # --------- Weapon Detection ----------
    results_yolo = model(frame)
    for r in results_yolo[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        label = model.names[cls]
        if label.lower() in ['knife','gun','pistol']:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            fight_alert = True

    # --------- Emotion Detection ----------
    emotions = detector.detect_emotions(frame)
    for e in emotions:
        (x, y, w, h) = e["box"]
        emotion = max(e["emotions"], key=e["emotions"].get)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # --------- Fight Alert ----------
    if fight_alert:
        cv2.putText(frame, "FIGHT ALERT!", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        winsound.Beep(1000, 500)  # Beep sound: 1000Hz for 500ms

    cv2.imshow("Weapon + Emotion + Fight Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
