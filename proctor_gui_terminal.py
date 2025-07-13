import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import csv
import os
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, refine_landmarks=True)

# CSV Logging
log_file = open("head_pose_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Pitch", "Yaw", "Roll", "Cheating Detected", "Reason"])

# Thresholds
PITCH_THRESHOLD = 20
YAW_THRESHOLD = 30
EYE_DOWN_RATIO_THRESHOLD = 0.65

# Facial landmark points
model_points = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 263, 33, 287, 57]
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS = 468

# Stats
cheating_counter = 0
cheating_start_time = None
cheating_total_duration = 0.0
attentive_duration = 0.0
session_start_time = time.time()
last_frame_time = session_start_time

# Screenshot and recording setup
screenshot_folder = "cheating_screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('session_recording.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    is_cheating = False
    reason = "None"
    pitch, yaw, roll = 0.0, 0.0, 0.0
    gaze_ratio = 0.0
    face_found = False

    if results.multi_face_landmarks:
        total_faces = len(results.multi_face_landmarks)
        if total_faces > 1:
            is_cheating = True
            reason = "Multiple Faces Detected"
        else:
            face_found = True
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            # Head Pose Estimation
            image_points = []
            for idx in LANDMARK_IDS:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                image_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            image_points = np.array(image_points, dtype=np.float64)

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            pitch, yaw, roll = euler_angles.flatten()

            # Eye Gaze
            eye_top = landmarks[LEFT_EYE_TOP].y
            eye_bottom = landmarks[LEFT_EYE_BOTTOM].y
            eye_height = eye_bottom - eye_top
            iris_center = landmarks[LEFT_IRIS].y if len(landmarks) > LEFT_IRIS else None
            is_eye_down = False
            if iris_center and eye_height > 0:
                iris_offset = iris_center - eye_top
                gaze_ratio = iris_offset / eye_height
                is_eye_down = gaze_ratio > EYE_DOWN_RATIO_THRESHOLD

            if abs(pitch) > PITCH_THRESHOLD or abs(yaw) > YAW_THRESHOLD:
                is_cheating = True
                reason = "Head Turned"
            elif is_eye_down:
                is_cheating = True
                reason = "Eyes Looking Down"
    else:
        is_cheating = True
        reason = "Face Not Detected"

    # Cheating Time Calculation
    if is_cheating:
        if cheating_start_time is None:
            cheating_start_time = time.time()
            cheating_counter += 1
    else:
        if cheating_start_time is not None:
            cheating_total_duration += time.time() - cheating_start_time
            cheating_start_time = None

    # Attentive Accuracy
    current_time = time.time()
    frame_duration = current_time - last_frame_time
    last_frame_time = current_time
    if not is_cheating:
        attentive_duration += frame_duration
    total_session_duration = current_time - session_start_time
    attentive_accuracy = (attentive_duration / total_session_duration) * 100 if total_session_duration > 0 else 100

    # Save log to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([
        timestamp, f"{pitch:.2f}", f"{yaw:.2f}", f"{roll:.2f}",
        "Yes" if is_cheating else "No", reason
    ])

    # On-screen overlays (only display)
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Gaze: {gaze_ratio:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    cv2.putText(frame, f"Face: {'Yes' if face_found else 'No'}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Events: {cheating_counter}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Cheating Time: {int(cheating_total_duration)} sec", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"🟢 Attentive Accuracy: {attentive_accuracy:.2f}%", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    if is_cheating:
        cv2.putText(frame, f"⚠ Cheating Detected! ({reason})", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)

    # Save frame with overlays to video
    video_writer.write(frame)

    # Save cheating screenshot with overlays
    if is_cheating:
        img_name = f"{screenshot_folder}/cheating_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(img_name, frame)

    # Show frame
    cv2.imshow("Proctoring - Head + Eye + Face Count", frame)
    if cv2.waitKey(1) == 27:
        break

# Wrap up
if cheating_start_time is not None:
    cheating_total_duration += time.time() - cheating_start_time

cap.release()
video_writer.release()
log_file.close()
cv2.destroyAllWindows()
