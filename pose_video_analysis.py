import cv2
import mediapipe as mp
import csv
import time

# Inicializar MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cargar video
video_path = "balance_test.mp4"
cap = cv2.VideoCapture(video_path)

# Preparar CSV
output_csv = "keypoints_data.csv"
csv_header = ["timestamp", "hip_x", "hip_y", "knee_x", "knee_y", "ankle_x", "ankle_y", "heel_x", "heel_y", "foot_x", "foot_y"]
csv_file = open(output_csv, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_header)

# Inicializar Pose
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        results = pose.process(image_rgb)

        # Timestamp del frame (en segundos)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if results.pose_landmarks:
            # Coordenadas normalizadas
            landmarks = results.pose_landmarks.landmark

            # Extraer puntos del lado derecho (o cambia a izquierdo si se prefiere)
            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            # Escribir al CSV
            row = [timestamp,
                   hip.x, hip.y,
                   knee.x, knee.y,
                   ankle.x, ankle.y,
                   heel.x, heel.y,
                   foot_index.x, foot_index.y]
            csv_writer.writerow(row)

        # Opcional: mostrar video con marcadores
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Cierre
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"Puntos clave guardados en: {output_csv}")
