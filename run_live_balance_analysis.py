import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Inicializar webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se pudo acceder a la cámara")

data = []
frame_idx = 0

print("Grabando... presiona 'q' para detener.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        data.append({
            "frame": frame_idx,
            "hip_x": hip.x,
            "hip_y": hip.y
        })

        # Dibuja los puntos clave
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Live Pose Tracking - Presiona 'q' para terminar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
pose.close()

# Guardar CSV
df = pd.DataFrame(data)
df.to_csv("live_keypoints_data.csv", index=False)

# Conversión a píxeles
df["hip_x_px"] = df["hip_x"] * 640
df["hip_y_px"] = df["hip_y"] * 480

# Cálculo de sway
hip_x = df["hip_x_px"]
hip_y = df["hip_y_px"]
delta_x = hip_x - hip_x.iloc[0]
delta_y = hip_y - hip_y.iloc[0]
sway = np.sqrt(delta_x**2 + delta_y**2)

# Estadísticas
print(f"Promedio de sway: {np.mean(sway):.4f} px")
print(f"Máximo sway: {np.max(sway):.4f} px")

# Gráfica
plt.figure(figsize=(10, 5))
plt.plot(sway, label="Sway (en tiempo real)")
plt.xlabel("Fotogramas")
plt.ylabel("Sway (px)")
plt.title("Análisis de Balance con una Pierna (en Vivo)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("live_sway_plot.png")
plt.show()
