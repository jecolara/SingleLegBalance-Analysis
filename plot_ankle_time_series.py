import csv
import matplotlib.pyplot as plt

# Leer datos del CSV
timestamps = []
ankle_x = []
ankle_y = []

with open("keypoints_data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        timestamps.append(float(row["timestamp"]))
        ankle_x.append(float(row["ankle_x"]))
        ankle_y.append(float(row["ankle_y"]))

# Gráficas X vs tiempo y Y vs tiempo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(timestamps, ankle_x, label='X')
plt.title("Movimiento horizontal del tobillo (X)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición X (normalizada)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(timestamps, ankle_y, label='Y', color='orange')
plt.title("Movimiento vertical del tobillo (Y)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición Y (normalizada)")
plt.grid(True)

plt.tight_layout()
plt.show()
