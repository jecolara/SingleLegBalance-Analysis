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

# Graficar trayectoria del tobillo (X vs Y)
plt.figure(figsize=(8, 6))
plt.plot(ankle_x, ankle_y, marker='o', markersize=2, linewidth=1)
plt.gca().invert_yaxis()  # porque el eje Y está invertido en la imagen
plt.title("Trayectoria del tobillo (derecho)")
plt.xlabel("Coordenada X (normalizada)")
plt.ylabel("Coordenada Y (normalizada)")
plt.grid(True)
plt.axis('equal')
plt.show()
