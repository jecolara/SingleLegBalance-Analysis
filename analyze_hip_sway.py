import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer los datos del CSV
df = pd.read_csv("keypoints_data.csv")

# Extraer coordenadas de la cadera (hip)
hip_x = df["hip_x"]
hip_y = df["hip_y"]

# Calcular el desplazamiento respecto a la posición inicial
delta_x = hip_x - hip_x.iloc[0]
delta_y = hip_y - hip_y.iloc[0]
sway = np.sqrt(delta_x**2 + delta_y**2)

# Calcular estadísticas simples
mean_sway = np.mean(sway)
max_sway = np.max(sway)

print(f"Promedio de sway: {mean_sway:.4f}")
print(f"Máximo sway: {max_sway:.4f}")

# Graficar el desplazamiento
plt.figure(figsize=(10, 5))
plt.plot(sway, label="Sway (distancia desde inicio)")
plt.xlabel("Fotogramas")
plt.ylabel("Sway (px)")
plt.title("Análisis del balance con una pierna usando posición de cadera")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sway_plot.png")
plt.show()
