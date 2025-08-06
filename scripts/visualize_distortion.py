import os
import yaml
import matplotlib.pyplot as plt

calibration_dir = 'calibration_results'
zoom_levels = []
k1_values, k2_values, p1_values, p2_values, k3_values = [], [], [], [], []

for fname in sorted(os.listdir(calibration_dir)):
    if not fname.endswith(".yaml"):
        continue

    try:
        zoom_level = int(fname.split("_")[1][0])
    except:
        print(f"❌ Dosya ismi beklenmedik formatta: {fname}")
        continue

    filepath = os.path.join(calibration_dir, fname)

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    if data is None or "distortion_coefficients" not in data:
        print(f"❌ {fname} içeriği geçersiz veya eksik.")
        continue

    coeffs = data["distortion_coefficients"][0]
    if len(coeffs) < 5:
        print(f"❌ {fname} içinde yeterli distortion parametresi yok.")
        continue

    zoom_levels.append(zoom_level)
    k1_values.append(coeffs[0])
    k2_values.append(coeffs[1])
    p1_values.append(coeffs[2])
    p2_values.append(coeffs[3])
    k3_values.append(coeffs[4])

# 🎨 Görselleştirme
plt.plot(zoom_levels, k1_values, marker='o', label='k1')
plt.plot(zoom_levels, k2_values, marker='s', label='k2')
plt.plot(zoom_levels, p1_values, marker='^', label='p1')
plt.plot(zoom_levels, p2_values, marker='v', label='p2')
plt.plot(zoom_levels, k3_values, marker='x', label='k3')

plt.xlabel('Zoom Level (x)')
plt.ylabel('Distortion Coefficients')
plt.title('Zoom Level vs Distortion Parameters')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
