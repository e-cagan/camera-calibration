import os
import yaml
import matplotlib.pyplot as plt

calibration_dir = 'calibration_results'
zoom_levels = []
cx_values = []
cy_values = []

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

    if data is None or "camera_matrix" not in data:
        print(f"❌ {fname} içeriği geçersiz veya eksik.")
        continue

    try:
        cx = data["camera_matrix"][0][2]
        cy = data["camera_matrix"][1][2]
    except Exception as e:
        print(f"❌ {fname} içinde cx/cy alınamadı: {e}")
        continue

    zoom_levels.append(zoom_level)
    cx_values.append(cx)
    cy_values.append(cy)

# 🎯 Görselleştirme
plt.plot(zoom_levels, cx_values, marker='o', label='cx (X merkezi)')
plt.plot(zoom_levels, cy_values, marker='s', label='cy (Y merkezi)')
plt.xlabel('Zoom Level (x)')
plt.ylabel('Center Coordinate (pixels)')
plt.title('Zoom Level vs Image Center (cx, cy)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
