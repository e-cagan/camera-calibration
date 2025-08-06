import os
import yaml
import matplotlib.pyplot as plt

calibration_dir = 'calibration_results'
zoom_levels = []
fx_values = []
fy_values = []

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

    # Boş ya da hatalı dosya kontrolü
    if data is None or "camera_matrix" not in data:
        print(f"❌ {fname} içeriği geçersiz veya eksik.")
        continue

    try:
        fx = data["camera_matrix"][0][0]
        fy = data["camera_matrix"][1][1]
    except Exception as e:
        print(f"❌ {fname} içinde fx/fy alınamadı → {e}")
        continue

    zoom_levels.append(zoom_level)
    fx_values.append(fx)
    fy_values.append(fy)

# 🎯 Görselleştirme
plt.plot(zoom_levels, fx_values, marker='o', label='fx (X ekseni odak)')
plt.plot(zoom_levels, fy_values, marker='s', label='fy (Y ekseni odak)')
plt.xlabel('Zoom Level (x)')
plt.ylabel('Focal Length (pixels)')
plt.title('Zoom Level vs Focal Length')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
