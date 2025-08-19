import os, re, yaml, math
import matplotlib.pyplot as plt

CALIB_DIR = "calibration_results"

# ---------- Yardımcılar ----------
def parse_zoom_from_name(name: str):
    """'zoom_10x_calib.yaml' -> 10"""
    m = re.search(r"zoom_(\d+)x", name.lower())
    return int(m.group(1)) if m else None

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def normalize_distortion(d):
    """
    distortion_coefficients hem [k1,k2,p1,p2,k3] hem [[...]] olabilir.
    İlk 5 katsayıyı döndür (yoksa 0 ile doldur).
    """
    if d is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    # iç içe tek katman ise düzleştir
    while isinstance(d, (list, tuple)) and len(d) == 1 and isinstance(d[0], (list, tuple)):
        d = d[0]
    if isinstance(d, (list, tuple)):
        coeffs = list(d)
        if len(coeffs) < 5:
            coeffs = coeffs + [0.0] * (5 - len(coeffs))
        return coeffs[:5]  # k1..k3 (rational varsa fazlasını kes
    return [0.0, 0.0, 0.0, 0.0, 0.0]

# ---------- Tüm YAML'ları topla (zoom -> en iyi sonuç) ----------
results = {}  # zoom:int -> dict
for fname in os.listdir(CALIB_DIR):
    if not fname.endswith(".yaml"):
        continue
    z = parse_zoom_from_name(fname)
    if z is None:
        print(f"❌ Dosya ismi beklenmedik: {fname}")
        continue

    data = load_yaml(os.path.join(CALIB_DIR, fname))
    K = data.get("camera_matrix")
    if not (isinstance(K, list) and len(K) >= 2 and isinstance(K[0], list)):
        print(f"❌ {fname}: camera_matrix formatı beklenmedik.")
        continue

    try:
        fx, fy = float(K[0][0]), float(K[1][1])
        cx, cy = float(K[0][2]), float(K[1][2])
    except Exception as e:
        print(f"❌ {fname}: fx/fy/cx/cy okunamadı -> {e}")
        continue

    k1, k2, p1, p2, k3 = normalize_distortion(data.get("distortion_coefficients"))
    rms = float(data.get("reprojection_error", math.inf))

    record = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
        "rms": rms, "file": fname
    }

    # Aynı zoom için birden fazla YAML varsa, RMS'i düşük olanı tut
    if z not in results or rms < results[z]["rms"]:
        results[z] = record

# ---------- Numerik sıraya çek ----------
zooms = sorted(results.keys())
if not zooms:
    raise SystemExit("⚠️ Hiç geçerli YAML bulunamadı.")

# ---------- Veri vektörleri ----------
cx_vals = [results[z]["cx"] for z in zooms]
cy_vals = [results[z]["cy"] for z in zooms]
fx_vals = [results[z]["fx"] for z in zooms]
fy_vals = [results[z]["fy"] for z in zooms]
k1_vals = [results[z]["k1"] for z in zooms]
k2_vals = [results[z]["k2"] for z in zooms]
p1_vals = [results[z]["p1"] for z in zooms]
p2_vals = [results[z]["p2"] for z in zooms]
k3_vals = [results[z]["k3"] for z in zooms]

# ---------- Plot 1: cx, cy ----------
plt.figure()
plt.plot(zooms, cx_vals, marker="o", label="cx (X merkezi)")
plt.plot(zooms, cy_vals, marker="s", label="cy (Y merkezi)")
plt.xticks(zooms)
plt.xlabel("Zoom (x)")
plt.ylabel("Center Coordinate (pixels)")
plt.title("Zoom vs Image Center (cx, cy)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Plot 2: distorsiyon katsayıları ----------
plt.figure()
plt.plot(zooms, k1_vals, marker="o", label="k1")
plt.plot(zooms, k2_vals, marker="s", label="k2")
plt.plot(zooms, p1_vals, marker="^", label="p1")
plt.plot(zooms, p2_vals, marker="v", label="p2")
plt.plot(zooms, k3_vals, marker="x", label="k3")
plt.xticks(zooms)
plt.xlabel("Zoom (x)")
plt.ylabel("Distortion Coefficients")
plt.title("Zoom vs Distortion Parameters")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Plot 3: fx, fy ----------
plt.figure()
plt.plot(zooms, fx_vals, marker="o", label="fx")
plt.plot(zooms, fy_vals, marker="s", label="fy")
plt.xticks(zooms)
plt.xlabel("Zoom (x)")
plt.ylabel("Focal Length (pixels)")
plt.title("Zoom vs Focal Length (fx, fy)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Özet (RMS'e göre seçilmiş en iyi sonuçlar) ===")
for z in zooms:
    r = results[z]
    print(f"{z:>2}x | fx={r['fx']:.1f}  fy={r['fy']:.1f}  "
          f"cx={r['cx']:.1f}  cy={r['cy']:.1f}  k1={r['k1']:.3f}  k2={r['k2']:.3f}  rms={r['rms']:.3f}  ({r['file']})")
