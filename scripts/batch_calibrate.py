import os
from utils import load_images_from_folder, find_corners, calibrate_camera, save_calibration_result

# Sabitler
CALIB_IMAGES_DIR = "calibration_images"
OUTPUT_DIR = "calibration_results"
PATTERN_SIZE = (22, 15)

# Sonuç klasörü varsa, yoksa oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

for folder in sorted(os.listdir(CALIB_IMAGES_DIR)):
    if not folder.startswith("zoom_"):
        continue

    folder_path = os.path.join(CALIB_IMAGES_DIR, folder)
    print(f"\n🔍 İşleniyor: {folder}")

    images = load_images_from_folder(folder_path)
    
    objpoints, imgpoints, image_size = find_corners(images, PATTERN_SIZE)
    
    if len(objpoints) == 0:
        print(f"❌ {folder}: Hiç köşe bulunamadı. Atlınıyor.")
        continue

    result = calibrate_camera(objpoints, imgpoints, image_size)

    output_path = os.path.join(OUTPUT_DIR, f"{folder}_calib.yaml")
    save_calibration_result(output_path, result)
