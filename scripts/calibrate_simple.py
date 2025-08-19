#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, yaml
import cv2
import numpy as np

# ====== KLASÖRLER ======
CALIB_IMAGES_DIR = "calibration_images"   # içinde zoom_1x, zoom_2x, ...
OUTPUT_DIR       = "calibration_results"

# ====== TAHTA ======
PATTERN_SIZE = (22, 15)   # (cols, rows) = iç köşe sayıları
SQUARE_SIZE  = 2.0        # birim önemli değil; tutarlı olsun

# ====== TEMEL AYARLAR ======
USE_SB = True                  # varsa findChessboardCornersSB kullan
EXCLUDE_BORDER = 1             # 0/1/2/3: kenardan şu kadar halka at
FIX_K3_DEFAULT = True          # K3 kilit (stabilite için)
ZERO_TANGENTIAL = True         # p1=p2=0 (bombeli kağıtta overfit'i engeller)
CENTER_TOL_RATIO = 0.05        # |cx-cx0|, |cy-cy0| uyarı eşiği (oran)
HIGH_ZOOM_PRINCIPAL_FIX_FROM = 4   # 4x ve üstü principal sabit
RMS_LIMIT_LOWZOOM = 2.5
RMS_LIMIT_HIGHZOOM = 3.0

# ====== YARDIMCILAR ======
def parse_zoom_level(name: str):
    m = re.search(r"zoom_(\d+)x", name.lower())
    return int(m.group(1)) if m else None

def list_images(folder_path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(exts)]
    files.sort()
    return files

def load_images_from_folder(folder_path):
    images = []
    for p in list_images(folder_path):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append((img, gray))
    return images

def assert_same_resolution(images):
    sizes = {(g.shape[1], g.shape[0]) for _, g in images}
    if len(sizes) != 1:
        raise ValueError(f"Farklı çözünürlükler: {sizes}")
    return next(iter(sizes))  # (W,H)

def generate_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    obj = np.zeros((cols*rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    obj[:, :2] = grid * square_size
    return obj  # (N,3)

def filter_border_points(corners, objpoints, pattern_size, border):
    """Deforme/lekeli kenar etkisini azalt: dış halkaları at."""
    if border <= 0:
        return corners, objpoints
    cols, rows = pattern_size
    n = cols * rows
    keep_idx = []
    for idx in range(n):
        c = idx % cols
        r = idx // cols
        if (border <= c < cols-border) and (border <= r < rows-border):
            keep_idx.append(idx)
    if len(keep_idx) < 30:  # çok az kaldıysa dokunma
        return corners, objpoints
    corners_f = corners.reshape(-1, 2)[keep_idx].reshape(-1, 1, 2)
    obj_f     = objpoints[keep_idx]
    return corners_f.astype(np.float32), obj_f.astype(np.float32)

def find_corners(images, pattern_size, square_size, use_sb=True, exclude_border=0):
    if not images:
        return [], [], (0, 0)
    image_size = assert_same_resolution(images)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_templ = generate_object_points(pattern_size, square_size)
    objpoints, imgpoints = [], []

    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
    classic_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_NORMALIZE_IMAGE |
                     cv2.CALIB_CB_FAST_CHECK)

    for bgr, gray in images:
        found, corners = False, None

        if use_sb and hasattr(cv2, "findChessboardCornersSB"):
            try:
                found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, sb_flags)
            except Exception:
                found, corners = False, None

        if not found:
            ret, c = cv2.findChessboardCorners(gray, PATTERN_SIZE, classic_flags)
            if ret:
                corners = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), criteria)
                found = True

        if found and corners is not None:
            corners_f, obj_f = filter_border_points(corners, obj_templ, PATTERN_SIZE, exclude_border)
            if corners_f.shape[0] < 10:
                continue
            objpoints.append(obj_f)
            imgpoints.append(corners_f.astype(np.float32))

    return objpoints, imgpoints, image_size

def per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Her görüntü için ortalama Öklidyen reproj. hatası (px/corner, mean)."""
    errs = []
    for obj, img, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        diff = img.reshape(-1, 2) - proj.reshape(-1, 2)
        e = float(np.sqrt((diff * diff).sum(axis=1)).mean())
        errs.append(e)
    return np.array(errs, dtype=np.float32)

def simple_outlier_mask(errs, factor=1.5):
    if len(errs) < 8:
        return np.ones_like(errs, dtype=bool)
    med = np.median(errs)
    q1, q3 = np.percentile(errs, 25), np.percentile(errs, 75)
    iqr = q3 - q1
    thresh = med + factor * iqr
    m = errs <= thresh
    if m.sum() < max(6, int(0.6 * len(errs))):
        return np.ones_like(errs, dtype=bool)
    return m

def _sanity_limits(z):
    """k1/k2 ve RMS için kaba limitler."""
    if z <= 5:
        return 2.5, 10.0   # |k1|<=2.5, |k2|<=10
    else:
        return 5.0, 30.0   # |k1|<=5,   |k2|<=30

def is_sane_result(res, z):
    k1, k2 = abs(res["k1"]), abs(res["k2"])
    rms = float(res["reprojection_error"])
    lim_k1, lim_k2 = _sanity_limits(z)
    rms_lim = RMS_LIMIT_LOWZOOM if z <= 5 else RMS_LIMIT_HIGHZOOM
    return (k1 <= lim_k1) and (k2 <= lim_k2) and (rms <= rms_lim)

def debug_sanity(res, z):
    lim_k1, lim_k2 = (5.0, 30.0) if z>5 else (2.5, 10.0)
    rms_lim = 3.0 if z>5 else 2.5
    print(f"[sanity] z={z}  k1={abs(res['k1']):.3f}/{lim_k1} "
          f"k2={abs(res['k2']):.3f}/{lim_k2}  RMS={res['reprojection_error']:.2f}/{rms_lim}")

def calibrate_camera(objpoints, imgpoints, image_size,
                     fix_principal=False,
                     fix_k1=False, fix_k2=False,
                     iqr_factor=1.5):
    """
    Basit kalibrasyon:
      - K0: merkez, f0=0.9*min(W,H)
      - Tangential kapalı (opsiyonel)
      - K3 daima kilitli (FIX_K3_DEFAULT)
      - fix_k1/fix_k2 True ise ilgili katsayılar kilitlenir
      - Outlier temizliği: IQR faktörü ile
    """
    W, H = image_size
    f0 = 0.9 * min(W, H)
    K0 = np.array([[f0, 0, W/2.0],
                   [0,  f0, H/2.0],
                   [0,   0,   1.0]], dtype=np.float64)
    dist0 = np.zeros((5, 1), dtype=np.float64)

    flags = 0
    if FIX_K3_DEFAULT:
        flags |= cv2.CALIB_FIX_K3
    if ZERO_TANGENTIAL:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if fix_principal:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if fix_k1:
        flags |= cv2.CALIB_FIX_K1
    if fix_k2:
        flags |= cv2.CALIB_FIX_K2
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    # 1) Kalibrasyon
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, K0, dist0, flags=flags
    )
    d = dist.reshape(-1, 1)

    # 2) Outlier temizliği + tekrar
    errs = per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, d)
    mask = simple_outlier_mask(errs, factor=iqr_factor)
    if mask.sum() != len(errs):
        obj_f = [o for o, k in zip(objpoints, mask) if k]
        img_f = [i for i, k in zip(imgpoints, mask) if k]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_f, img_f, image_size, K, d, flags=flags
        )
        d = dist.reshape(-1, 1)
        errs = per_view_errors(obj_f, img_f, rvecs, tvecs, K, d)

    # Özet metrikler
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fx, fy = float(K[0, 0]), float(K[1, 1])

    # dist'ten isimli katsayıları çek (varsa)
    dist_list = dist.reshape(-1).astype(float).tolist()
    def _get(idx, default=0.0):
        return float(dist_list[idx]) if idx < len(dist_list) else default
    k1, k2, p1, p2, k3 = _get(0), _get(1), _get(2), _get(3), _get(4)

    # Uyarılar
    warns = []
    if abs(cx - W/2.0) > CENTER_TOL_RATIO * W:
        warns.append(f"cx merkezden sapmış: {cx:.2f} vs {W/2:.2f}")
    if abs(cy - H/2.0) > CENTER_TOL_RATIO * H:
        warns.append(f"cy merkezden sapmış: {cy:.2f} vs {H/2:.2f}")
    lim_k1, lim_k2 = _sanity_limits(parse_zoom_level("zoom_{}x".format(0)) or 0)
    # (uyarı limitleri sabit kalsın, temel eşikler aşağıda ayrıca var)
    if abs(k1) > 0.6: warns.append(f"|k1| yüksek: {k1:.3f}")
    if abs(k2) > 2.0: warns.append(f"|k2| yüksek: {k2:.3f}")

    return {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist_list,
        "reprojection_error": float(rms),
        "per_view_errors": [float(e) for e in errs.reshape(-1).tolist()],
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
        "warnings": warns
    }

def save_calibration_result(yaml_path, data_dict):
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_dict, f, sort_keys=False, allow_unicode=True)
    if data_dict.get("warnings"):
        print("   ⚠️ Uyarılar:", " | ".join(data_dict["warnings"]))
    print("💾 YAML kaydedildi →", yaml_path)

# ====== ANA DÖNGÜ ======
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folders = [d for d in os.listdir(CALIB_IMAGES_DIR)
           if d.lower().startswith("zoom_")
           and os.path.isdir(os.path.join(CALIB_IMAGES_DIR, d))]
    folders.sort(key=lambda f: parse_zoom_level(f) or 0)

    for folder in folders:
        if not folder.lower().startswith("zoom_"):
            continue
        folder_path = os.path.join(CALIB_IMAGES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        z = parse_zoom_level(folder) or 0
        print(f"\n🔍 İşleniyor: {folder}")

        images = load_images_from_folder(folder_path)
        if not images:
            print(f"❌ {folder}: Görüntü yok, atlandı.")
            continue

        try:
            image_size = assert_same_resolution(images)
        except Exception as e:
            print(f"❌ {folder}: {e}. Atlandı.")
            continue

        # Deforme kağıt için adaptif kenar filtresi
        border = EXCLUDE_BORDER if z < 4 else max(EXCLUDE_BORDER, 2)
        if z >= 6:
            border = max(border, 3)

        objpoints, imgpoints, _ = find_corners(
            images, PATTERN_SIZE, SQUARE_SIZE, use_sb=USE_SB, exclude_border=border
        )
        if len(objpoints) == 0:
            print(f"❌ {folder}: Köşe bulunamadı, atlandı.")
            continue

        # MODE A: standart (k1,k2 serbest; K3 kilit; tangential kapalı)
        fix_pp = (z >= HIGH_ZOOM_PRINCIPAL_FIX_FROM)
        res = calibrate_camera(
            objpoints, imgpoints, image_size,
            fix_principal=fix_pp,
            fix_k1=False, fix_k2=False,
            iqr_factor=(1.5 if z < 6 else 1.0)  # yüksek zoom'da outlier filtresi sertleşsin
        )
        mode = "standard"

        # sanity: kötü ise MODE B: k1-only (k2 kilit)
        if not is_sane_result(res, z) and z >= 6:
            print("   ↪ fallback: k1-only (K2 kilit)")
            # kenar filtresini daha da sertleştir ve yeniden köşe bul
            objpoints, imgpoints, _ = find_corners(
                images, PATTERN_SIZE, SQUARE_SIZE, use_sb=USE_SB, exclude_border=max(border, 3)
            )
            res = calibrate_camera(
                objpoints, imgpoints, image_size,
                fix_principal=True,   # tele'de şart
                fix_k1=False, fix_k2=True,
                iqr_factor=1.0
            )
            mode = "k1_only"

        # hâlâ kötü ise MODE C: no-distortion (k1,k2 kilit) — sadece intrinsics
        if not is_sane_result(res, z) and z >= 6:
            print("   ↪ fallback: no-distortion (K1,K2,K3=0; yalnız intrinsics)")
            objpoints, imgpoints, _ = find_corners(
                images, PATTERN_SIZE, SQUARE_SIZE, use_sb=USE_SB, exclude_border=max(border, 3)
            )
            res = calibrate_camera(
                objpoints, imgpoints, image_size,
                fix_principal=True,
                fix_k1=True, fix_k2=True,
                iqr_factor=0.8
            )
            mode = "no_distortion"
            debug_sanity(res, z)

        # YAML çıkışı
        out_path = os.path.join(OUTPUT_DIR, f"{folder}_calib.yaml")
        data = {
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
            "camera_matrix": res["camera_matrix"],
            "distortion_coefficients": res["distortion_coefficients"],
            "reprojection_error": res["reprojection_error"],
            "per_view_errors": res["per_view_errors"],
            "fx": res["fx"], "fy": res["fy"], "cx": res["cx"], "cy": res["cy"],
            "k1": res["k1"], "k2": res["k2"], "p1": res["p1"], "p2": res["p2"], "k3": res["k3"],
            "version": "simple_v3",
            "zoom_name": folder,
            "pattern_size": {"cols": PATTERN_SIZE[0], "rows": PATTERN_SIZE[1]},
            "square_size": SQUARE_SIZE,
            "model": mode,
            "settings": {
                "use_sb": USE_SB,
                "exclude_border": int(border),
                "fix_principal": bool(z >= HIGH_ZOOM_PRINCIPAL_FIX_FROM),
                "zero_tangential": ZERO_TANGENTIAL,
                "fallbacks": ["standard", "k1_only", "no_distortion"]
            },
            "warnings": res["warnings"]
        }
        save_calibration_result(out_path, data)

if __name__ == "__main__":
    main()
