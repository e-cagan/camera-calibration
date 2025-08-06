import os
import cv2
import numpy as np
import yaml

def load_images_from_folder(folder_path):
    """Dosyadan görüntüleri yükleme fonksiyonu."""

    images = []
    
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg', '.png')):  # sadece resimleri al
            continue
        
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images.append((img, gray))

    return images


def find_corners(images, pattern_size):
    """Köşeleri bulma fonksiyonu."""

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []


    #  3D points real world coordinates
    objectp3d = np.zeros((1, pattern_size[0] 
                        * pattern_size[1], 
                        3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:pattern_size[0],
                                0:pattern_size[1]].T.reshape(-1, 2)
    
    image_size = None

    for img, gray in images:
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        gray, pattern_size, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(img, 
                                            pattern_size, 
                                            corners2, ret)
            
            cv2.imshow("Chessboard Corners", image)
            cv2.waitKey(100)

            if image_size is None:
                image_size = gray.shape[::-1]
    
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    return threedpoints, twodpoints, image_size


def calibrate_camera(threedpoints, twodpoints, image_size):
    """Kamera kalibrasyonu fonksiyonu."""
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        threedpoints, twodpoints, image_size, None, None
    )

    total_error = 0
    for i in range(len(threedpoints)):
        imgpoints2, _ = cv2.projectPoints(threedpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(threedpoints)

    return {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "reprojection_error": mean_error
    }


def save_calibration_result(yaml_path, data_dict):
    """Kalibrasyon sonuçlarını kaydetme fonksiyonu."""

    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(data_dict, f, sort_keys=False)
            print("YAML dosyasına kaydedildi.")
    except Exception as error:
        print(f"YAML dosyasına kaydedilirken hata oluştu: {error}")
