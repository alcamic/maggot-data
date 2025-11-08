import cv2
import os
import numpy as np
import sys
import joblib
import pandas as pd
from scipy.ndimage import binary_fill_holes
from sklearn.preprocessing import StandardScaler 
import warnings

warnings.filterwarnings('ignore')

# Fitur yang digunakan oleh model terbaik (HSV + Morfologi)
BEST_FEATURES_LIST = [
    "H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std",
    "area", "aspect_ratio", "solidity"
]

# --- FUNGSI HELPER (Tidak Berubah) ---
def bwareaopen(img, min_area=500):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1
    img_clean = np.zeros(output.shape, dtype=np.uint8)
    for i in range(nb_components):
        if sizes[i] >= min_area:
            img_clean[output == i + 1] = 255
    return img_clean

def adjust_contrast_brightness(img, contrast=1.5, brightness=30):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def select_largest_object_enhanced(binary_img, min_area_ratio=0.1):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels < 2:
        return np.zeros_like(binary_img)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    clean_mask = np.zeros_like(binary_img)
    clean_mask[labels == largest_label] = 255
    return clean_mask

def create_white_background(original_img, mask):
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if len(original_img.shape) < 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    white_bg = np.ones_like(original_img) * 255
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_normalized = mask_3ch.astype(float) / 255.0
    if original_img.shape != white_bg.shape:
        white_bg = cv2.resize(white_bg, (original_img.shape[1], original_img.shape[0]))
    if original_img.shape != mask_normalized.shape:
         mask_normalized = cv2.resize(mask_normalized, (original_img.shape[1], original_img.shape[0]))
    result = (original_img * mask_normalized + white_bg * (1 - mask_normalized)).astype(np.uint8)
    return result

def extract_color_features(img, mask):
    if np.sum(mask) == 0: 
        return {k: 0 for k in ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
                             'H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std',
                             'L_mean', 'A_mean', 'B_LAB_mean', 'L_std', 'A_std', 'B_LAB_std']}
    masked_region = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(masked_region)
    rgb_mean = [np.mean(r[mask > 0]), np.mean(g[mask > 0]), np.mean(b[mask > 0])]
    rgb_std = [np.std(r[mask > 0]), np.std(g[mask > 0]), np.std(b[mask > 0])]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    h, s, v = cv2.split(hsv_masked)
    hsv_mean = [np.mean(h[mask > 0]), np.mean(s[mask > 0]), np.mean(v[mask > 0])]
    hsv_std = [np.std(h[mask > 0]), np.std(s[mask > 0]), np.std(v[mask > 0])]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_masked = cv2.bitwise_and(lab, lab, mask=mask)
    l, a, b_lab = cv2.split(lab_masked)
    lab_mean = [np.mean(l[mask > 0]), np.mean(a[mask > 0]), np.mean(b_lab[mask > 0])]
    lab_std = [np.std(l[mask > 0]), np.std(a[mask > 0]), np.std(b_lab[mask > 0])]
    return {
        'R_mean': rgb_mean[0], 'G_mean': rgb_mean[1], 'B_mean': rgb_mean[2],
        'R_std': rgb_std[0], 'G_std': rgb_std[1], 'B_std': rgb_std[2],
        'H_mean': hsv_mean[0], 'S_mean': hsv_mean[1], 'V_mean': hsv_mean[2],
        'H_std': hsv_std[0], 'S_std': hsv_std[1], 'V_std': hsv_std[2],
        'L_mean': lab_mean[0], 'A_mean': lab_mean[1], 'B_LAB_mean': lab_mean[2],
        'L_std': lab_std[0], 'A_std': lab_std[1], 'B_LAB_std': lab_std[2]
    }

def extract_morphological_features(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {k: 0 for k in [
            "area", "perimeter", "rect_width", "rect_height",
            "aspect_ratio", "extent", "solidity",
            "major_axis", "minor_axis", "eccentricity", "equivalent_diameter"
        ]}
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if len(c) >= 5:
        (xc, yc), (MA, ma), angle = cv2.fitEllipse(c)
        major_axis, minor_axis = max(MA, ma), min(MA, ma)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
    else:
        major_axis = minor_axis = eccentricity = 0
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
    return {
        "area": area, "perimeter": perimeter, "rect_width": w, "rect_height": h,
        "aspect_ratio": aspect_ratio, "extent": extent, "solidity": solidity,
        "major_axis": major_axis, "minor_axis": minor_axis,
        "eccentricity": eccentricity, "equivalent_diameter": equivalent_diameter
    }
# === 3. FUNGSI PIPELINE UTAMA (Dimodifikasi) ===
def process_and_predict(img_original, artifacts_path):
    
    # === TAHAP 1: PREPROCESSING (Watershed Crop) ===
    try:
        blue = img_original[:, :, 0]
        blur = cv2.GaussianBlur(blue, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        nb, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img_original, markers)
        mask_ws = np.uint8(markers > 1) * 255
        contours, _ = cv2.findContours(mask_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise Exception("Watershed tidak menemukan kontur.")
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad = 10
        crop1 = img_original[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        gray_crop = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        blur_crop = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        _, otsu_crop = cv2.threshold(blur_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel2 = np.ones((3, 3), np.uint8)
        morph_crop = cv2.morphologyEx(otsu_crop, cv2.MORPH_CLOSE, kernel2, iterations=2)
        clean_crop = bwareaopen(morph_crop, min_area=800) 
        contours2, _ = cv2.findContours(clean_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours2) == 0:
            raise Exception("Otsu refinement tidak menemukan kontur.")
        c2 = max(contours2, key=cv2.contourArea)
        x2, y2, w2, h2 = cv2.boundingRect(c2)
        pad2 = 10
        img_cropped = crop1[max(0, y2 - pad2):y2 + h2 + pad2, max(0, x2 - pad2):x2 + w2 + pad2]
    except Exception as e:
        raise Exception(f"Error pada TAHAP 1 (Preprocessing): {e}")

    # === TAHAP 2: SEGMENTASI FINAL (Blue Channel + Morphology) ===
    try:
        blue = img_cropped[:, :, 0]
        blue_adj = adjust_contrast_brightness(blue, contrast=0.5, brightness=2)
        blur = cv2.GaussianBlur(blue_adj, (7, 7), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        disk20 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        erosi =  cv2.morphologyEx(otsu, cv2.MORPH_ERODE, disk20, iterations=2)
        dilasi = cv2.morphologyEx(erosi, cv2.MORPH_DILATE, disk20, iterations=1)
        opening = cv2.morphologyEx(dilasi, cv2.MORPH_OPEN, disk20, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, disk20, iterations=2)
        filled = binary_fill_holes(closing).astype(np.uint8) * 255
        final_mask = select_largest_object_enhanced(filled)
    except Exception as e:
        raise Exception(f"Error pada TAHAP 2 (Segmentasi): {e}")

    # === TAHAP 3: EKSTRAKSI FITUR ===
    try:
        color_features = extract_color_features(img_cropped, final_mask)
        morph_features = extract_morphological_features(final_mask)
        all_features_dict = {**color_features, **morph_features}
    except Exception as e:
        raise Exception(f"Error pada TAHAP 3 (Ekstraksi Fitur): {e}")

    # === TAHAP 4: PREDIKSI ===
    try:
        model = joblib.load(os.path.join(artifacts_path, "model_terbaik.joblib"))
        scaler = joblib.load(os.path.join(artifacts_path, "scaler_terbaik.joblib"))
        encoder = joblib.load(os.path.join(artifacts_path, "label_encoder.joblib"))

        df_single = pd.DataFrame([all_features_dict])
        df_single_ordered = df_single[BEST_FEATURES_LIST]
        features_scaled = scaler.transform(df_single_ordered)

        prediction_encoded = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        prediction_label = encoder.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100
        
        probabilities = {encoder.classes_[i]: prediction_proba[0][i]*100 for i in range(len(encoder.classes_))}
        
    except Exception as e:
        raise Exception(f"Error pada TAHAP 4 (Prediksi): {e}. Pastikan file model/scaler/encoder ada.")

    try:
        vis_img = create_white_background(img_cropped, final_mask)
    except Exception as e:
        vis_img = img_cropped
        print(f"Peringatan: Gagal membuat visualisasi akhir: {e}", file=sys.stderr)

    # === RETURN HASIL ===
    return prediction_label, confidence, probabilities, vis_img, img_cropped, final_mask