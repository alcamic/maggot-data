import cv2
import numpy as np
import sys
import os
from scipy.ndimage import binary_fill_holes
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

# === KONFIGURASI DASAR ===
base_input = "Single Larvae data/Citra2"
folders = ["Grade A", "Grade B", "Grade C"]

output_blue = os.path.join(base_input, "folder_konversi")
output_otsu = os.path.join(base_input, "folder_otsu")
output_morph = os.path.join(base_input, "folder_morfologi")
output_bwareaopen = os.path.join(base_input, "folder_bwareaopen")
output_rgb = os.path.join(base_input, "folder_rgb_overlay")
output_white_bg = os.path.join(base_input, "folder_white_background")
output_features = os.path.join(base_input, "folder_features")
output_final = os.path.join(base_input, "folder_final_larva_only")
output_vis_morph = os.path.join(base_input, "folder_visualisasi_morfologi")

os.makedirs(output_vis_morph, exist_ok=True)
os.makedirs(output_blue, exist_ok=True)
os.makedirs(output_otsu, exist_ok=True)
os.makedirs(output_bwareaopen, exist_ok=True)
os.makedirs(output_rgb, exist_ok=True)
os.makedirs(output_white_bg, exist_ok=True)
os.makedirs(output_features, exist_ok=True)
os.makedirs(output_final, exist_ok=True)

morph_subfolders = ["opening", "closing","dilasi","erosi", "hole_filling"]
for sub in morph_subfolders:
    os.makedirs(os.path.join(output_morph, sub), exist_ok=True)

rgb_subfolders = ["opening", "closing", "hole_filling", "bwareaopen", "final"]
for sub in rgb_subfolders:
    os.makedirs(os.path.join(output_rgb, sub), exist_ok=True)

white_bg_subfolders = ["opening", "closing", "hole_filling", "bwareaopen", "final"]
for sub in white_bg_subfolders:
    os.makedirs(os.path.join(output_white_bg, sub), exist_ok=True)

# === FUNGSI PENINGKATAN KONTRAS DAN BRIGHTNESS ===
def adjust_contrast_brightness(img, contrast=1.5, brightness=30):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

# === FUNGSI BARU: MEMILIH OBJEK TERBESAR + CLEANING ===
def select_largest_object_enhanced(binary_img, min_area_ratio=0.1):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels < 2:
        return np.zeros_like(binary_img)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    clean_mask = np.zeros_like(binary_img)
    clean_mask[labels == largest_label] = 255
    return clean_mask

# === FUNGSI RGB OVERLAY ===
def create_rgb_overlay(original_img, mask):
    overlay = original_img.copy()
    red_mask = np.zeros_like(original_img)
    red_mask[:, :, 2] = mask
    result = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
    return result

# === FUNGSI WHITE BACKGROUND ===
def create_white_background(original_img, mask):
    white_bg = np.ones_like(original_img) * 255
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_normalized = mask_3ch.astype(float) / 255.0
    result = (original_img * mask_normalized + white_bg * (1 - mask_normalized)).astype(np.uint8)
    return result

# === FUNGSI EKSTRAKSI FITUR WARNA ===
def extract_color_features(img, mask):
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

# === FUNGSI EKSTRAKSI MORFOLOGI ===
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

# === VISUALISASI MORFOLOGI ===
def visualize_morphological_features(img, mask, features_dict, save_path):
    vis = img.copy()
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imwrite(save_path, vis)
        return
    c = max(contours, key=cv2.contourArea)
    
    # Gambar kontur asli objek (warna kuning tebal)
    cv2.drawContours(vis, [c], -1, (0, 255, 255), 3)
    
    # Bounding rectangle (warna biru)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convex hull (warna hijau)
    hull = cv2.convexHull(c)
    cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)
    
    # Fitted ellipse (warna merah)
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(vis, ellipse, (0, 0, 255), 2)
        
        # Gambar garis sumbu mayor dan minor
        (xc, yc), (MA, ma), angle = ellipse
        angle_rad = np.deg2rad(angle)
        
        # Sumbu mayor (warna ungu)
        major_len = MA / 2
        x1_major = int(xc + major_len * np.cos(angle_rad))
        y1_major = int(yc + major_len * np.sin(angle_rad))
        x2_major = int(xc - major_len * np.cos(angle_rad))
        y2_major = int(yc - major_len * np.sin(angle_rad))
        cv2.line(vis, (x1_major, y1_major), (x2_major, y2_major), (255, 0, 255), 2)
        
        # Sumbu minor (warna cyan)
        minor_len = ma / 2
        x1_minor = int(xc - minor_len * np.sin(angle_rad))
        y1_minor = int(yc + minor_len * np.cos(angle_rad))
        x2_minor = int(xc + minor_len * np.sin(angle_rad))
        y2_minor = int(yc - minor_len * np.cos(angle_rad))
        cv2.line(vis, (x1_minor, y1_minor), (x2_minor, y2_minor), (255, 255, 0), 2)
        
        # Centroid
        cv2.circle(vis, (int(xc), int(yc)), 5, (0, 165, 255), -1)
    
    # Teks informasi dengan background
    teks = [
        f"Area: {features_dict['area']:.1f}",
        f"Perimeter: {features_dict['perimeter']:.1f}",
        f"Aspect: {features_dict['aspect_ratio']:.2f}",
        f"Solidity: {features_dict['solidity']:.2f}",
        f"Eccentricity: {features_dict['eccentricity']:.2f}"
    ]
    
    # Tambahkan legenda
    legend = [
        "Yellow: Contour",
        "Blue: Bounding Box",
        "Green: Convex Hull",
        "Red: Fitted Ellipse",
        "Purple: Major Axis",
        "Cyan: Minor Axis"
    ]
    
    y0, dy = 25, 25
    # Background semi-transparan untuk teks
    overlay = vis.copy()
    cv2.rectangle(overlay, (5, 5), (300, y0 + (len(teks) + len(legend)) * dy + 10), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)
    
    # Gambar teks features
    for i, line in enumerate(teks):
        y_text = y0 + i * dy
        cv2.putText(vis, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Gambar teks legend
    for i, line in enumerate(legend):
        y_text = y0 + (len(teks) + i) * dy
        cv2.putText(vis, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.imwrite(save_path, vis)
    
def extract_all_features(img, mask, filename, grade, stage):
    color_features = extract_color_features(img, mask)
    morph_features = extract_morphological_features(mask)
    all_features = {
        'filename': filename,
        'grade': grade,
        'stage': stage,
        **color_features,
        **morph_features
    }
    return all_features


# === PROSES MASSAL UNTUK SEMUA FOLDER ===
all_features_list = []

for folder in folders:
    input_dir = os.path.join(base_input, folder)
    blue_dir = os.path.join(output_blue, folder)
    otsu_dir = os.path.join(output_otsu, folder)
    bwareaopen_dir = os.path.join(output_bwareaopen, folder)
    final_dir = os.path.join(output_final, folder)

    os.makedirs(blue_dir, exist_ok=True)
    os.makedirs(otsu_dir, exist_ok=True)
    os.makedirs(bwareaopen_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    for sub in morph_subfolders:
        os.makedirs(os.path.join(output_morph, sub, folder), exist_ok=True)

    for sub in rgb_subfolders:
        os.makedirs(os.path.join(output_rgb, sub, folder), exist_ok=True)

    for sub in white_bg_subfolders:
        os.makedirs(os.path.join(output_white_bg, sub, folder), exist_ok=True)

    print(f"\n Memproses folder: {folder}")

    for f in os.listdir(input_dir):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. KONVERSI KE BLUE CHANNEL + KONTRAS/BRIGHTNESS
        blue = img[:, :, 0]
        blue_adj = adjust_contrast_brightness(blue, contrast=0.5, brightness=2)
        cv2.imwrite(os.path.join(blue_dir, f), blue_adj)

        # 2. SEGMENTASI OTSU
        blur = cv2.GaussianBlur(blue_adj, (7, 7), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(otsu_dir, f), otsu)

        # 3. STRUKTUR ELEMEN
        disk20 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        # 4. OPENING 
        # 4.1 Erosi
        erosi =  cv2.morphologyEx(otsu, cv2.MORPH_ERODE, disk20, iterations=2)
        cv2.imwrite(os.path.join(output_morph, "erosi", f), erosi)
        # 4.2 Dilasi
        dilasi = cv2.morphologyEx(erosi, cv2.MORPH_DILATE, disk20, iterations=1)
        cv2.imwrite(os.path.join(output_morph, "dilasi", f), dilasi)
        # 4.3 Opening 
        opening = cv2.morphologyEx(dilasi, cv2.MORPH_OPEN, disk20, iterations=2)
        cv2.imwrite(os.path.join(output_morph, "opening", f), opening)

        # 5. CLOSING
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, disk20, iterations=2)
        cv2.imwrite(os.path.join(output_morph, "closing", f), closing)

        # 6. HOLE FILLING
        filled = binary_fill_holes(closing).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_morph, "hole_filling", f), filled)

        # 7. BWAREAOPEN
        final_larva = select_largest_object_enhanced(filled)
        cv2.imwrite(os.path.join(bwareaopen_dir, f), final_larva)

        rgb_bwareaopen = create_rgb_overlay(img, final_larva)
        cv2.imwrite(os.path.join(output_rgb, "bwareaopen", folder, f), rgb_bwareaopen)

        white_bg_bwareaopen = create_white_background(img, final_larva)
        cv2.imwrite(os.path.join(output_white_bg, "bwareaopen", folder, f), white_bg_bwareaopen)

        features_bwareaopen = extract_all_features(img, final_larva, f, folder, 'bwareaopen')
        all_features_list.append(features_bwareaopen)

        # 8. SMOOTHING BERTAHAP
        disk15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        final_larva = cv2.morphologyEx(final_larva, cv2.MORPH_CLOSE, disk15, iterations=2)
        final_larva = binary_fill_holes(final_larva).astype(np.uint8) * 255

        disk8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        final_larva = cv2.morphologyEx(final_larva, cv2.MORPH_OPEN, disk8, iterations=1)

        disk12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        final_larva = cv2.morphologyEx(final_larva, cv2.MORPH_CLOSE, disk12, iterations=1)

        cv2.imwrite(os.path.join(final_dir, f), final_larva)

        rgb_final = create_rgb_overlay(img, final_larva)
        cv2.imwrite(os.path.join(output_rgb, "final", folder, f), rgb_final)

        white_bg_final = create_white_background(img, final_larva)
        cv2.imwrite(os.path.join(output_white_bg, "final", folder, f), white_bg_final)

        features_final = extract_all_features(img, final_larva, f, folder, 'final')
        all_features_list.append(features_final)

    print(f" Selesai memproses {folder}")


# === VISUALISASI FITUR ===
output_vis = os.path.join(base_input, "folder_visualisasi_fitur")
os.makedirs(output_vis, exist_ok=True)
for folder in folders:
    os.makedirs(os.path.join(output_vis, folder), exist_ok=True)


def save_feature_visual(img, mask, features_dict, save_path):
    white_bg = create_white_background(img, mask)
    overlay = white_bg.copy()
    teks = [
        "=== COLOR FEATURES ===",
        f"R: {features_dict['R_mean']:.1f}, G: {features_dict['G_mean']:.1f}, B: {features_dict['B_mean']:.1f}",
        f"H: {features_dict['H_mean']:.1f}, S: {features_dict['S_mean']:.1f}, V: {features_dict['V_mean']:.1f}",
        f"L: {features_dict['L_mean']:.1f}, A: {features_dict['A_mean']:.1f}, B_LAB: {features_dict['B_LAB_mean']:.1f}"
    ]
    y0, dy = 30, 25
    for i, line in enumerate(teks):
        y = y0 + i * dy
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, overlay)


print("\n Menyimpan visualisasi ekstraksi fitur...")

for feat in all_features_list:
    filename = feat['filename']
    grade = feat['grade']
    stage = feat['stage']

    img_path = os.path.join(base_input, grade, filename)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    mask_path = os.path.join(output_final if stage == 'final' else output_bwareaopen, grade, filename)
    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    save_dir = os.path.join(output_vis, grade, stage)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    save_feature_visual(img, mask, feat, save_path)

    morph_vis_dir = os.path.join(base_input, "folder_visualisasi_morfologi", grade, stage)
    os.makedirs(morph_vis_dir, exist_ok=True)
    morph_vis_path = os.path.join(morph_vis_dir, filename)
    visualize_morphological_features(img, mask, feat, morph_vis_path)

print("   Visualisasi tersimpan di folder_visualisasi_fitur")


# === SIMPAN CSV ===
print("\n Menyimpan fitur ke CSV...")
df_features = pd.DataFrame(all_features_list)
df_bwareaopen = df_features[df_features["stage"] == "bwareaopen"]
df_final = df_features[df_features["stage"] == "final"]

csv_bwareaopen = os.path.join(output_features, "bwareaopen_features.csv")
csv_final = os.path.join(output_features, "final_features.csv")

df_bwareaopen.to_csv(csv_bwareaopen, index=False)
df_final.to_csv(csv_final, index=False)

print(f" Bwareaopen features: {csv_bwareaopen}")
print(f" Final features: {csv_final}")


# === SUMMARY ===
print("\n" + "="*60)
print("=== SUMMARY EKSTRAKSI FITUR ===")
print("="*60)
print(f"Total gambar diproses: {len(all_features_list)//2}")
print(f"Fitur Warna (RGB, HSV, LAB): 18 fitur")
print(f"Fitur Tekstur GLCM (Grayscale): 6 fitur")
print(f"Fitur Morfologi: 11 fitur")
print(f"Total Fitur per Gambar: 35 fitur")
print("\n SEMUA PROSES SELESAI ")
print("="*60)