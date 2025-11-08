import cv2, os, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

# === PATH FOLDER ===
base_input = r"Single Larvae data"
base_output = r"Single Larvae data/Citra2"
folders = ["Grade A", "Grade B", "Grade C"]

# Fungsi mirip MATLAB bwareaopen (hapus area kecil)
def bwareaopen(img, min_area=500):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1
    img_clean = np.zeros(output.shape, dtype=np.uint8)
    for i in range(nb_components):
        if sizes[i] >= min_area:
            img_clean[output == i + 1] = 255
    return img_clean


# === PROSES SETIAP FOLDER ===
for folder in folders:
    input_dir = os.path.join(base_input, folder)
    output_dir = os.path.join(base_output, f"{folder}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n Memproses folder: {folder}")

    for f in os.listdir(input_dir):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # === 1. Baca gambar ===
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None:
            print(f"Gagal membaca {f}")
            continue

        # === 2. Tahap pertama: Segmentasi dengan WATERSHED ===
        blue = img[:, :, 0]
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
        markers = cv2.watershed(img, markers)
        mask = np.uint8(markers > 1) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"Tidak ada kontur di {f}")
            continue

        # === 3. Crop hasil dari watershed ===
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad = 10
        crop1 = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]

        # === 4. Tahap kedua: Otsu + Morphology + bwareaopen pada hasil crop ===
        gray_crop = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        blur_crop = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        _, otsu = cv2.threshold(blur_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphology Close untuk menutup lubang kecil
        kernel2 = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel2, iterations=2)

        # Hapus area kecil (bwareaopen)
        clean = bwareaopen(morph, min_area=800)

        # Temukan kontur dari hasil refined mask
        contours2, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours2) == 0:
            print(f"Tidak ada larva pada tahap Otsu di {f}")
            continue

        # Ambil kontur terbesar lagi
        c2 = max(contours2, key=cv2.contourArea)
        x2, y2, w2, h2 = cv2.boundingRect(c2)
        pad2 = 10
        crop2 = crop1[max(0, y2 - pad2):y2 + h2 + pad2, max(0, x2 - pad2):x2 + w2 + pad2]

        # === 5. Simpan hasil akhir ===
        save_path = os.path.join(output_dir, f)
        cv2.imwrite(save_path, crop2)
        print(f"{f} → Watershed + Otsu refinement crop selesai")

print("\n=== Semua gambar selesai diproses (Watershed + Otsu + bwareaopen)! ===")
