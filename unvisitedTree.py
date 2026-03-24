import cv2
import numpy as np

# ─────────────────────────────────────────────
# PARAMETER – sesuaikan jika hasil kurang tepat
# ─────────────────────────────────────────────
IMAGE_PATH      = "Images/gambar2.png"
OUTPUT_PATH     = "hasil_deteksi.png"

# Threshold warna putih
BRIGHTNESS_MIN  = 200      # piksel dengan V (HSV) >= ini dianggap "putih"
SATURATION_MAX  = 50       # piksel dengan S (HSV) <= ini dianggap "putih"

# Watershed tuning
DIST_THRESHOLD  = 0.35     # 0.0–1.0: makin kecil makin mudah pisah titik
                            # coba turunkan ke 0.25 jika titik masih bergabung
MIN_AREA        = 50       # area minimum per titik (buang noise)
MAX_AREA        = 8000     # area maksimum per titik
MORPH_KERNEL    = 3        # kernel closing awal
# ─────────────────────────────────────────────


def hitung_titik_putih(image_path: str, output_path: str | None = None) -> int:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # ── 1. Isolasi warna putih via HSV ──────────────────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_white = cv2.bitwise_and(
        cv2.threshold(v, BRIGHTNESS_MIN, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(s, SATURATION_MAX, 255, cv2.THRESH_BINARY_INV)[1],
    )

    # ── 2. Morphological closing kecil ──────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask_closed = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    # ── 3. Distance Transform + Watershed ───────────────────────────────────
    # Distance transform: tiap piksel = jarak ke tepi terdekat
    dist = cv2.distanceTransform(mask_closed, cv2.DIST_L2, 5)

    # Lokasi "puncak" = pusat tiap titik putih
    _, dist_thresh = cv2.threshold(
        dist, DIST_THRESHOLD * dist.max(), 255, cv2.THRESH_BINARY
    )
    dist_thresh = np.uint8(dist_thresh)

    # Label connected components dari puncak-puncak tersebut
    num_markers, markers = cv2.connectedComponents(dist_thresh)

    # Watershed memisahkan titik yang menempel
    markers = markers + 1                          # background = 1, bukan 0
    markers[mask_closed == 0] = 0                  # area non-putih = unknown
    markers_ws = cv2.watershed(img, markers.copy())

    # ── 4. Hitung & filter region hasil watershed ───────────────────────────
    valid_labels = []
    for label in range(2, num_markers + 1):        # label 1 = background
        region = (markers_ws == label).astype(np.uint8)
        area = int(region.sum())
        if MIN_AREA <= area <= MAX_AREA:
            valid_labels.append(label)

    count = len(valid_labels)

    # ── 5. Visualisasi ───────────────────────────────────────────────────────
    vis = img.copy()
    for i, label in enumerate(valid_labels):
        region = np.uint8(markers_ws == label) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), max(int(radius), 8)

        cv2.circle(vis, (cx, cy), radius + 4, (0, 255, 0), 2)
        cv2.putText(
            vis, str(i + 1),
            (cx - 6, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1, cv2.LINE_AA,
        )

    # Label total
    label_text = f"Titik terdeteksi: {count}"
    cv2.rectangle(vis, (5, 5), (len(label_text) * 10 + 10, 28), (0, 0, 0), -1)
    cv2.putText(vis, label_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Hasil disimpan: {output_path}")

    # ── 6. Debug info ────────────────────────────────────────────────────────
    print(IMAGE_PATH)
    print(f"\n{'─'*40}")
    print(f"  Total region watershed : {num_markers - 1}")
    print(f"  Lolos filter area      : {count}")
    print(f"  Jumlah titik putih     : {count}")
    print(f"{'─'*40}\n")

    return count


if __name__ == "__main__":
    hitung_titik_putih(IMAGE_PATH, OUTPUT_PATH)