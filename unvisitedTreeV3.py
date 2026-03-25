"""
Deteksi Titik Bulatan Putih pada Gambar
========================================
Menggunakan pendekatan:
  1. Masking warna putih murni (HSV)
  2. Morphological opening (menghapus garis tipis, hanya menyisakan lingkaran solid)
  3. Distance Transform (mencari pusat tiap lingkaran)
  4. Non-Maximum Suppression (menghilangkan duplikat)

Cara pakai:
    pip install opencv-python numpy scipy
    python deteksi_bulatan_putih.py --input gambar.png --output hasil.png

Parameter bisa disesuaikan:
    python deteksi_bulatan_putih.py --input gambar.png --brightness 240 --saturation 20
"""

import cv2
import numpy as np
import argparse
import sys

try:
    from scipy import ndimage as ndi
except ImportError:
    print("[ERROR] scipy tidak terinstall. Jalankan: pip install scipy")
    sys.exit(1)


def deteksi_bulatan_putih(
    path_input: str,
    path_output: str = "hasil_deteksi.png",
    brightness_threshold: int = 240,
    saturation_max: int = 20,
    open_kernel_size: int = 12,
    nms_min_dist: int = 15,
    dist_threshold: float = 10.0,
):
    """
    Mendeteksi lingkaran/bulatan putih dalam gambar menggunakan
    Distance Transform + Non-Maximum Suppression.

    Keunggulan vs HoughCircles biasa:
    - Tidak terpengaruh garis putih tipis (jalur, kontur, dsb.)
    - Dapat memisahkan lingkaran yang berdekatan/berdempetan
    - Tidak mendeteksi lingkaran hijau atau warna lain

    Parameters
    ----------
    path_input          : path gambar input
    path_output         : path gambar hasil anotasi
    brightness_threshold: nilai minimum kecerahan HSV (0-255). Default 240 = putih murni
    saturation_max      : nilai maksimum saturasi HSV (0-255). Default 20 = hampir tidak berwarna
    open_kernel_size    : ukuran kernel opening untuk menghapus garis tipis.
                          Harus < diameter lingkaran target. Default 12
    nms_min_dist        : jarak minimum (pixel) antar pusat lingkaran. Default 15
    dist_threshold      : jarak minimum dari tepi blob agar dianggap pusat lingkaran. Default 10

    Returns
    -------
    list of dict : setiap item berisi {'id', 'x', 'y'}
    """

    # ------------------------------------------------------------------ #
    # 1. Baca gambar
    # ------------------------------------------------------------------ #
    img = cv2.imread(path_input)
    if img is None:
        print(f"[ERROR] Gambar tidak ditemukan: {path_input}")
        sys.exit(1)

    h_img, w_img = img.shape[:2]
    print(f"[INFO] Gambar dibaca: {path_input}  ukuran={w_img}x{h_img} px")

    # ------------------------------------------------------------------ #
    # 2. Masking warna putih murni menggunakan HSV
    #    Putih murni: saturasi sangat rendah (≈0) + kecerahan sangat tinggi (≈255)
    # ------------------------------------------------------------------ #
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,   0,                  brightness_threshold])
    upper_white = np.array([180, saturation_max,     255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # ------------------------------------------------------------------ #
    # 3. Morphological Opening — hapus garis tipis, simpan blob bulat
    #    Logika: garis tipis (lebar ~2px) akan dihancurkan oleh kernel besar,
    #    sementara lingkaran solid (diameter ~28px) tetap bertahan.
    # ------------------------------------------------------------------ #
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
    )
    mask_no_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Tutup celah kecil di dalam lingkaran
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_filled = cv2.morphologyEx(mask_no_lines, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # ------------------------------------------------------------------ #
    # 4. Distance Transform
    #    Nilai tiap pixel = jarak ke tepi putih terdekat.
    #    Pusat lingkaran = nilai lokal tertinggi (paling jauh dari tepi).
    # ------------------------------------------------------------------ #
    dist = cv2.distanceTransform(mask_filled, cv2.DIST_L2, 5)

    # ------------------------------------------------------------------ #
    # 5. Cari puncak lokal (local maxima) pada distance map
    # ------------------------------------------------------------------ #
    local_max_mask = (
        (dist == ndi.maximum_filter(dist, size=nms_min_dist * 2)) &
        (dist > dist_threshold)
    )
    raw_coords = np.column_stack(np.where(local_max_mask))  # format: [row, col]

    # ------------------------------------------------------------------ #
    # 6. Non-Maximum Suppression — hapus duplikat yang terlalu berdekatan
    # ------------------------------------------------------------------ #
    def nms_peaks(peaks, min_dist):
        kept = []
        for r, c in peaks:
            too_close = any(
                abs(r - kr) < min_dist and abs(c - kc) < min_dist
                for kr, kc in kept
            )
            if not too_close:
                kept.append((r, c))
        return kept

    peaks = nms_peaks(raw_coords.tolist(), nms_min_dist)

    # Urutkan dari kiri-atas ke kanan-bawah
    peaks_sorted = sorted(peaks, key=lambda p: (p[0] // 40, p[1]))

    # ------------------------------------------------------------------ #
    # 7. Gambar anotasi pada salinan gambar asli
    # ------------------------------------------------------------------ #
    def get_radius(r, c):
        val = dist[r, c]
        return max(12, min(int(val) + 5, 35))

    hasil = img.copy()
    hasil_list = []

    for i, (r, c) in enumerate(peaks_sorted, start=1):
        rad = get_radius(r, c)
        # Lingkaran merah
        cv2.circle(hasil, (c, r), rad + 4, (0, 0, 220), 2)
        # Titik pusat
        cv2.circle(hasil, (c, r), 2, (0, 0, 220), -1)
        # Nomor urut
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
        cv2.putText(
            hasil, label,
            (c - tw // 2, r + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 2, cv2.LINE_AA
        )
        hasil_list.append({"id": i, "x": c, "y": r})

    # ------------------------------------------------------------------ #
    # 8. Teks ringkasan di pojok kiri atas
    # ------------------------------------------------------------------ #
    label_total = f"Total terdeteksi: {len(hasil_list)} bulatan"
    cv2.rectangle(hasil, (5, 5), (len(label_total) * 9 + 10, 28), (0, 0, 0), -1)
    cv2.putText(
        hasil, label_total,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # ------------------------------------------------------------------ #
    # 9. Simpan gambar hasil
    # ------------------------------------------------------------------ #
    cv2.imwrite(path_output, hasil)
    print(f"[INFO] Gambar hasil disimpan ke: {path_output}")

    # ------------------------------------------------------------------ #
    # 10. Cetak ringkasan ke terminal
    # ------------------------------------------------------------------ #
    print(f"\n{'='*45}")
    print(f"  Jumlah bulatan putih terdeteksi: {len(hasil_list)}")
    print(f"{'='*45}")
    print(f"  {'No':>3}  {'X':>5}  {'Y':>5}")
    print(f"  {'-'*22}")
    for item in hasil_list:
        print(f"  {item['id']:>3}  {item['x']:>5}  {item['y']:>5}")
    print(f"{'='*45}\n")

    return hasil_list


# ====================================================================== #
#  Main
# ====================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deteksi bulatan putih pada gambar (kebal terhadap garis & warna lain)"
    )
    parser.add_argument("--input",       default="gambar.png",        help="Path gambar input")
    parser.add_argument("--output",      default="hasil_deteksi.png", help="Path gambar output")
    parser.add_argument("--brightness",  type=int,   default=240,
                        help="Threshold kecerahan putih (0-255). Default: 240")
    parser.add_argument("--saturation",  type=int,   default=20,
                        help="Max saturasi warna putih (0-255). Default: 20")
    parser.add_argument("--kernel",      type=int,   default=12,
                        help="Ukuran kernel opening untuk hapus garis tipis. Default: 12")
    parser.add_argument("--min_dist",    type=int,   default=15,
                        help="Jarak minimum antar pusat lingkaran (px). Default: 15")
    parser.add_argument("--dist_thresh", type=float, default=10.0,
                        help="Threshold distance transform untuk filter tepi. Default: 10.0")
    args = parser.parse_args()

    deteksi_bulatan_putih(
        path_input=args.input,
        path_output=args.output,
        brightness_threshold=args.brightness,
        saturation_max=args.saturation,
        open_kernel_size=args.kernel,
        nms_min_dist=args.min_dist,
        dist_threshold=args.dist_thresh,
    )