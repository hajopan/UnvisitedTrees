"""
Deteksi Titik Bulatan Putih pada Gambar
========================================
Menggunakan OpenCV: threshold warna putih + Hough Circle Transform

Cara pakai:
    pip install opencv-python numpy
    python deteksi_bulatan_putih.py

Atau dengan gambar custom:
    python deteksi_bulatan_putih.py --input gambar_saya.png --output hasil.png
"""

import cv2
import numpy as np
import argparse
import sys


def deteksi_bulatan_putih(
    path_input: str,
    path_output: str = "hasil_deteksi.png",
    min_radius: int = 10,
    max_radius: int = 28,
    min_dist: int = 18,
    brightness_threshold: int = 180,
    saturation_max: int = 60,
):
    """
    Mendeteksi lingkaran/bulatan putih dalam gambar.

    Parameters
    ----------
    path_input          : path gambar input
    path_output         : path gambar hasil anotasi
    min_radius          : radius minimum bulatan (pixel)
    max_radius          : radius maksimum bulatan (pixel)
    min_dist            : jarak minimum antar pusat bulatan (pixel)
    brightness_threshold: nilai minimum kecerahan (0-255) untuk dianggap putih
    saturation_max      : nilai maksimum saturasi (0-255) untuk dianggap putih

    Returns
    -------
    list of dict : setiap item berisi {'id', 'x', 'y', 'radius'}
    """

    # ------------------------------------------------------------------ #
    # 1. Baca gambar
    # ------------------------------------------------------------------ #
    img = cv2.imread(path_input)
    if img is None:
        print(f"[ERROR] Gambar tidak ditemukan: {path_input}")
        sys.exit(1)

    print(f"[INFO] Gambar dibaca: {path_input}  ukuran={img.shape[1]}x{img.shape[0]} px")

    # ------------------------------------------------------------------ #
    # 2. Buat mask warna putih menggunakan HSV
    #    Putih = saturasi rendah + value (kecerahan) tinggi
    # ------------------------------------------------------------------ #
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,   0,                  brightness_threshold])
    upper_white = np.array([180, saturation_max,     255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # ------------------------------------------------------------------ #
    # 3. Blur ringan agar HoughCircles lebih stabil
    # ------------------------------------------------------------------ #
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 1)

    # ------------------------------------------------------------------ #
    # 4. Deteksi lingkaran dengan Hough Circle Transform
    #    param1 : sensitivitas edge detector Canny (lebih tinggi = lebih ketat)
    #    param2 : threshold akumulasi (lebih rendah = lebih banyak kandidat)
    # ------------------------------------------------------------------ #
    circles_raw = cv2.HoughCircles(
        mask_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=11,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles_raw is None:
        print("[INFO] Tidak ada bulatan putih yang terdeteksi.")
        print("[TIPS] Coba turunkan nilai brightness_threshold atau naikkan saturation_max.")
        return []

    circles = np.round(circles_raw[0]).astype(int)

    # ------------------------------------------------------------------ #
    # 5. Urutkan dari kiri-atas ke kanan-bawah (atas ke bawah, kiri ke kanan)
    # ------------------------------------------------------------------ #
    circles = sorted(circles, key=lambda c: (c[1] // 30, c[0]))  # grid ~30px

    # ------------------------------------------------------------------ #
    # 6. Gambar anotasi pada salinan gambar asli
    # ------------------------------------------------------------------ #
    hasil = img.copy()

    hasil_list = []
    for i, (x, y, r) in enumerate(circles, start=1):
        # Lingkaran merah
        cv2.circle(hasil, (x, y), r + 5, (0, 0, 220), 2)
        # Titik pusat
        cv2.circle(hasil, (x, y), 2, (0, 0, 220), -1)
        # Nomor urut
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.putText(
            hasil, label,
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 2, cv2.LINE_AA
        )
        hasil_list.append({"id": i, "x": int(x), "y": int(y), "radius": int(r)})

    # ------------------------------------------------------------------ #
    # 7. Teks ringkasan di pojok kiri atas
    # ------------------------------------------------------------------ #
    label_total = f"Total terdeteksi: {len(hasil_list)} bulatan"
    cv2.rectangle(hasil, (5, 5), (len(label_total) * 9 + 10, 28), (0, 0, 0), -1)
    cv2.putText(
        hasil, label_total,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # ------------------------------------------------------------------ #
    # 8. Simpan gambar hasil
    # ------------------------------------------------------------------ #
    cv2.imwrite(path_output, hasil)
    print(f"[INFO] Gambar hasil disimpan ke: {path_output}")

    # ------------------------------------------------------------------ #
    # 9. Cetak ringkasan ke terminal
    # ------------------------------------------------------------------ #
    print(f"\n{'='*45}")
    print(f"  Jumlah bulatan putih terdeteksi: {len(hasil_list)}")
    print(f"{'='*45}")
    print(f"  {'No':>3}  {'X':>5}  {'Y':>5}  {'Radius':>7}")
    print(f"  {'-'*30}")
    for item in hasil_list:
        print(f"  {item['id']:>3}  {item['x']:>5}  {item['y']:>5}  {item['radius']:>7} px")
    print(f"{'='*45}\n")

    return hasil_list


# ====================================================================== #
#  Main
# ====================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi bulatan putih pada gambar")
    parser.add_argument("--input",  default="gambar8.png",        help="Path gambar input")
    parser.add_argument("--output", default="hasil_deteksi.png",  help="Path gambar output")
    parser.add_argument("--min_radius",   type=int, default=10,   help="Radius minimum (default: 10)")
    parser.add_argument("--max_radius",   type=int, default=28,   help="Radius maksimum (default: 28)")
    parser.add_argument("--min_dist",     type=int, default=18,   help="Jarak min antar pusat (default: 18)")
    parser.add_argument("--brightness",   type=int, default=180,  help="Threshold kecerahan putih (default: 180)")
    parser.add_argument("--saturation",   type=int, default=60,   help="Max saturasi warna putih (default: 60)")
    args = parser.parse_args()

    deteksi_bulatan_putih(
        path_input=args.input,
        path_output=args.output,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_dist=args.min_dist,
        brightness_threshold=args.brightness,
        saturation_max=args.saturation,
    )