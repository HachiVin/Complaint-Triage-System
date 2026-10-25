"""Generate the fictional demo reviews in data/sample/demo_reviews.jsonl.

Two made-up apps ("DemoBank Mobile" and "KitaPay", not real products), ten weeks of reviews, and a
planted incident: version 3.2.0 of DemoBank breaks transfers in week 7, so the dashboard and agent have a
trend to find. Every row is marked synthetic. Used for tests, the demo model and the live demo until the
real dataset is loaded; never for reported results.

    python scripts/make_demo_data.py
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample" / "demo_reviews.jsonl"

# (aspect, sentiment, urgency, rating range, templates)
TEMPLATES = [
    ("kualitas_transaksi", "negatif", "kritis", (1, 1), [
        "Transfer gagal tapi saldo saya kepotong {amt}, sudah {days} hari belum kembali",
        "Saldo terpotong tapi transaksi gagal, uang saya nyangkut. Tolong segera dikembalikan",
        "Top up e-wallet {amt} gagal, saldo berkurang dan tidak ada refund sampai sekarang",
        "Bayar tagihan listrik gagal tapi saldo terdebet dua kali, mohon dicek",
    ]),
    ("keamanan_privasi", "negatif", "kritis", (1, 1), [
        "Ada transaksi yang tidak saya lakukan sebesar {amt}, akun saya dibobol",
        "Saldo hilang {amt} padahal saya tidak transaksi apa pun, tolong blokir dan usut",
    ]),
    ("kualitas_sistem", "negatif", "tinggi", (1, 2), [
        "Tidak bisa login sejak update kemarin, selalu error",
        "Aplikasi force close terus setiap buka menu transfer",
        "Tidak bisa dibuka sama sekali, layar putih lalu keluar sendiri",
    ]),
    ("keamanan_privasi", "negatif", "tinggi", (1, 2), [
        "OTP tidak masuk masuk, sudah coba berkali kali",
        "Verifikasi wajah gagal terus padahal cahaya sudah terang",
    ]),
    ("kualitas_sistem", "negatif", "normal", (2, 3), [
        "Aplikasinya agak lambat kalau jam sibuk", "Kadang loading lama waktu cek mutasi",
    ]),
    ("kemudahan_penggunaan", "negatif", "normal", (2, 3), [
        "Menu terlalu banyak, susah cari fitur transfer", "Tampilan baru membingungkan, lebih enak yang lama",
    ]),
    ("biaya_risiko", "negatif", "normal", (2, 3), [
        "Biaya admin transfer ke bank lain kemahalan", "Kenapa ada potongan bulanan padahal jarang dipakai",
    ]),
    ("layanan_pelanggan", "negatif", "normal", (1, 2), [
        "Call center susah dihubungi, nunggu lama", "Komplain lewat chat CS tidak ada balasan",
    ]),
    ("fitur_inovasi", "netral", "normal", (3, 4), [
        "Tolong tambahkan fitur jadwal transfer otomatis", "Semoga ada fitur pisah kantong tabungan",
    ]),
    ("kualitas_informasi", "netral", "normal", (3, 3), [
        "Notifikasi transaksi kadang telat masuk", "Keterangan di mutasi kurang jelas",
    ]),
    ("kemudahan_penggunaan", "positif", "normal", (4, 5), [
        "Aplikasinya mudah dipakai dan cepat", "Mantap, transfer jadi gampang", "Tampilannya rapi dan simpel",
    ]),
    ("kualitas_transaksi", "positif", "normal", (5, 5), [
        "Transfer cepat masuk, tidak pernah gagal", "Bayar QRIS lancar terus",
    ]),
    ("umum", "positif", "normal", (5, 5), ["Mantap", "Bagus sekali aplikasinya", "Sangat membantu"]),
    ("umum", "negatif", "normal", (1, 2), ["Jelek", "Aplikasi tidak jelas", "Kecewa"]),
]
WEIGHTS = [6, 2, 4, 3, 3, 3, 3, 3, 3, 2, 6, 3, 5, 3]


def main() -> None:
    rng = random.Random(7)
    start = date(2026, 7, 6)
    rows = []
    n = 0
    for app, per_week, versions in (("DemoBank Mobile", 28, ["3.0.0", "3.1.0", "3.2.0"]),
                                    ("KitaPay", 14, ["1.8.2", "1.9.0"])):
        for week in range(10):
            count = per_week
            weights = list(WEIGHTS)
            if app == "DemoBank Mobile" and week in (6, 7):  # incident: 3.2.0 breaks transfers
                count += 14
                weights[0] *= 4
            for _ in range(count):
                tpl = rng.choices(TEMPLATES, weights=weights)[0]
                aspect, sentiment, urgency, (lo, hi), texts = tpl
                text = rng.choice(texts).format(amt=f"Rp{rng.choice([50, 150, 250, 500, 1200])}.000",
                                                days=rng.randint(2, 6))
                if rng.random() < 0.3:
                    text += rng.choice([" min", ", tolong diperbaiki", " 🙏", ". Gimana ini?"])
                d = start + timedelta(days=7 * week + rng.randint(0, 6))
                if app == "DemoBank Mobile":
                    version = versions[2] if week >= 6 else versions[1] if week >= 3 else versions[0]
                else:
                    version = versions[1] if week >= 5 else versions[0]
                n += 1
                rows.append({"id": f"demo-{n:04d}", "app": app, "text": text, "rating": rng.randint(lo, hi),
                             "date": d.isoformat(), "app_version": version, "aspect": aspect, "sentiment": sentiment,
                             "urgency": urgency, "source": "synthetic", "synthetic": True})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} fictional reviews -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
