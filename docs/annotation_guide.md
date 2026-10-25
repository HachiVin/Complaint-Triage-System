# Panduan anotasi urgensi

Aspek dan sentimen sudah ada di dataset USH BRImo. Label **urgensi** dibuat sendiri di proyek ini,
karena itu pertanyaan yang benar-benar dihadapi tim customer experience setiap pagi:
*ulasan mana yang harus dibaca manusia hari ini?*

Satu ulasan = satu label. Kalau ragu di antara dua, pilih yang **lebih tinggi** (lebih murah membaca satu
ulasan tambahan daripada melewatkan nasabah yang uangnya hilang).

## Kritis: uang atau akun nasabah terdampak

| Contoh | Kenapa |
|---|---|
| "Transfer gagal tapi saldo terpotong, sudah 3 hari belum kembali" | Uang nasabah tertahan |
| "Ada transaksi yang tidak saya lakukan, saldo berkurang 2 juta" | Indikasi pembobolan |
| "Bayar tagihan terdebet dua kali" | Kelebihan potong |
| "Akun saya diambil alih, nomor HP diganti orang lain" | Akun dikuasai pihak lain |
| "Saya ditipu orang yang mengaku CS lewat WA" | Penipuan, perlu edukasi dan pelaporan |

## Tinggi: tidak bisa memakai layanan utama, uang belum terdampak

| Contoh | Kenapa |
|---|---|
| "Tidak bisa login sejak update" | Akses terputus |
| "OTP tidak masuk-masuk" | Tidak bisa transaksi |
| "Verifikasi wajah gagal terus, tidak bisa aktivasi" | Tidak bisa mulai memakai |
| "Aplikasi force close setiap dibuka" | Aplikasi tidak bisa dipakai |
| "Rekening terblokir, tidak tahu kenapa" | Akses ke dana terhambat |

## Normal: sisanya

Saran fitur, pujian, keluhan tampilan, lambat sesekali, biaya admin mahal, pertanyaan umum.

| Contoh | Label |
|---|---|
| "Aplikasi agak lambat kalau jam sibuk" | normal (masih bisa dipakai) |
| "Biaya admin transfer mahal" | normal (keluhan, bukan gangguan) |
| "Tolong tambah fitur dark mode" | normal |

## Kasus sulit

- **Rating 1 tanpa penjelasan** ("aplikasi sampah"): normal. Urgensi dinilai dari isi, bukan rating.
- **Masalah sudah selesai** ("kemarin saldo terpotong tapi sudah kembali, terima kasih"): normal.
- **Masalah lama tapi masih terjadi** ("dari bulan lalu dana belum kembali"): kritis.
- **Tidak bisa login dan saldo berkurang**: kritis (ambil yang paling berat).
- **Bahasa daerah / campuran**: beri label seperti biasa; tandai *skip* hanya kalau benar-benar tidak paham.

## Alur kerja

1. `python scripts/labelstudio.py sample data/interim/brimo.jsonl data/interim/ls_urgency.json --n 800`
   Sampel sengaja diperkaya: 20% ditandai aturan sebagai kritis, 20% sebagai tinggi, 35% ulasan negatif
   lainnya, dan 25% sisanya acak, karena ulasan kritis jarang. Distribusi asli dilaporkan terpisah di `docs/results.md`.
2. Buat proyek di Label Studio, tempel `docs/labelstudio_config.xml`, impor `ls_urgency.json`.
   Tebakan aturan sudah terpilih; tugas anotator memeriksa dan mengubah bila salah.
3. Ekspor JSON, lalu
   `python scripts/labelstudio.py merge data/interim/brimo.jsonl ls_export.json data/annotated/brimo.jsonl`.
4. **Cek konsistensi**: anotasi ulang 100 ulasan acak seminggu kemudian tanpa melihat label lama,
   hitung kesepakatan (Cohen's kappa). Tulis angkanya di README.

Catatan: tebakan aturan memang membantu kecepatan, tapi bisa membuat anotator ikut-ikutan aturan.
Karena itu evaluasi aturan vs model di test set tetap dilaporkan, dan kasus di mana anotator
**mengubah** tebakan aturan dicatat jumlahnya.