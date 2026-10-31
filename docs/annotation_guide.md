# Panduan anotasi urgensi

Aspek dan sentimen sudah ada di dataset USH BRImo. Label **urgensi** dibuat sendiri di proyek ini,
karena itu pertanyaan yang benar-benar dihadapi tim customer experience setiap pagi:
*ulasan mana yang harus dibaca manusia hari ini?*

**Tidak perlu keahlian keuangan.** Urgensi di sini bukan penilaian hukum atau finansial (siapa yang
salah, apakah bank wajib mengganti). Yang dinilai hanya **apa yang dialami nasabah menurut ulasannya**,
lewat dua pertanyaan ya/tidak. Definisi ini adalah SOP proyek ini: yang penting **konsisten**, bukan
"benar menurut ahli".

## Dua pertanyaan, berurutan

```
1. Apakah UANG atau AKUN nasabah terdampak?
   (saldo terpotong / berkurang, dana hilang atau tertahan, terdebet dua kali,
    transaksi yang tidak dia lakukan, akun dikuasai orang lain, uang terkirim ke penipu)
        ya  → KRITIS
        tidak ↓
2. Apakah nasabah TIDAK BISA MEMAKAI aplikasi / layanan utama sama sekali?
   (tidak bisa login, OTP tidak masuk, aplikasi tidak bisa dibuka atau keluar sendiri,
    verifikasi gagal terus, rekening terblokir, tidak bisa transfer sama sekali)
        ya  → TINGGI
        tidak ↓
3. Selain itu → NORMAL
```

Kalau ragu di antara dua label setelah menjawab kedua pertanyaan, pilih yang **lebih tinggi** (lebih
murah membaca satu ulasan tambahan daripada melewatkan nasabah yang uangnya hilang). Kalau tetap tidak
yakin, centang **Ragu** (lihat bawah).

## Kritis: uang atau akun nasabah terdampak

| Contoh | Kenapa |
|---|---|
| "Transfer gagal tapi saldo terpotong, sudah 3 hari belum kembali" | Uang nasabah tertahan |
| "Ada transaksi yang tidak saya lakukan, saldo berkurang 2 juta" | Indikasi pembobolan |
| "Bayar tagihan terdebet dua kali" | Kelebihan potong |
| "Akun saya diambil alih, nomor HP diganti orang lain" | Akun dikuasai pihak lain |
| "Saya ditipu orang yang mengaku CS, uang saya sudah terkirim" | Uang hilang karena penipuan |

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
| "Kenapa harus tutup aplikasi lain dulu baru bisa transaksi" | normal (merepotkan, tapi tetap bisa) |

## Kasus sulit

- **Rating 1 tanpa penjelasan** ("aplikasi sampah"): normal. Urgensi dinilai dari isi, bukan rating.
- **Masalah sudah selesai** ("kemarin saldo terpotong tapi sudah kembali, terima kasih"): normal.
- **Masalah lama tapi masih terjadi** ("dari bulan lalu dana belum kembali"): kritis.
- **Tidak bisa login dan saldo berkurang**: kritis (ambil yang paling berat).
- **"Sering error" tanpa keterangan tidak bisa dipakai**: normal. Tinggi hanya kalau jelas tidak bisa
  dipakai ("error terus, tidak bisa transaksi sama sekali").
- **Takut / khawatir tanpa kejadian** ("takut saldo hilang karena banyak yang dibobol"): normal.
  Yang dinilai kejadian yang dialami, bukan kekhawatiran.
- **Bahasa daerah / campuran / singkatan**: beri label seperti biasa kalau maksudnya bisa ditangkap.

## Ragu

Centang **Ragu** (tombol `4`) kalau setelah membaca panduan ini kamu tetap tidak bisa memutuskan,
misalnya kalimatnya tidak jelas apakah uangnya benar-benar hilang. Tetap pilih label tebakan terbaikmu.

Ulasan yang ditandai ragu **tidak dipakai** untuk training maupun test, dan dikumpulkan di file
`data/annotated/brimo.ragu.jsonl` untuk dibahas lalu dilabeli ulang. Lebih baik membuang 30 ulasan
yang tidak jelas daripada memasukkan label tebakan ke test set. Jumlahnya ditulis di README.

Kalau ulasan sejenis sering muncul di daftar ragu, itu tanda panduan ini kurang jelas: tambahkan aturan
baru di bagian *Kasus sulit*, lalu periksa ulang ulasan yang sudah dilabeli.

## Jalan pintas: pemeriksaan silang

Kalau waktu terbatas, sampel diberi **dua tebakan independen**: aturan kata kunci dan anotator kedua
(di proyek ini LLM, `data/annotation/second_opinion.jsonl`).

```
python scripts/labelstudio.py crosscheck data/interim/ls_urgency.json \
    data/annotation/second_opinion.jsonl data/interim/ls_urgency_crosscheck.json
```

- **Periode test: selalu diputuskan manusia**, apa pun tebakannya. Angka evaluasi harus dibandingkan
  dengan keputusan manusia, bukan dengan mesin lain.
- Di luar periode test, kalau aturan dan anotator kedua **sepakat**, label langsung diterima (diimpor
  sebagai sudah dianotasi, `cek = tidak`).
- Kalau keduanya **berbeda** atau anotator kedua **ragu**: `cek = ya`, dikerjakan manusia kalau sempat.
  Yang tidak sempat dikerjakan tidak diberi label (tidak dipakai).
- Urutan daftar di Label Studio sudah diatur: test dulu, lalu validasi, lalu training, lalu yang sudah
  diterima di paling bawah. Kerjakan dari atas, berhenti kapan saja setelah semua ulasan periode test
  selesai. Kolom `alasan_cek` menjelaskan kenapa ulasan perlu dicek.
- Label yang sudah diterima tetap bisa dibuka dan diubah kalau terlihat salah.

## Alur kerja

1. `python scripts/labelstudio.py sample data/interim/brimo.jsonl data/interim/ls_urgency.json --n 800`
   Sampel sengaja diperkaya: 20% ditandai aturan sebagai kritis, 20% sebagai tinggi, 35% ulasan negatif
   lainnya, dan 25% sisanya acak, karena ulasan kritis jarang. Distribusi asli dilaporkan terpisah di
   `docs/results.md`.
2. Buat proyek di Label Studio, tempel `docs/labelstudio_config.xml`, impor `ls_urgency.json`.
   Tebakan aturan sudah terpilih; tugas anotator memeriksa dan mengubah bila salah.
   Urutan tugas: **periode test dulu, lalu validasi, lalu training** (kolom `periode`). Kerjakan dari
   atas. Kalau waktu terbatas, pakai jalan pintas pemeriksaan silang di atas.
3. Ekspor JSON, lalu
   `python scripts/labelstudio.py merge data/interim/brimo.jsonl data/interim/ls_export.json data/annotated/brimo.jsonl --tasks data/interim/ls_urgency_crosscheck.json`.
   Skrip mencetak berapa label yang berbeda dari tebakan aturan dan berapa yang ditandai ragu.
4. **Cek konsistensi**: anotasi ulang 100 ulasan acak seminggu kemudian tanpa melihat label lama,
   hitung kesepakatan (Cohen's kappa). Tulis angkanya di README.

Catatan: tebakan aturan memang membantu kecepatan, tapi bisa membuat anotator ikut-ikutan aturan.
Karena itu evaluasi aturan vs model di test set tetap dilaporkan, dan jumlah label yang **diubah** dari
tebakan aturan dicatat.

## Test set v0.2: diputuskan manusia, tanpa petunjuk

Label urgensi test di v0.1 berasal dari LLM (silver). Di v0.2 test set dilabeli ulang oleh manusia, dengan
aturan berikut supaya hasilnya bisa dipercaya:

1. `python scripts/labelstudio.py testset data/interim/brimo.jsonl data/interim/ls_testset.json --tasks data/interim/ls_urgency.json --n 400`
   Isinya 400 ulasan periode test: semua ulasan periode test dari sampel v0.1 (supaya label silver bisa
   dibandingkan dengan label manusia) ditambah ulasan baru yang diperkaya dengan cara yang sama. Keduanya
   diacak, jadi anotator tidak tahu mana yang mana.
2. **Tanpa petunjuk (blind).** Tugas tidak membawa tebakan aturan, label LLM, kata pemicu, rating, aspek,
   maupun sentimen. Yang terlihat hanya teks, tanggal dan versi aplikasi. Jangan membuka label lama
   selama menganotasi.
3. Buat **proyek Label Studio baru** (jangan dicampur dengan proyek v0.1), tempel
   `docs/labelstudio_config.xml`, impor `ls_testset.json`. Tombol: `1` kritis, `2` tinggi, `3` normal,
   `4` ragu, `Ctrl+Enter` simpan. Kerjakan semua 400, sekitar 100 per sesi supaya tetap fokus.
4. Ekspor JSON ke `data/interim/ls_testset_export.json`, lalu
   `python scripts/labelstudio.py human data/interim/ls_testset_export.json data/annotation/human_test.jsonl --tasks data/interim/ls_testset.json`.
   File hasil hanya berisi id dan label, jadi boleh di-commit.
5. `python scripts/evaluate.py data/processed --baseline models/baseline --multitask models/indobert-mt --human data/annotation/human_test.jsonl`
   Urgensi dinilai hanya terhadap label manusia (yang ragu tidak dihitung), dan skrip melaporkan seberapa
   sering label silver v0.1 sama dengan label manusia (persentase dan Cohen's kappa).

Batasan yang tetap ditulis di README: satu anotator, dan sampel diperkaya ulasan yang ditandai aturan,
jadi angkanya bukan distribusi asli seluruh ulasan.