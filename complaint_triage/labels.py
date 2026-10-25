"""The three label sets the model predicts for every review.

* ``ASPECTS``: the nine service aspects of the USH BRImo dataset, so its labels can be used directly.
* ``SENTIMENTS``: negative, neutral, positive.
* ``URGENCY``: added by this project and annotated by hand. It answers the question a CX team actually
  has each morning: which reviews must a person read today?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    key: str
    name: str
    description: str


ASPECT_LABELS: tuple[Label, ...] = (
    Label("kualitas_transaksi", "Kualitas transaksi",
          "Transfer, pembayaran, top-up, tarik tunai: gagal, tertunda, saldo terpotong, salah nominal."),
    Label("kualitas_sistem", "Kualitas sistem",
          "Aplikasi error, force close, lambat, gangguan server, gagal update."),
    Label("keamanan_privasi", "Keamanan dan privasi",
          "Akun dibobol, transaksi tidak dikenal, OTP, PIN, verifikasi wajah, data pribadi."),
    Label("kemudahan_penggunaan", "Kemudahan penggunaan",
          "Tampilan, navigasi, alur yang membingungkan, aksesibilitas."),
    Label("fitur_inovasi", "Fitur dan inovasi", "Permintaan atau penilaian fitur, fitur baru, fitur yang hilang."),
    Label("biaya_risiko", "Biaya dan nilai finansial", "Biaya admin, potongan, limit, promo, bunga."),
    Label("layanan_pelanggan", "Layanan pelanggan", "Call center, CS, respons pengaduan, kantor cabang."),
    Label("kualitas_informasi", "Kualitas informasi",
          "Informasi tidak jelas, notifikasi, mutasi atau riwayat yang membingungkan."),
    Label("umum", "Pengalaman umum", "Komentar umum tanpa aspek spesifik: 'mantap', 'aplikasi jelek'."),
)

SENTIMENT_LABELS: tuple[Label, ...] = (
    Label("negatif", "Negatif", "Keluhan atau ketidakpuasan."),
    Label("netral", "Netral", "Pertanyaan, saran, atau informasi tanpa penilaian jelas."),
    Label("positif", "Positif", "Pujian atau kepuasan."),
)

URGENCY_LABELS: tuple[Label, ...] = (
    Label("kritis", "Kritis",
          "Uang atau akun nasabah terdampak: saldo terpotong tapi transaksi gagal, dana hilang, transaksi "
          "tidak dikenal, akun diambil alih. Harus ditangani hari ini."),
    Label("tinggi", "Tinggi",
          "Nasabah tidak bisa memakai layanan utama: tidak bisa login, OTP tidak masuk, aplikasi tidak bisa "
          "dibuka, verifikasi gagal terus. Uang belum terdampak."),
    Label("normal", "Normal", "Saran, pujian, keluhan tampilan, pertanyaan umum, keluhan ringan."),
)

ASPECTS: tuple[str, ...] = tuple(x.key for x in ASPECT_LABELS)
SENTIMENTS: tuple[str, ...] = tuple(x.key for x in SENTIMENT_LABELS)
URGENCY: tuple[str, ...] = tuple(x.key for x in URGENCY_LABELS)

TASKS: dict[str, tuple[str, ...]] = {"aspect": ASPECTS, "sentiment": SENTIMENTS, "urgency": URGENCY}
LABEL_INFO: dict[str, dict[str, Label]] = {
    "aspect": {x.key: x for x in ASPECT_LABELS},
    "sentiment": {x.key: x for x in SENTIMENT_LABELS},
    "urgency": {x.key: x for x in URGENCY_LABELS},
}

# Raw values seen in source datasets -> our keys. Extend this when a new source uses other spellings;
# importing fails loudly on any value that is not listed, so nothing is mapped by accident.
ASPECT_ALIASES: dict[str, str] = {
    "transaction quality": "kualitas_transaksi", "kualitas transaksi": "kualitas_transaksi",
    "system quality": "kualitas_sistem", "kualitas sistem": "kualitas_sistem",
    "security and privacy": "keamanan_privasi", "security & privacy": "keamanan_privasi",
    "keamanan dan privasi": "keamanan_privasi",
    "usability": "kemudahan_penggunaan", "kemudahan penggunaan": "kemudahan_penggunaan",
    "features and innovation": "fitur_inovasi", "features & innovation": "fitur_inovasi",
    "fitur dan inovasi": "fitur_inovasi",
    "financial value and risk": "biaya_risiko", "financial value & risk": "biaya_risiko",
    "nilai finansial dan risiko": "biaya_risiko",
    "customer service": "layanan_pelanggan", "layanan pelanggan": "layanan_pelanggan",
    "information quality": "kualitas_informasi", "kualitas informasi": "kualitas_informasi",
    "other or general experience": "umum", "other / general experience": "umum", "other": "umum",
    "general": "umum", "umum": "umum",
    # USH BRImo dataset codes (column "Aspek")
    "tq": "kualitas_transaksi", "sq": "kualitas_sistem", "sp": "keamanan_privasi", "us": "kemudahan_penggunaan",
    "fi": "fitur_inovasi", "fv": "biaya_risiko", "cs": "layanan_pelanggan", "iq": "kualitas_informasi",
    "ot": "umum",
}
SENTIMENT_ALIASES: dict[str, str] = {
    "negative": "negatif", "negatif": "negatif", "neg": "negatif", "-1": "negatif",
    "neutral": "netral", "netral": "netral", "neu": "netral", "0": "netral",
    "positive": "positif", "positif": "positif", "pos": "positif", "1": "positif",
}


def normalise_label(task: str, raw) -> str | None:
    """Map a raw dataset value to our key. ``None`` for empty values; ``ValueError`` for unknown ones."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.lower() in ("nan", "none", "null"):
        return None
    if value in TASKS[task]:
        return value
    aliases = {"aspect": ASPECT_ALIASES, "sentiment": SENTIMENT_ALIASES, "urgency": {}}[task]
    key = aliases.get(value.lower().replace("_", " "))
    if key is None:
        raise ValueError(f"unknown {task} value {value!r}; add it to the aliases in complaint_triage/labels.py")
    return key
