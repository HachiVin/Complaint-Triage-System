"""Keyword rules for urgency.

Two jobs: pre-label reviews in Label Studio so annotation is mostly confirming, and act as the baseline
the urgency model must beat. Rules are not used as ground truth: a human confirms every label.
"""

from __future__ import annotations

import re

_KRITIS = [
    r"saldo\s*(?:saya\s*)?(?:ke)?potong", r"saldo\s*(?:berkurang|hilang|raib|terdebet)", r"dana\s*(?:hilang|raib|nyangkut|tertahan)",
    r"uang\s*(?:saya\s*)?(?:hilang|raib|nyangkut|tertahan|ga(?:k)?\s*kembali|belum\s*kembali)",
    r"(?:transfer|tf|transaksi|top\s*up|bayar|pembayaran)\s*(?:\w+\s){0,3}gagal\s*(?:\w+\s){0,4}(?:saldo|uang|potong)",
    r"terpotong\s*(?:tapi|tetapi|namun)", r"double\s*(?:debet|potong)", r"terdebet\s*(?:dua|2)\s*kali",
    r"transaksi\s*(?:yang\s*)?(?:tidak|tak|gak|ga)\s*(?:saya\s*)?(?:kenal|lakukan)", r"dibobol", r"kebobolan",
    r"akun\s*(?:saya\s*)?(?:diambil\s*alih|dihack|di\s*hack|kena\s*hack)", r"penipuan", r"\bscam\b",
]
_TINGGI = [
    r"(?:tidak|tak|gak|ga|nggak|gk)\s*bisa\s*(?:login|masuk|log\s*in|dibuka|buka\s*aplikasi|transfer|transaksi)",
    r"gagal\s*(?:login|masuk|verifikasi|aktivasi|registrasi|daftar)", r"otp\s*(?:tidak|tak|gak|ga|belum)\s*(?:masuk|dikirim|muncul)",
    r"force\s*close", r"keluar\s*sendiri", r"(?:akun|rekening)\s*(?:terblokir|diblokir|ter\s*blokir)",
    r"verifikasi\s*wajah\s*(?:gagal|terus|tidak)", r"error\s*terus", r"selalu\s*error",
]
KRITIS_RE = re.compile("|".join(_KRITIS), re.IGNORECASE)
TINGGI_RE = re.compile("|".join(_TINGGI), re.IGNORECASE)


def urgency_by_rules(text: str) -> tuple[str, list[str]]:
    """Return (urgency, matched phrases)."""
    hits = [m.group(0) for m in KRITIS_RE.finditer(text)]
    if hits:
        return "kritis", hits
    hits = [m.group(0) for m in TINGGI_RE.finditer(text)]
    if hits:
        return "tinggi", hits
    return "normal", []
