# Sistem Prediksi Stok Pupuk Bersubsidi (LSTM) — Streamlit App

Implementasi interaktif dari metodologi skripsi:
*"Implementasi Long Short-Term Memory (LSTM) untuk Prediksi Stok Pupuk Urea dan NPK Phonska
Bersubsidi sebagai Dasar Perencanaan Safety Stock dan Reorder Point"*
(Studi Kasus: Kios Tani Gunung Salak, Kalapanunggal, Kabupaten Sukabumi).

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Lalu buka `http://localhost:8501` di browser.

## Struktur alur (5 tab)

1. **Data & Preprocessing** — upload CSV/Excel, pemetaan kolom otomatis (bisa disesuaikan manual),
   pembersihan data (konversi tanggal, hapus duplikat, clipping ≥0, forward/backward-fill).
2. **Eksplorasi Data** — statistik deskriptif, distribusi musim tanam, grafik tren stok harian.
3. **Safety Stock & ROP** — hitung Safety Stock (Z × σ × √Lead Time) dan Reorder Point,
   dengan lead time & service level yang bisa diatur.
4. **Training & Evaluasi LSTM** — melatih Stacked LSTM (64→32 unit, Dropout 0.2, Dense 16→1)
   untuk Urea dan/atau NPK Phonska pada skenario timesteps 7/14/21/30 hari, dengan
   EarlyStopping & ReduceLROnPlateau. Menampilkan tabel evaluasi (MAE, RMSE, MAPE, wMAPE,
   SMAPE, R²), pemilihan model terbaik (wMAPE terendah), loss curve, dan grafik aktual vs prediksi.
5. **Prediksi & Rekomendasi** — memprediksi stok periode berikutnya menggunakan model terbaik,
   membandingkannya dengan Safety Stock/ROP, dan memberi status **Aman** / **Waspada**.

## Format data yang diharapkan

Data harian dengan kolom (nama bebas, dipetakan manual di tab 1):

| Peran | Contoh nama kolom |
|---|---|
| Tanggal | `tanggal` |
| Musim tanam | `musim_tanam` (isi: MT1/MT2/MT3) |
| Penyaluran Urea | `penyaluran_urea` |
| Penyaluran NPK Phonska | `penyaluran_npk` |
| Sisa stok Urea | `stok_urea` |
| Sisa stok NPK Phonska | `stok_npk` |

Belum punya data siap? Klik **"Gunakan data contoh (demo)"** di tab 1 untuk mencoba seluruh
pipeline dengan data sintetis terlebih dahulu.

## File

- `app.py` — antarmuka Streamlit (5 tab di atas)
- `utils.py` — logika pipeline: preprocessing, rekayasa fitur musiman, Safety Stock/ROP,
  arsitektur LSTM, metrik evaluasi, generator data contoh
- `requirements.txt` — daftar dependensi

## Catatan

- Training berjalan langsung di dalam app (bisa memakan waktu beberapa menit per skenario,
  tergantung jumlah data dan epoch). Untuk pengujian cepat, kurangi "Maks. Epoch" di
  pengaturan lanjutan pada tab 4.
- Kategori akurasi wMAPE mengikuti skala umum: <10% Sangat Baik, 10–20% Baik, 20–50% Cukup,
  >50% Kurang (skala Lewis, 1982) — sesuai yang dipakai pada Bab IV skripsi.
