"""
Modul pendukung untuk Sistem Prediksi Stok Pupuk Bersubsidi (LSTM).
Mengimplementasikan pipeline sesuai Bab III & Bab IV skripsi:
- Preprocessing & rekayasa fitur musiman
- Perhitungan Safety Stock & Reorder Point
- Arsitektur Stacked LSTM
- Evaluasi model (MAE, RMSE, MAPE, wMAPE, SMAPE, R2)
- Prediksi periode berikutnya & rekomendasi status stok
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    _TF_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    tf = None
    _TF_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


# ----------------------------------------------------------------------
# Deteksi kolom otomatis (agar app fleksibel terhadap penamaan kolom user)
# ----------------------------------------------------------------------

COLUMN_HINTS = {
    "tanggal": ["tanggal", "date", "tgl"],
    "musim_tanam": ["musim_tanam", "musim tanam", "musim", "mt"],
    "penyaluran_urea": ["penyaluran_urea", "penyaluran urea", "salur_urea", "urea_out", "distribusi_urea"],
    "penyaluran_npk": ["penyaluran_npk", "penyaluran npk", "penyaluran_npk_phonska", "salur_npk", "npk_out", "distribusi_npk"],
    "stok_urea": ["sisa_stok_urea", "sisa stok urea", "stok_urea", "stock_urea"],
    "stok_npk": ["sisa_stok_npk", "sisa stok npk phonska", "sisa_stok_npk_phonska", "stok_npk", "stok_npk_phonska", "stock_npk"],
}


def _normalize_col(name):
    import re
    s = str(name).strip().lower()
    s = re.sub(r"\(.*?\)", " ", s)          # buang satuan dalam kurung, misal "(Kg)"
    s = re.sub(r"[^a-z0-9]+", "_", s)       # non-alnum -> underscore
    return s.strip("_")


def guess_column_mapping(columns):
    """Menebak pemetaan kolom user ke peran yang dibutuhkan pipeline."""
    normalized = {c: _normalize_col(c) for c in columns}
    mapping = {}
    for role, hints in COLUMN_HINTS.items():
        found = None
        best_score = 0
        for col, norm in normalized.items():
            for hint in hints:
                h = _normalize_col(hint)
                if h == norm:
                    found, best_score = col, 100
                    break
                if h in norm or norm in h:
                    score = len(h)
                    if score > best_score:
                        found, best_score = col, score
            if best_score == 100:
                break
        mapping[role] = found
    return mapping


# ----------------------------------------------------------------------
# Preprocessing & rekayasa fitur (Sub-bab 3.6.1 & 4.2)
# ----------------------------------------------------------------------

def preprocess_dataframe(df, col_map):
    """
    Menjalankan tahapan pra-pemrosesan sesuai Sub-bab 4.2.1:
    (1) konversi tanggal & urutkan kronologis
    (2) hapus duplikasi tanggal
    (3) clipping nilai numerik >= 0
    (4) forward-fill / backward-fill, sisanya diisi 0
    Mengembalikan dataframe bersih berkolom baku:
    tanggal, musim_tanam, penyaluran_urea, penyaluran_npk, stok_urea, stok_npk
    """
    out = pd.DataFrame()
    out["tanggal"] = pd.to_datetime(df[col_map["tanggal"]], errors="coerce", dayfirst=True)
    out["musim_tanam"] = df[col_map["musim_tanam"]].astype(str).str.strip().str.upper()
    numeric_roles = ["penyaluran_urea", "penyaluran_npk", "stok_urea", "stok_npk"]
    for role in numeric_roles:
        col = col_map[role]
        series = df[col]
        if series.dtype == object:
            series = (
                series.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^0-9\.\-]", "", regex=True)
            )
        out[role] = pd.to_numeric(series, errors="coerce")

    n_before = len(out)
    out = out.dropna(subset=["tanggal"]).sort_values("tanggal")
    duplicated = int(out["tanggal"].duplicated().sum())
    out = out.drop_duplicates(subset="tanggal", keep="last").reset_index(drop=True)

    for role in numeric_roles:
        out[role] = out[role].clip(lower=0)
        out[role] = out[role].ffill().bfill().fillna(0)

    info = {
        "n_baris_awal": n_before,
        "n_baris_akhir": len(out),
        "tanggal_duplikat_dihapus": duplicated,
        "tanggal_mulai": out["tanggal"].min(),
        "tanggal_akhir": out["tanggal"].max(),
    }
    return out, info


def add_seasonal_features(df):
    """Rekayasa fitur musiman (Sub-bab 4.2.2): bulan_sin/cos + one-hot musim tanam."""
    d = df.copy()
    d["bulan"] = d["tanggal"].dt.month
    d["bulan_sin"] = np.sin(2 * np.pi * d["bulan"] / 12)
    d["bulan_cos"] = np.cos(2 * np.pi * d["bulan"] / 12)

    musim_dummies = pd.get_dummies(d["musim_tanam"], prefix="musim").astype(int)
    for expected in ["musim_MT1", "musim_MT2", "musim_MT3"]:
        if expected not in musim_dummies.columns:
            musim_dummies[expected] = 0
    musim_dummies = musim_dummies[[c for c in musim_dummies.columns if c.startswith("musim_")]]
    d = pd.concat([d, musim_dummies], axis=1)
    return d


FEATURE_COLUMNS_TEMPLATE = ["bulan_sin", "bulan_cos"]  # + musim one-hot + penyaluran fitur


def get_feature_columns(df):
    musim_cols = sorted([c for c in df.columns if c.startswith("musim_") and c != "musim_tanam"])
    return FEATURE_COLUMNS_TEMPLATE + musim_cols


# ----------------------------------------------------------------------
# Safety Stock & Reorder Point (Sub-bab 3.6.2 / 4.4)
# ----------------------------------------------------------------------

def compute_safety_stock(df, stok_col, lead_time_days, service_level):
    """
    Formula: Safety Stock = Z * sigma * sqrt(Lead Time)
    sigma dihitung dari penurunan stok harian ternormalisasi jarak hari
    antar observasi berurutan (Sub-bab 4.4).
    """
    z = stats.norm.ppf(service_level)
    stok = df[stok_col].values
    tanggal = df["tanggal"].values

    delta_stok = -np.diff(stok)  # penurunan stok positif berarti konsumsi
    delta_hari = np.diff(tanggal).astype("timedelta64[D]").astype(float)
    delta_hari[delta_hari == 0] = 1
    penurunan_harian = delta_stok / delta_hari
    # Hari dengan kenaikan stok (restocking) tidak dihitung sebagai "penurunan" (diset 0),
    # sehingga rata-rata & variabilitas mencerminkan kecepatan konsumsi riil, bukan tercampur
    # dengan lonjakan pengisian ulang dari distributor.
    penurunan_harian = np.clip(penurunan_harian, 0, None)


    rata_penurunan = float(np.mean(penurunan_harian))
    sigma = float(np.std(penurunan_harian))

    safety_stock = z * sigma * np.sqrt(lead_time_days)
    rop = rata_penurunan * lead_time_days + safety_stock

    return {
        "z": z,
        "rata_penurunan_harian": rata_penurunan,
        "sigma": sigma,
        "safety_stock": safety_stock,
        "rop": rop,
    }


def stock_status(prediksi, rop, threshold=1.2):
    batas_waspada = rop * threshold
    if prediksi >= batas_waspada:
        return "Aman"
    else:
        return "Waspada"


# ----------------------------------------------------------------------
# Sequence building untuk LSTM (sliding window)
# ----------------------------------------------------------------------

def build_sequences(scaled_target, scaled_features, timesteps):
    """
    scaled_target: array (n,) target (stok) sudah dinormalisasi
    scaled_features: array (n, f) fitur eksogen sudah dinormalisasi (boleh None -> 0 kolom)
    Mengembalikan X (n-timesteps, timesteps, 1+f), y (n-timesteps,)
    Fitur historis stok (window) digabung dengan fitur eksogen pada tiap langkah waktu.
    """
    n = len(scaled_target)
    X, y = [], []
    for i in range(timesteps, n):
        window_target = scaled_target[i - timesteps:i].reshape(-1, 1)
        if scaled_features is not None and scaled_features.shape[1] > 0:
            window_feat = scaled_features[i - timesteps:i]
            window = np.concatenate([window_target, window_feat], axis=1)
        else:
            window = window_target
        X.append(window)
        y.append(scaled_target[i])
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# Arsitektur Stacked LSTM (Sub-bab 4.5.1)
# ----------------------------------------------------------------------

def build_lstm_model(timesteps, n_features, units=(64, 32), dropout=0.2, dense_units=16, lr=0.001):
    if tf is None:
        raise RuntimeError(
            "TensorFlow tidak berhasil di-import di environment ini.\n"
            f"Detail error asli: {_TF_IMPORT_ERROR}"
        )
    model = Sequential([
        LSTM(units[0], activation="tanh", return_sequences=True, input_shape=(timesteps, n_features)),
        Dropout(dropout),
        LSTM(units[1], activation="tanh"),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mae")
    return model


def get_callbacks(es_patience=35, rlr_patience=12):
    if tf is None:
        return []
    return [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=rlr_patience, factor=0.5),
    ]


# ----------------------------------------------------------------------
# Metrik evaluasi (Sub-bab 2.6 / 4.6)
# ----------------------------------------------------------------------

def evaluate_predictions(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))

    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")

    denom = np.sum(np.abs(actual))
    wmape = float(np.sum(np.abs(actual - pred)) / denom * 100) if denom != 0 else float("nan")

    smape_denom = (np.abs(actual) + np.abs(pred))
    smape_denom[smape_denom == 0] = 1e-9
    smape = float(np.mean(2 * np.abs(actual - pred) / smape_denom) * 100)

    r2 = float(r2_score(actual, pred))

    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape, "wMAPE": wmape, "SMAPE": smape, "R2": r2,
        "Kategori": wmape_category(wmape),
    }


def wmape_category(wmape):
    """Skala akurasi peramalan berbasis persentase error (Lewis, 1982)."""
    if wmape < 10:
        return "Sangat Baik"
    elif wmape < 20:
        return "Baik"
    elif wmape < 50:
        return "Cukup"
    else:
        return "Kurang"


# ----------------------------------------------------------------------
# Data contoh (sintetis) — hanya untuk mencoba app sebelum data asli diunggah
# ----------------------------------------------------------------------

def generate_demo_data(n_days=885, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-28")
    dates = pd.date_range(start, periods=n_days, freq="D")

    musim_cycle = (["MT1"] * 130 + ["MT2"] * 90 + ["MT3"] * 60)
    musim = []
    while len(musim) < n_days:
        musim.extend(musim_cycle)
    musim = musim[:n_days]

    def simulate_stock(base_cap, mean_use):
        stock = np.zeros(n_days)
        distribusi = np.zeros(n_days)
        level = base_cap * rng.uniform(0.6, 1.0)
        for i in range(n_days):
            if level < base_cap * 0.15 or rng.random() < 0.03:
                level += base_cap * rng.uniform(0.7, 1.0)
            use = max(0, rng.normal(mean_use, mean_use * 1.4))
            if rng.random() < 0.35:
                use = 0
            use = min(use, level)
            level -= use
            stock[i] = level
            distribusi[i] = use
        return stock, distribusi

    stok_urea, sal_urea = simulate_stock(4956, 366)
    stok_npk, sal_npk = simulate_stock(4974, 393)

    df = pd.DataFrame({
        "tanggal": dates,
        "musim_tanam": musim,
        "penyaluran_urea": np.round(sal_urea, 2),
        "penyaluran_npk": np.round(sal_npk, 2),
        "stok_urea": np.round(stok_urea, 2),
        "stok_npk": np.round(stok_npk, 2),
    })
    return df