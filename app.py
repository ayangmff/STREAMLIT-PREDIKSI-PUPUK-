import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from utils import (
    guess_column_mapping, preprocess_dataframe, add_seasonal_features,
    get_feature_columns, compute_safety_stock, stock_status,
    build_sequences, build_lstm_model, get_callbacks, evaluate_predictions,
    generate_demo_data,
)

st.set_page_config(
    page_title="Prediksi Stok Pupuk Bersubsidi — LSTM",
    page_icon="🌾",
    layout="wide",
)

# ----------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------
for key, default in [
    ("df_clean", None), ("prep_info", None), ("df_feat", None),
    ("ss_urea", None), ("ss_npk", None),
    ("results", {}), ("trained_models", {}), ("scalers", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🌾 Sistem Prediksi Stok Pupuk Bersubsidi (LSTM)")
st.caption(
    "Implementasi Skripsi: *Implementasi Long Short-Term Memory (LSTM) untuk Prediksi Stok Pupuk Urea dan "
    "NPK Phonska Bersubsidi sebagai Dasar Perencanaan Safety Stock dan Reorder Point* — "
    "Studi Kasus: Kios Tani Gunung Salak, Kalapanunggal, Kabupaten Sukabumi."
)

tab_data, tab_eda, tab_ss, tab_train, tab_predict = st.tabs([
    "1️⃣ Data & Preprocessing",
    "2️⃣ Eksplorasi Data",
    "3️⃣ Safety Stock & ROP",
    "4️⃣ Training & Evaluasi LSTM",
    "5️⃣ Prediksi & Rekomendasi",
])

# ========================================================================
# TAB 1 — DATA & PREPROCESSING
# ========================================================================
with tab_data:
    st.header("Unggah & Bersihkan Data")
    st.markdown(
        "Unggah data transaksi harian stok dan penyaluran pupuk (format CSV/Excel). "
        "Data idealnya memuat kolom: **tanggal, musim tanam, penyaluran Urea, penyaluran NPK Phonska, "
        "sisa stok Urea, sisa stok NPK Phonska** — sesuai dataset akhir pada Bab IV skripsi."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded = st.file_uploader("Upload dataset (.csv / .xlsx)", type=["csv", "xlsx", "xls"])
    with col_b:
        st.write("")
        st.write("")
        use_demo = st.button("Gunakan data contoh (demo)", use_container_width=True)

    raw_df = None
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded)
            else:
                raw_df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    elif use_demo:
        raw_df = generate_demo_data()
        st.info("Menggunakan data contoh (sintetis) — ganti dengan data asli kapan saja.")

    if raw_df is not None:
        st.session_state["raw_df"] = raw_df

    raw_df = st.session_state.get("raw_df")

    if raw_df is not None:
        st.subheader("Pratinjau Data Mentah")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"Jumlah baris: {len(raw_df):,} | Jumlah kolom: {raw_df.shape[1]}")

        st.subheader("Pemetaan Kolom")
        st.markdown("Sesuaikan kolom pada data Anda dengan peran yang dibutuhkan pipeline.")
        guess = guess_column_mapping(raw_df.columns)
        cols = list(raw_df.columns)

        def _select(role, label):
            options = ["(pilih kolom)"] + cols
            default_idx = options.index(guess[role]) if guess.get(role) in cols else 0
            return st.selectbox(label, options, index=default_idx, key=f"map_{role}")

        m1, m2, m3 = st.columns(3)
        with m1:
            tanggal_col = _select("tanggal", "Kolom Tanggal")
            musim_col = _select("musim_tanam", "Kolom Musim Tanam")
        with m2:
            urea_sal_col = _select("penyaluran_urea", "Kolom Penyaluran Urea")
            npk_sal_col = _select("penyaluran_npk", "Kolom Penyaluran NPK Phonska")
        with m3:
            urea_stok_col = _select("stok_urea", "Kolom Sisa Stok Urea")
            npk_stok_col = _select("stok_npk", "Kolom Sisa Stok NPK Phonska")

        col_map = {
            "tanggal": tanggal_col, "musim_tanam": musim_col,
            "penyaluran_urea": urea_sal_col, "penyaluran_npk": npk_sal_col,
            "stok_urea": urea_stok_col, "stok_npk": npk_stok_col,
        }
        mapping_complete = all(v != "(pilih kolom)" and v is not None for v in col_map.values())

        if not mapping_complete:
            st.warning("Lengkapi seluruh pemetaan kolom di atas untuk melanjutkan.")
        else:
            if st.button("Jalankan Preprocessing", type="primary"):
                with st.spinner("Membersihkan data..."):
                    df_clean, info = preprocess_dataframe(raw_df, col_map)
                    df_feat = add_seasonal_features(df_clean)
                st.session_state["df_clean"] = df_clean
                st.session_state["prep_info"] = info
                st.session_state["df_feat"] = df_feat
                st.success("Preprocessing selesai.")

    if st.session_state["df_clean"] is not None:
        info = st.session_state["prep_info"]
        st.subheader("Hasil Preprocessing")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris awal", f"{info['n_baris_awal']:,}")
        c2.metric("Baris akhir (bersih)", f"{info['n_baris_akhir']:,}")
        c3.metric("Duplikat tanggal dihapus", info["tanggal_duplikat_dihapus"])
        c4.metric("Rentang tanggal", f"{info['tanggal_mulai'].date()} — {info['tanggal_akhir'].date()}")

        st.markdown(
            "**Langkah yang dijalankan:** (1) konversi tanggal & urutkan kronologis; "
            "(2) hapus duplikasi tanggal; (3) clipping nilai numerik ≥ 0; "
            "(4) forward-fill / backward-fill nilai kosong, sisanya diisi 0."
        )
        st.dataframe(st.session_state["df_feat"].head(10), use_container_width=True)

# ========================================================================
# TAB 2 — EDA
# ========================================================================
with tab_eda:
    st.header("Eksplorasi Data (EDA)")
    df_feat = st.session_state["df_feat"]
    if df_feat is None:
        st.info("Selesaikan tahap Preprocessing pada tab sebelumnya terlebih dahulu.")
    else:
        st.subheader("Statistik Deskriptif")
        desc_cols = {
            "stok_urea": "Sisa Stok Urea (kg)",
            "stok_npk": "Sisa Stok NPK Phonska (kg)",
            "penyaluran_urea": "Penyaluran Urea (kg)",
            "penyaluran_npk": "Penyaluran NPK Phonska (kg)",
        }
        desc = df_feat[list(desc_cols.keys())].agg(["mean", "std", "min", "max"]).T
        desc.columns = ["Rata-rata", "Std. Deviasi", "Minimum", "Maksimum"]
        desc.index = [desc_cols[i] for i in desc.index]
        st.dataframe(desc.style.format("{:,.2f}"), use_container_width=True)

        st.subheader("Distribusi Musim Tanam")
        musim_counts = df_feat["musim_tanam"].value_counts()
        musim_pct = (musim_counts / len(df_feat) * 100).round(2)
        musim_table = pd.DataFrame({"Jumlah Hari": musim_counts, "Persentase (%)": musim_pct})
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(musim_table, use_container_width=True)
        with c2:
            fig = px.pie(musim_table, names=musim_table.index, values="Jumlah Hari",
                         title="Proporsi Hari per Musim Tanam", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Visualisasi Tren Stok Harian")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_feat["tanggal"], y=df_feat["stok_urea"],
                                   name="Stok Urea", line=dict(color="#2E7D32")))
        fig2.add_trace(go.Scatter(x=df_feat["tanggal"], y=df_feat["stok_npk"],
                                   name="Stok NPK Phonska", line=dict(color="#EF6C00")))
        fig2.update_layout(title="Tren Stok Urea & NPK Phonska", xaxis_title="Tanggal", yaxis_title="Stok (kg)",
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Pola 'gigi gergaji' (sawtooth) yang berulang mencerminkan siklus pengisian ulang (restocking) "
            "diikuti penurunan bertahap akibat penyaluran ke petani."
        )

# ========================================================================
# TAB 3 — SAFETY STOCK & ROP
# ========================================================================
with tab_ss:
    st.header("Perhitungan Safety Stock & Reorder Point (Awal)")
    df_feat = st.session_state["df_feat"]
    if df_feat is None:
        st.info("Selesaikan tahap Preprocessing terlebih dahulu.")
    else:
        st.markdown("Formula: **Safety Stock = Z × σ × √(Lead Time)**, "
                     "**Stok Minimum (ROP) = (rata-rata penurunan harian × Lead Time) + Safety Stock**")
        c1, c2 = st.columns(2)
        with c1:
            lead_time = st.number_input("Lead Time pengadaan (hari)", min_value=1, max_value=60, value=4)
        with c2:
            service_level = st.slider("Service Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

        ss_urea = compute_safety_stock(df_feat, "stok_urea", lead_time, service_level)
        ss_npk = compute_safety_stock(df_feat, "stok_npk", lead_time, service_level)
        st.session_state["ss_urea"] = ss_urea
        st.session_state["ss_npk"] = ss_npk
        st.session_state["lead_time"] = lead_time
        st.session_state["service_level"] = service_level

        table = pd.DataFrame({
            "Jenis Pupuk": ["Urea", "NPK Phonska"],
            "Lead Time (hari)": [lead_time, lead_time],
            "Service Level": [f"{service_level*100:.0f}%"] * 2,
            "Z": [round(ss_urea["z"], 3), round(ss_npk["z"], 3)],
            "Rata Penurunan Harian (kg)": [round(ss_urea["rata_penurunan_harian"], 2), round(ss_npk["rata_penurunan_harian"], 2)],
            "Safety Stock (kg)": [round(ss_urea["safety_stock"], 2), round(ss_npk["safety_stock"], 2)],
            "Stok Minimum / ROP (kg)": [round(ss_urea["rop"], 2), round(ss_npk["rop"], 2)],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Nilai Safety Stock & ROP ini akan digunakan sebagai ambang pembanding terhadap hasil "
            "prediksi stok periode berikutnya pada tab 'Prediksi & Rekomendasi'."
        )

# ========================================================================
# TAB 4 — TRAINING & EVALUASI LSTM
# ========================================================================
with tab_train:
    st.header("Training & Evaluasi Model LSTM")
    df_feat = st.session_state["df_feat"]
    if df_feat is None:
        st.info("Selesaikan tahap Preprocessing terlebih dahulu.")
    else:
        st.markdown(
            "Arsitektur baku (mengikuti Sub-bab 4.5.1): **Stacked LSTM** — LSTM(64, tanh) → Dropout(0.2) → "
            "LSTM(32, tanh) → Dropout(0.2) → Dense(16, ReLU) → Dense(1). Optimizer Adam (lr=0.001), loss MAE."
        )

        with st.expander(" Pengaturan Lanjutan (opsional)"):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                max_epoch = st.number_input("Maks. Epoch", 10, 1000, 300, step=10)
                batch_size = st.number_input("Batch Size", 4, 256, 16, step=4)
            with adv2:
                es_patience = st.number_input("EarlyStopping patience", 5, 200, 35, step=5)
                rlr_patience = st.number_input("ReduceLROnPlateau patience", 3, 100, 12, step=1)
            with adv3:
                test_ratio = st.slider("Rasio data uji", 0.10, 0.40, 0.20, step=0.05)
                val_ratio = st.slider("Rasio validasi internal (dari data latih)", 0.05, 0.30, 0.10, step=0.05)

        target_choice = st.multiselect(
            "Jenis pupuk yang dilatih",
            ["Urea", "NPK Phonska"], default=["Urea", "NPK Phonska"]
        )
        timesteps_choice = st.multiselect(
            "Skenario timesteps (hari) yang diuji",
            [7, 14, 21, 30], default=[7, 14, 21, 30]
        )

        run = st.button("Latih Model", type="primary", disabled=(not target_choice or not timesteps_choice))

        target_map = {"Urea": "stok_urea", "NPK Phonska": "stok_npk"}

        if run:
            feature_cols_base = get_feature_columns(df_feat)
            all_results = {}
            trained_models = {}
            scalers = {}
            progress = st.progress(0.0)
            total_runs = len(target_choice) * len(timesteps_choice)
            done = 0
            status_box = st.empty()

            for jenis in target_choice:
                target_col = target_map[jenis]
                exog_col = "penyaluran_urea" if jenis == "Urea" else "penyaluran_npk"
                feature_cols = [exog_col] + feature_cols_base

                data = df_feat[[target_col] + feature_cols].astype(float).values
                n = len(data)
                n_test = int(n * test_ratio)
                n_train = n - n_test

                train_raw = data[:n_train]
                test_raw = data[n_train - max(timesteps_choice):]  # sisakan buffer window untuk sequence uji

                target_scaler = MinMaxScaler()
                feat_scaler = MinMaxScaler()
                target_scaler.fit(train_raw[:, [0]])
                if len(feature_cols) > 0:
                    feat_scaler.fit(train_raw[:, 1:])

                scalers[jenis] = {"target": target_scaler, "features": feat_scaler, "feature_cols": feature_cols}

                for ts in timesteps_choice:
                    status_box.info(f"Melatih model **{jenis}** — timesteps {ts} hari...")

                    train_scaled_t = target_scaler.transform(train_raw[:, [0]]).flatten()
                    train_scaled_f = feat_scaler.transform(train_raw[:, 1:]) if len(feature_cols) > 0 else None
                    X_train, y_train = build_sequences(train_scaled_t, train_scaled_f, ts)

                    n_val = max(1, int(len(X_train) * val_ratio))
                    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
                    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

                    test_scaled_t = target_scaler.transform(test_raw[:, [0]]).flatten()
                    test_scaled_f = feat_scaler.transform(test_raw[:, 1:]) if len(feature_cols) > 0 else None
                    X_test, y_test = build_sequences(test_scaled_t, test_scaled_f, ts)
                    # buang overlap buffer supaya jumlah sample uji == n_test
                    X_test, y_test = X_test[-n_test:], y_test[-n_test:]

                    n_features = X_train.shape[2]
                    model = build_lstm_model(ts, n_features)
                    history = model.fit(
                        X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=max_epoch, batch_size=batch_size, verbose=0,
                        callbacks=get_callbacks(es_patience, rlr_patience),
                    )
                    n_epoch_run = len(history.history["loss"])

                    pred_scaled = model.predict(X_test, verbose=0).flatten()
                    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                    metrics = evaluate_predictions(actual, pred)
                    dates_test = df_feat["tanggal"].values[-n_test:]

                    all_results[(jenis, ts)] = {
                        "metrics": metrics, "epoch": n_epoch_run,
                        "actual": actual, "pred": pred, "dates": dates_test,
                        "history": history.history,
                    }
                    trained_models[(jenis, ts)] = model

                    done += 1
                    progress.progress(done / total_runs)

            status_box.success("Selesai melatih seluruh skenario.")
            st.session_state["results"] = all_results
            st.session_state["trained_models"] = trained_models
            st.session_state["scalers"] = scalers
            st.session_state["timesteps_choice"] = timesteps_choice
            st.session_state["target_choice"] = target_choice

        if st.session_state["results"]:
            st.subheader("Tabel Hasil Evaluasi — Seluruh Skenario")
            rows = []
            for (jenis, ts), r in st.session_state["results"].items():
                m = r["metrics"]
                rows.append({
                    "Jenis Stok": jenis, "Timesteps": ts, "Epoch": r["epoch"],
                    "MAE (kg)": round(m["MAE"], 2), "RMSE (kg)": round(m["RMSE"], 2),
                    "MAPE (%)": round(m["MAPE"], 2), "wMAPE (%)": round(m["wMAPE"], 2),
                    "SMAPE (%)": round(m["SMAPE"], 2), "R2": round(m["R2"], 3),
                    "Kategori": m["Kategori"],
                })
            eval_df = pd.DataFrame(rows).sort_values(["Jenis Stok", "Timesteps"])
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

            fig = px.bar(eval_df, x="Timesteps", y="wMAPE (%)", color="Jenis Stok", barmode="group",
                         title="Perbandingan wMAPE per Skenario Timesteps")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Model Terbaik (kriteria wMAPE terendah)")
            best_rows = []
            best_keys = {}
            for jenis in eval_df["Jenis Stok"].unique():
                sub = eval_df[eval_df["Jenis Stok"] == jenis]
                best = sub.loc[sub["wMAPE (%)"].idxmin()]
                best_rows.append(best)
                best_keys[jenis] = int(best["Timesteps"])
            best_df = pd.DataFrame(best_rows)
            st.dataframe(best_df, use_container_width=True, hide_index=True)
            st.session_state["best_keys"] = best_keys

            st.subheader("Grafik Prediksi vs Aktual — Model Terbaik")
            for jenis, ts in best_keys.items():
                r = st.session_state["results"][(jenis, ts)]
                cA, cB = st.columns(2)
                with cA:
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(y=r["history"]["loss"], name="Train Loss"))
                    loss_fig.add_trace(go.Scatter(y=r["history"]["val_loss"], name="Validation Loss"))
                    loss_fig.update_layout(title=f"Loss Curve — {jenis} (timesteps={ts})",
                                            xaxis_title="Epoch", yaxis_title="MAE (scaled)")
                    st.plotly_chart(loss_fig, use_container_width=True)
                with cB:
                    pred_fig = go.Figure()
                    pred_fig.add_trace(go.Scatter(x=r["dates"], y=r["actual"], name="Aktual"))
                    pred_fig.add_trace(go.Scatter(x=r["dates"], y=r["pred"], name="Prediksi"))
                    pred_fig.update_layout(title=f"Aktual vs Prediksi — {jenis} (timesteps={ts})",
                                            xaxis_title="Tanggal", yaxis_title="Stok (kg)")
                    st.plotly_chart(pred_fig, use_container_width=True)

# ========================================================================
# TAB 5 — PREDIKSI & REKOMENDASI
# ========================================================================
with tab_predict:
    st.header("Prediksi Periode Berikutnya & Rekomendasi Stok")
    df_feat = st.session_state["df_feat"]

    if df_feat is None or not st.session_state["results"] or "best_keys" not in st.session_state:
        st.info("Selesaikan tahap Training & Evaluasi terlebih dahulu (pilih model terbaik akan otomatis dari wMAPE terendah).")
    elif st.session_state["ss_urea"] is None:
        st.info("Hitung Safety Stock & ROP pada tab sebelumnya terlebih dahulu.")
    else:
        best_keys = st.session_state["best_keys"]
        feature_cols_base = get_feature_columns(df_feat)
        target_map = {"Urea": "stok_urea", "NPK Phonska": "stok_npk"}
        ss_map = {"Urea": st.session_state["ss_urea"], "NPK Phonska": st.session_state["ss_npk"]}

        rows = []
        for jenis, ts in best_keys.items():
            model = st.session_state["trained_models"][(jenis, ts)]
            sc = st.session_state["scalers"][jenis]
            target_col = target_map[jenis]
            exog_col = "penyaluran_urea" if jenis == "Urea" else "penyaluran_npk"
            feature_cols = [exog_col] + feature_cols_base

            last_window = df_feat[[target_col] + feature_cols].astype(float).values[-ts:]
            target_scaled = sc["target"].transform(last_window[:, [0]]).flatten()
            feat_scaled = sc["features"].transform(last_window[:, 1:]) if len(feature_cols) > 0 else None

            if feat_scaled is not None:
                window = np.concatenate([target_scaled.reshape(-1, 1), feat_scaled], axis=1)
            else:
                window = target_scaled.reshape(-1, 1)
            X_next = window.reshape(1, ts, window.shape[1])

            pred_scaled = model.predict(X_next, verbose=0).flatten()[0]
            pred_value = sc["target"].inverse_transform([[pred_scaled]])[0, 0]

            ss = ss_map[jenis]
            status = stock_status(pred_value, ss["rop"])

            rows.append({
                "Jenis Pupuk": jenis,
                "Tanggal Data Terakhir": pd.Timestamp(df_feat["tanggal"].max()).date(),
                "Prediksi Stok Periode Berikutnya (kg)": round(pred_value, 2),
                "Safety Stock (kg)": round(ss["safety_stock"], 2),
                "Stok Minimum / ROP (kg)": round(ss["rop"], 2),
                "Status": status,
            })

        rec_df = pd.DataFrame(rows)

        def _highlight(row):
            color = "#ffe5e5" if row["Status"] == "Waspada" else "#e6f4ea"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(rec_df.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)

        st.subheader("Interpretasi")
        for _, row in rec_df.iterrows():
            jenis = row["Jenis Pupuk"]
            pred_value = row["Prediksi Stok Periode Berikutnya (kg)"]
            rop = row["Stok Minimum / ROP (kg)"]
            batas_waspada = rop * 1.2
            selisih = pred_value - rop
            if row["Status"] == "Waspada":
                st.warning(
                    f"**{jenis}** — status **Waspada**. Prediksi stok ({pred_value:,.2f} kg) berada "
                    f"{'di bawah' if selisih < 0 else 'sedikit di atas'} Stok Minimum ({rop:,.2f} kg) dan "
                    f"masih di bawah ambang batas Waspada ({batas_waspada:,.2f} kg). "
                    f"Disarankan segera mempersiapkan proses reorder ke distributor."
                )
            else:
                st.success(
                    f"**{jenis}** — status **Aman**. Prediksi stok ({pred_value:,.2f} kg) berada di atas "
                    f"ambang batas Waspada ({batas_waspada:,.2f} kg). Belum diperlukan tindakan pengadaan segera, "
                    f"namun pemantauan berkala tetap disarankan."
                )

        st.caption(
            "Status ditentukan: **Waspada** apabila prediksi stok < 120% dari Stok Minimum (ROP); "
            "**Aman** apabila prediksi stok ≥ 120% dari Stok Minimum (ROP)."
        )

st.divider()
st.caption(
    "Dibangun berdasarkan metodologi skripsi LSTM prediksi stok pupuk bersubsidi — "
    "Kios Tani Gunung Salak, Kalapanunggal, Kabupaten Sukabumi."
)
