import streamlit as st
import pandas as pd
from inference import predict_graduation

st.set_page_config(page_title="Prediksi Kelulusan", page_icon="🎓")

st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa Informatika")
st.markdown("Early Warning System")

# Sidebar
st.sidebar.header("Input Data")
semester = st.sidebar.slider("Semester Mahasiswa Saat Ini", 2, 8, 3)

st.sidebar.subheader("Riwayat Akademik")
history_data = []

# Dynamic Inputs
for i in range(1, semester):
    with st.sidebar.expander(f"Semester {i}", expanded=True):
        col1, col2 = st.columns(2)
        ips = col1.number_input(f"IPS Sem {i}", 0.0, 4.0, 3.0, step=0.01, key=f"ips_{i}")
        sks = col2.number_input(f"SKS Diambil {i}", 0, 24, 20, key=f"sks_{i}")
        sks_lulus = st.number_input(f"SKS Lulus {i}", 0, sks, min(sks, 20), key=f"sks_lulus_{i}")
        
        history_data.append({
            'ips': ips,
            'sks': sks,
            'sks_lulus': sks_lulus
        })

if st.sidebar.button("Prediksi Status", type="primary"):
    with st.spinner("Menganalisis performa akademik..."):
        result = predict_graduation(semester, history_data)
        
    if 'error' in result:
        st.error(result['error'])
    else:
        st.divider()
        st.header("Hasil Analisis")
        
        # Row 1: Main Prediction (Full Width)
        st.metric("Prediksi", result['prediction'])
        
        # Row 2: Details
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Probabilitas Risiko", f"{result['probability']:.1%}")
            
        with col2:
            risk_map = {"HIGH": "🔴 TINGGI", "MEDIUM": "🟡 SEDANG", "LOW": "🟢 RENDAH"}
            st.metric("Level Risiko", risk_map.get(result['risk_level'], result['risk_level']))
            
        # Analysis Details
        st.divider()
        st.subheader("Detail Faktor Risiko")
        
        # Calculate features for display
        total_sks_lulus = sum(d['sks_lulus'] for d in history_data)
        target_sks = (semester - 1) * 18
        gap = target_sks - total_sks_lulus
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total SKS Lulus", total_sks_lulus)
        c2.metric("Target Minimum", target_sks)
        c3.metric("Gap SKS", abs(gap), delta_color="inverse" if gap > 0 else "normal", delta=-gap)
        
        if result['risk_level'] == 'HIGH':
            st.error("⚠️ MAHASISWA BERISIKO TINGGI. Disarankan segera menemui dosen wali.")
        elif result['risk_level'] == 'MEDIUM':
            st.warning("⚡ PERLU PERHATIAN. Tingkatkan performa dan pastikan SKS lulus tercapai.")
        else:
            st.success("✅ ON TRACK. Pertahankan prestasi akademik.")

st.sidebar.info("#WEARETHEONE")
