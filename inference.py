import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

import joblib

# Load config
CONFIG_PATH = Path(__file__).parent / "config.pkl"
with open(CONFIG_PATH, "rb") as f:
    CONFIG = pickle.load(f)

OPTIMAL_THRESHOLDS = CONFIG["optimal_thresholds"]
MODELS_INFO = CONFIG["final_models_info"]

def load_model_by_semester(semester):
    model_path = Path(__file__).parent / f"models/model_scenario_{semester}.pkl"
    return joblib.load(model_path)

def calculate_single_student_features(semester, history_data):
    # history_data: list of dicts [{'ips': 3.5, 'sks': 20, 'sks_lulus': 20}, ...]
    row = {}
    
    # 1. Base IPS Series
    ips_values = []
    for i in range(1, semester):
        if i <= len(history_data):
            val = history_data[i-1]['ips']
            row[f'ips_{i}'] = val
            ips_values.append(val)
        else:
            row[f'ips_{i}'] = 0.0
            
    if not ips_values:
        row['mean_ips'] = 0
        row['total_sks_lulus'] = 0
        return pd.DataFrame([row])

    # 2. Cumulative Features
    row['mean_ips'] = np.mean(ips_values)
    
    if len(ips_values) >= 2:
        slope, _, _, _, _ = stats.linregress(range(len(ips_values)), ips_values)
        row['ips_trend'] = slope
    else:
        row['ips_trend'] = 0.0
        
    row['ips_std'] = np.std(ips_values, ddof=1) if len(ips_values) >= 2 else 0.0
    row['min_ips'] = np.min(ips_values)
    
    drops = [ips_values[i] - ips_values[i+1] for i in range(len(ips_values)-1)]
    pos_drops = [d for d in drops if d > 0]
    row['max_ips_drop'] = max(pos_drops) if pos_drops else 0.0
    
    # SKS Features
    total_lulus = sum(d.get('sks_lulus', 0) for d in history_data[:semester-1])
    total_diambil = sum(d.get('sks', 0) for d in history_data[:semester-1])
    
    target_sks = 18 * (semester - 1)
    row['total_sks_lulus'] = total_lulus
    row['sks_gap'] = target_sks - total_lulus
    row['on_track'] = 1 if total_lulus >= target_sks else 0
    row['pass_rate'] = (total_lulus / total_diambil * 100) if total_diambil > 0 else 0.0
    row['avg_sks_lulus'] = total_lulus / (semester - 1)
    
    row['gender_encoded'] = 1
    
    return pd.DataFrame([row])

def validate_input(semester, history):
    if semester < 2 or semester > 8:
        return False, "Semester must be 2-8"
        
    if len(history) != semester - 1:
        return False, f"Need history for {semester-1} semesters"
        
    return True, None

def classify_risk(prob, threshold):
    if prob >= 0.70: return "HIGH"
    elif prob >= threshold: return "MEDIUM"
    return "LOW"

def predict_graduation(semester, history):
    is_valid, msg = validate_input(semester, history)
    if not is_valid: return {'error': msg}
    
    try:
        model = load_model_by_semester(semester)
        X = calculate_single_student_features(semester, history)
        
        # Determine probability
        if 'XGB' in str(type(model)):
             proba = model.predict_proba(X)[0]
             prob = proba[1]
        else:
             proba = model.predict_proba(X)[0]
             # safe way
             if hasattr(model, 'classes_'):
                 classes = list(model.classes_)
                 idx = 1
                 if 'Tidak Lulus Tepat Waktu' in classes:
                     idx = classes.index('Tidak Lulus Tepat Waktu')
                 prob = proba[idx]
             else:
                 prob = proba[1]
        
        threshold = OPTIMAL_THRESHOLDS.get(semester, 0.5)
        pred_label = "Tidak Lulus Tepat Waktu" if prob >= threshold else "Lulus Tepat Waktu"
        risk = classify_risk(prob, threshold)
        
        return {
            'prediction': pred_label,
            'probability': prob,
            'risk_level': risk,
            'threshold': threshold
        }
    except Exception as e:
        return {'error': str(e)}
