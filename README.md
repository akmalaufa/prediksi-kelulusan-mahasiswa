# Student Graduation Prediction System

## Deployment Package

### Files
- `models/` - 7 final models (scenario 2-8)
- `config.pkl` - Thresholds and metadata
- `inference.py` - Core inference functions
- `app.py` - Streamlit application

### Quick Start

1. Install dependencies:
```bash
pip install pandas scikit-learn xgboost streamlit
```

2. Run Streamlit app:
```bash
streamlit run app.py
```

### API Usage

```python
from inference import predict_graduation

result = predict_graduation(semester=3, ips_list=[3.5, 3.6])
print(result["prediction"])
```

### Model Information

- **Scenarios**: 7 models (semester 2-8)
- **Priority**: Recall (detect at-risk students)
- **Risk Levels**: HIGH (≥0.70), MEDIUM (0.50-0.69), LOW (<0.50)

### Important Notes

⚠️ Model adalah alat prediksi, bukan penentu keputusan final

✅ Sistem tidak menyimpan data mahasiswa

🔄 Retrain model setiap tahun dengan data terbaru
