# Flight Delay Predictor

Binary classification — predict whether a flight arrives **≥15 minutes late** using only information available **24 hours before departure**.

## Architecture

```
┌─────────────────┐   POST /predict   ┌──────────────────┐
│   React UI      │ ────────────────► │  FastAPI         │
│   (Port 3000)   │ ◄──────────────── │  (Port 8000)     │
└─────────────────┘   JSON response   └────────┬─────────┘
                                               │
                                      ┌────────▼─────────┐
                                      │  LightGBM Model  │
                                      │  + SHAP Values   │
                                      └──────────────────┘
```

## Features

| Feature Group | Features | Method |
|---|---|---|
| Temporal | Dep. time, day of week, month | Cyclic sin/cos encoding |
| Categorical | Airline, origin, dest, tail# | K-Fold target encoding (λ=20) |
| Geospatial | Route distance | Haversine great-circle formula |
| Historical | Route delay rate, airline delay rate | Smoothed rolling aggregates |
| Calendar | Holiday proximity | Binary flag (±2 days of US holidays) |

**T-24h Rule (strict):** every feature is knowable 24 hours before departure. No post-departure information is used.

## Quick Start

### Option 1: Docker (recommended)

```bash
# Requires: Docker + Docker Compose + trained model artifacts in backend/artifacts/
docker-compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Option 2: Local development

**Backend:**
```bash
pip install -r requirements.txt
cd backend
ARTIFACTS_DIR=./artifacts uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

## Training Pipeline

### 1. Get data
Download BTS On-Time Performance data from:
https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ

Required fields: `FL_DATE`, `OP_CARRIER`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`, `ArrDelay`, `DepDelay`, `TAIL_NUM`, `DISTANCE`, `Year`, `Month`, `DayofMonth`, `DayOfWeek`

### 2. Train model

```bash
# Basic training
python -m src.model.train data/flights_2023.csv backend/artifacts/

# With Optuna hyperparameter tuning (50 trials)
python -m src.model.optimize data/flights_2023.csv 50 backend/artifacts/
```

### 3. Evaluate

```python
from src.model.evaluate import evaluate_model
metrics = evaluate_model(model, X_test, y_test, output_dir=Path("reports/"))
print(f"PR-AUC: {metrics['pr_auc']:.4f}")
```

## Project Structure

```
flight-delay-predictor/
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py               # BTS data loading + blacklist enforcement
│   │   └── splitter.py             # Temporal train/val/test split
│   ├── features/
│   │   ├── cyclic.py               # Sin/cos time encoding
│   │   ├── target_encoding.py      # K-Fold smoothed target encoding
│   │   ├── geospatial.py           # Haversine distance
│   │   └── pipeline.py             # Full feature pipeline
│   ├── model/
│   │   ├── train.py                # LightGBM training
│   │   ├── evaluate.py             # PR-AUC, SHAP, calibration plots
│   │   └── optimize.py             # Optuna hyperparameter tuning
│   └── utils/
│       └── leakage_check.py        # Feature audit
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 # FastAPI routes
│   │   ├── model.py                # Model loading + inference
│   │   ├── features.py             # Online feature engineering
│   │   └── schemas.py              # Pydantic request/response models
│   └── artifacts/                  # Trained model + encoders (gitignored)
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── FlightForm.jsx
│       │   ├── PredictionResult.jsx
│       │   ├── GaugeMeter.jsx
│       │   ├── FactorsChart.jsx
│       │   ├── LoadingSkeleton.jsx
│       │   └── ErrorToast.jsx
│       └── utils/api.js
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## API Reference

### POST /predict

```json
// Request
{
  "airline": "AA",
  "origin": "JFK",
  "destination": "LAX",
  "scheduled_departure": "2026-07-12T14:30:00",
  "tail_number": "N123AA"
}

// Response
{
  "delay_probability": 0.67,
  "prediction": 1,
  "risk_level": "High",
  "top_factors": [
    {"feature": "airline_delay_rate", "impact": 0.15},
    {"feature": "departure_hour",     "impact": 0.12},
    {"feature": "route_distance_km",  "impact": 0.08}
  ],
  "threshold_used": 0.5
}
```

**Risk levels:** Low (<0.3) · Medium (0.3–0.6) · High (>0.6)

### GET /health

Returns model status and version.

## Design Decisions

| Decision | Rationale |
|---|---|
| LightGBM over deep learning | Handles mixed types natively, trains fast, interpretable via SHAP |
| PR-AUC as primary metric | Flight delays are ~20% base rate — PR-AUC more informative than ROC-AUC on imbalanced data |
| Temporal split (no shuffle) | Simulates real deployment — model only ever sees future flights at inference |
| `scale_pos_weight` for imbalance | Native LightGBM parameter; no SMOTE needed |
| K-Fold target encoding | Prevents target leakage from naive mean encoding of categoricals |
| Threshold selection | Plot PR curve and pick threshold per operational preference (catch more delays vs. fewer false alarms) |

## Expected Performance

| Metric | Target |
|---|---|
| PR-AUC | >0.35 (realistic without real-time weather) |
| Latency | <200ms per prediction |
| Data leakage | Zero — verified by blacklist check + SHAP audit |

## License

MIT
