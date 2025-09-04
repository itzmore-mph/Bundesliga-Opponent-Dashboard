# Baseline xG Model – La Liga 2017/18

## 🎯 Goal
Build a baseline Expected Goals (xG) model using **StatsBomb Open Data**.
Estimate scoring probability of shots via Logistic Regression and compare with more complex models.

## 📂 Structure
```
baseline-xg-model/
├─ data/
│  ├─ raw/                # Original StatsBomb JSON (events)
│  ├─ interim/            # Cleaned/filtered shots
│  └─ processed/          # Final features for modeling
├─ notebooks/
│  └─ 01_xg_model.ipynb   # Main notebook (EDA, model, eval, plots)
├─ src/
│  ├─ data_loading.py     # Functions to load & filter shots
│  ├─ features.py         # Feature engineering (distance, angle, body part)
│  ├─ model.py            # Logistic regression & evaluation
│  └─ viz.py              # Plotting functions (shot maps, ROC)
├─ reports/
│  ├─ figures/            # Plots & shot maps
│  └─ xg_report.md        # Short summary (results, insights, next steps)
├─ environment.yml
├─ .gitignore
└─ README.md
```

## ⚙️ Method
- **Features:** distance to goal, angle, body part
- **Model:** Logistic Regression (scikit-learn)
- **Evaluation:** AUC, Brier Score
- **Visualization:** Shot maps with predicted xG overlay

## 📊 Results (example)
- Logistic Regression: AUC 0.71, Brier 0.20
- Random Forest: AUC 0.75, Brier 0.19
- XGBoost: AUC 0.77, Brier 0.18

## 🚀 Next Steps
- Add contextual features (assist type, shot pressure, match state)
- Apply to full season and generate xG-based player/team tables
- Deploy simple dashboard (Streamlit/Power BI)

## 📝 Data
StatsBomb Open Data (CC BY-NC-SA 4.0)  
https://github.com/statsbomb/open-data
