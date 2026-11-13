# lung-cancer-dss
This is my thesis, a year-long project of research and coding. This is the project I am most proud of
# Lung Cancer Risk Clinical Decision Support System (CDSS)

An end-to-end Machine Learning Clinical Decision Support System (CDSS) designed for primary care triage and early risk stratification of lung cancer using non-invasive epidemiological, lifestyle, environmental, and clinical parameters.

> **BSc Thesis Project**  
> **Title:** Ανάπτυξη μοντέλου διάγνωσης για καρκίνο του πνεύμονα, με τεχνικές τεχνητής νοημοσύνης (*Development of a diagnostic model for lung cancer using Artificial Intelligence techniques*)  
> **Author:** Aikaterini Analyti (inf2022010)  
> **Supervisor:** Assoc. Prof. Themistoklis Exarchos  
> **Committee:** Assoc. Prof. Themistoklis Exarchos, Prof. Panagiotis Vlamos, Asst. Prof. Marios Krokidis  
> **Institution:** Ionian University, Department of Informatics

---

## Overview

More than 50% of lung cancer cases are diagnosed at an advanced or metastatic stage, where the 5-year survival rate drops below 10% (compared to >63% when detected at a localized stage). This project provides a **Clinical Decision Support System (CDSS)** tailored for primary care physicians to rapidly evaluate patients during initial consultations and identify high-risk individuals who should be referred for confirmatory imaging such as Low-Dose Computed Tomography (LDCT).

### Key Methodological Highlights
* **Elimination of Data Leakage via Explainable AI (SHAP):** Initial exploratory modeling on all 29 features revealed target leakage from post-diagnostic/paraclinical variables (`xray_abnormal`, `crp_level`, `copd`, `fev1_x10`) using **TreeSHAP** analysis. Removing these 4 proxy features ensured compliance with **TRIPOD** guidelines for primary care prediction models, relying strictly on 25 first-line non-invasive indicators.
* **Three-Tier Microservices Architecture:** Strict separation of concerns across the Presentation Tier (**Streamlit**), Business Logic & Data Validation Firewall Tier (**FastAPI** + **Pydantic**), and ML Serving Tier (**Scikit-Learn Pipeline** + **XGBoost** serialized via **Joblib**).
* **Zero-Configuration Reproducibility:** Fully containerized using **Docker** and orchestrated via **Docker Compose** for seamless deployment in clinical environments.

---

## Model Performance (10-Fold Stratified Cross-Validation)

Models were evaluated across 5,000 patient profiles using 10-fold Stratified Cross-Validation. Below is the comparison before and after removing the 4 data-leakage features:

| Dataset State | Algorithm | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **With Data Leakage** *(29 features)* | Decision Tree | 94.64% | 89.61% | 88.75% | 89.18% |
| | Random Forest | 96.40% | 93.27% | 92.44% | 92.85% |
| | **XGBoost** | **97.68%** | **95.85%** | **94.77%** | **95.31%** |
| **No Data Leakage** *(25 features - Final)* | Decision Tree | 93.70% | 87.37% | 87.30% | 87.34% |
| | Random Forest | 95.38% | 91.15% | 90.19% | 90.66% |
| | **XGBoost (Selected)** | **95.64%** | **91.17%** | **91.32%** | **91.24%** |

* **Sensitivity (Recall) - High Risk Class:** `91.32%` (False Negative Rate limited to `8.68%`)
* **Specificity - Low Risk Class:** `97.07%` (False Positive Rate limited to `2.93%`)

---

## System Architecture & Features

### Input Parameters (25 First-Line Features)
The Pydantic validation schema strictly enforces the 25 non-invasive features, grouped into four clinical categories in the UI:
1. **Demographics & Vitals:** `age`, `gender`, `bmi`, `oxygen_saturation`, `education_years`, `income_level`
2. **Smoking History:** `smoker`, `smoking_years`, `cigarettes_per_day`, `pack_years`, `passive_smoking`
3. **Environmental & Medical History:** `air_pollution_index`, `occupational_exposure`, `radon_exposure`, `family_history_cancer`, `asthma`, `previous_tb`
4. **Symptoms & Lifestyle:** `chronic_cough`, `chest_pain`, `shortness_of_breath`, `fatigue`, `exercise_hours_per_week`, `diet_quality`, `alcohol_units_per_week`, `healthcare_access`

### Repository Structure
```text
├── api.py                      # FastAPI backend server & Pydantic validation firewall
├── app.py                      # Streamlit clinical user interface (Frontend)
├── train_pipeline.py           # Preprocessing (ColumnTransformer) & XGBoost training script
├── xgboost_pipeline.joblib     # Serialized Scikit-Learn & XGBoost serving artifact
├── Dockerfile.api              # Docker container specification for the FastAPI backend
├── Dockerfile.app              # Docker container specification for the Streamlit frontend
├── docker-compose.yml          # Multi-container orchestration & bridge networking
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
