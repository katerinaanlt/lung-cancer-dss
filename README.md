# 🫁 Lung Cancer Risk Clinical Decision Support System (CDSS)

> **BSc Thesis Project**  
> **Title:** Development of a Lung Cancer Diagnosis Model Using Artificial Intelligence Techniques  
> **Author:** Aikaterini Analyti (inf2022010)  
> **Supervisor:** Assoc. Prof. Themistoklis Exarchos  
> **Institution:** Department of Informatics, Ionian University  

---

## 📖 Introduction

Lung cancer remains the leading cause of cancer-related mortality worldwide, accounting for approximately 2.5 million new cases and 1.8 million deaths annually (GLOBOCAN). The primary clinical challenge driving this high mortality rate is not solely the biological aggressiveness of the malignancy, but the timing of its detection. Because the lungs lack internal pain receptors and early-stage symptoms (such as a mild cough or fatigue) are largely non-specific and easily misattributed to common respiratory infections or smoking habits, over 50% of patients are diagnosed at an advanced or metastatic stage. Statistical data from the SEER program highlights a stark survival gap: while the 5-year survival rate for Non-Small Cell Lung Cancer (NSCLC) exceeds 63% when detected at a localized stage, it plummets to below 10% once distant metastasis has occurred.

In primary care settings, physicians are tasked with synthesizing a complex array of heterogeneous patient data—ranging from quantified smoking exposure (pack-years) and environmental hazards (radon, air pollution) to subtle clinical symptoms—under significant time constraints. This project addresses this diagnostic bottleneck by developing a machine learning-driven **Clinical Decision Support System (CDSS)** designed for primary care triage. Rather than replacing clinical judgment, this tool acts as an objective, non-invasive screening mechanism that evaluates 25 first-line patient parameters in real time. Its goal is to stratify patient risk early, ensuring high-risk individuals are promptly referred for confirmatory Low-Dose CT (LDCT) imaging while sparing low-risk patients from unnecessary radiation and invasive procedures.

---

## 🔬 Methodology

The development of the system followed a structured, four-stage pipeline adhering to international biomedical prediction standards (TRIPOD statement):

1. **Exploratory Data Analysis (EDA) & Preprocessing:**  
   Utilized a dataset of $5,000$ patient records (`lung_cancer.csv`) originally comprising 30 demographic, lifestyle, environmental, and clinical variables. Categorical binary and ordinal features were systematically mapped using `OrdinalEncoder`, while continuous numerical features were preserved in their natural scale via a unified `Scikit-learn` `ColumnTransformer` pipeline.
2. **Explainable AI (XAI) & Feature Pruning:**  
   Baseline models were initially trained on all 29 input features and audited using **TreeSHAP** (SHapley Additive exPlanations) to quantify each variable's marginal contribution to the prediction and screen for post-diagnostic bias.
3. **Model Training & Stratified Validation:**  
   Evaluated three tree-based architectures of increasing complexity: **Decision Trees**, **Random Forest**, and **Extreme Gradient Boosting (XGBoost)**. Models were assessed using Stratified 10-Fold Cross-Validation, prioritizing **Recall (Sensitivity)** and **F1-Score** to minimize clinically dangerous False Negatives.
4. **Three-Tier System Architecture & Containerization:**  
   The optimal XGBoost pipeline was serialized via `Joblib` and deployed within a decoupled, three-tier microservices architecture:
   * **Presentation Tier:** An ergonomic, 4-column clinical web interface built with **Streamlit**.
   * **Application/Logic Tier:** An asynchronous RESTful API powered by **FastAPI** and **Pydantic**.
   * **Deployment:** Containerized using **Docker** and orchestrated via **Docker Compose** for zero-configuration clinical deployment.

### 📊 Model Performance (10-Fold Cross-Validation)

| Dataset State | Algorithm | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **With Data Leakage** *(29 features)* | XGBoost | 97.68% | 95.85% | 94.77% | — | 95.31% |
| **Clean Triage Pipeline** *(25 features)* | Decision Tree | 93.70% | 87.37% | 87.30% | — | 87.34% |
| **Clean Triage Pipeline** *(25 features)* | Random Forest | 95.38% | 91.15% | 90.19% | — | 90.66% |
| **Clean Triage Pipeline** *(25 features)* | **XGBoost (Final)** | **95.64%** | **91.17%** | **91.32%** | **97.07%** | **91.24%** |

---

## 🛠️ How Key Challenges Were Tackled

* **Eliminating Target Data Leakage via SHAP:**  
  Initial baseline experiments yielded an artificially inflated F1-Score of $95.31\%$. Through **SHAP** analysis, it was proven that the model was heavily relying on four post-diagnostic or specialized clinical proxies: abnormal chest X-rays (`xray_abnormal`), serum C-reactive protein levels (`crp_level`), diagnosed COPD (`copd`), and spirometry forced expiratory volume (`fev1_x10`). Because these metrics are outcomes of secondary diagnostic workups rather than primary triage inputs, they were pruned. Retraining on the 25 strictly non-invasive features yielded a realistic, generalizable F1-Score of $91.24\%$. To prevent leakage in production, a **Pydantic schema firewall** was implemented in the FastAPI backend to strictly validate the 25 allowed features and reject any unauthorized parameters with an `HTTP 422` error.
* **Handling Class Imbalance Without Synthetic Noise:**  
  The dataset exhibited a natural $1:3$ class ratio ($24.88\%$ High Risk vs. $75.12\%$ Low Risk). Rather than applying undersampling (which discards valuable clinical records) or synthetic oversampling like SMOTE (which can introduce artificial noise artifacts into sensitive biomedical distributions), the natural epidemiological distribution was preserved. The imbalance was effectively managed using XGBoost's sequential gradient optimization and L2 regularization, achieving a Macro Average F1-Score of $0.94$.
* **Overcoming Clinical Distrust ("Black-Box" Effect) & UI Fatigue:**  
  Literature highlights that physicians often resist CDSS tools due to algorithmic opacity and cumbersome interfaces. Using TreeSHAP confirmed that the model's top decision drivers (`pack_years`, `age`, `chronic_cough`, `air_pollution_index`) align directly with established medical literature. Furthermore, the Streamlit UI groups inputs into four logical clinical categories with dynamic biological constraints (e.g., `smoking_years` cannot exceed `age`) and color-coded triage guidance.
* **Preventing Pipeline Skew & Dependency Drift:**  
  To avoid data transformation mismatches between training and serving, the `ColumnTransformer` (`OrdinalEncoder` with `unknown_value=-1`) and `XGBClassifier` were bundled into a single `Scikit-learn` `Pipeline` artifact (`xgboost_pipeline.joblib`). The entire application was then containerized into isolated Docker services connected via an internal bridge network (`dss-network`).
## System Architecture & Software Implementation

### 1. Three-Tier Architecture Plan & End-to-End Workflow
Transitioning a validated machine learning model from a research simulation into daily clinical practice requires more than mathematical accuracy; as highlighted in clinical literature (e.g., Klarenbeek et al.), adoption depends heavily on usability, low latency, and seamless integration into the medical workflow. To achieve this, the Decision Support System (DSS) was engineered using a decoupled **Three-Tier Architecture** rather than a monolithic structure, enforcing a strict separation of concerns across three independent layers:
* **Presentation Tier (Frontend UI):** Built with **Streamlit**, strictly responsible for user interaction, clinical data collection, and visual triage feedback without executing any underlying algorithmic logic.
* **Application / Business Logic Tier (Backend API):** An asynchronous RESTful API powered by **FastAPI** and **Pydantic**, acting as the central mediator and data validation firewall.
* **Data & ML Serving Tier:** A unified, serialized **Scikit-Learn & XGBoost Pipeline** (`joblib`) that handles feature transformation and probability estimation.

**End-to-End Execution Flow:**
1. The clinician inputs the patient's 25 parameters into the Streamlit form during the consultation.
2. Streamlit packages the inputs into a structured JSON payload and dispatches an `HTTP POST` request to `http://api:8000/predict`.
3. FastAPI intercepts the request and validates it against the Pydantic schema. If valid, it passes the data to the loaded ML pipeline; otherwise, it halts execution and returns an `HTTP 422` error.
4. The pipeline's `ColumnTransformer` encodes the categorical features, and the `XGBClassifier` computes both the binary class (`0` or `1`) and the continuous risk probability via `.predict_proba()`.
5. The API responds in `<50ms` with a structured JSON object, which Streamlit deserializes to render color-coded clinical triage guidance.

---

### 2. Data & ML Serving Pipeline (`joblib` & `ColumnTransformer`)
While 10-fold cross-validation was used to evaluate generalizability (achieving 95.64% Accuracy and 91.32% Recall on unseen folds), the final production model was trained on 100% of the cleaned dataset (5,000 records) to maximize pattern recognition prior to deployment. 

To prevent *data skew* or *pipeline breakage* between training and live inference, preprocessing and classification were unified into a single software artifact (`xgboost_pipeline.joblib`):
* **Feature Separation:** After permanently dropping the 4 data-leakage variables (`xray_abnormal`, `copd`, `crp_level`, `fev1_x10`), the remaining 25 first-line features are split into 15 categorical (nominal/ordinal) and 10 numerical variables.
* **Automated Preprocessing (`ColumnTransformer`):** Categorical variables are routed through Scikit-Learn's `OrdinalEncoder` configured with `handle_unknown='use_encoded_value'` and `unknown_value=-1` to prevent runtime crashes if an unexpected category is encountered. Numerical variables are set to `passthrough`, as tree-based splits are invariant to monotonic feature scaling.
* **Serialization:** The `preprocessor` and `XGBClassifier(eval_metric='logloss', random_state=42)` are bound inside a Scikit-Learn `Pipeline` and serialized with `joblib`. It is loaded into RAM once upon server startup, eliminating intermediate data conversion steps during live requests.

---

### 3. Backend Development & Data Validation Firewall (FastAPI & Pydantic)
The backend service (`api.py`) runs on the ASGI server **Uvicorn**, providing asynchronous request handling and sub-second response times necessary for uninterrupted clinical consultations. It exposes two REST endpoints:
* `GET /`: A health-check endpoint confirming the server is active and ready to accept classification requests.
* `POST /predict`: The core inference engine. If the `.joblib` pipeline fails to load at startup, the endpoint safely raises an `HTTP 500 Internal Server Error` rather than attempting unverified predictions.

**The Pydantic Firewall (`PatientData`):**
In medical informatics, incomplete or noisy inputs can lead to dangerous misclassifications. By inheriting from Pydantic's `BaseModel`, the `PatientData` class enforces strict type hints across all 25 required attributes (`int` and `float`):
* **Completeness & Type Safety:** If a field is missing or a string is passed instead of a number, FastAPI immediately blocks the request and returns a detailed `HTTP 422 Unprocessable Entity` response.
* **Leakage Prevention in Production:** Any attempt to inject post-diagnostic variables (like `xray_abnormal` or `crp_level`) into the JSON payload is stripped or rejected at the schema level, guaranteeing the model only evaluates approved primary-care predictors.
* **Output Formatting:** Validated objects are converted via `.model_dump()` into a single-row Pandas DataFrame, evaluated by the pipeline, and returned as an `HTTP 200 OK` JSON response containing `"prediction"`, `"risk_probability"`, and `"assessment"` (`"High Risk"` or `"Low Risk"`). Unexpected runtime errors are caught and returned as `HTTP 400 Bad Request`.

---

### 4. Frontend UI & Clinical Triage Workflow (Streamlit)
Complex, cluttered software interfaces cause cognitive fatigue and time delays—two major barriers to CDSS adoption in oncology. The frontend (`app.py`) was built in **Streamlit** with a focus on ergonomics, rapid data entry, and immediate visual interpretability.

* **Thematic 4-Column Form (`st.form`):** To mirror standard medical history-taking and prevent unnecessary page reloads on every keystroke, all 25 inputs are wrapped inside `st.form("patient_data_form")` and distributed evenly across four parallel columns (`st.columns(4)`):
  1. **Demographics & Vitals:** `age` (18–100), `gender` (Male/Female mapped to `1`/`0`), `bmi`, `oxygen_saturation` (%), `education_years`, and `income_level` (1–4).
  2. **Smoking History:** `smoker` status, `smoking_years`, `cigarettes_per_day`, `pack_years`, and `passive_smoking`. **Dynamic Biological Constraint:** The maximum allowable value for `smoking_years` dynamically updates to match the patient's entered `age` (`max_value=int(age)`), making it impossible to enter biologically invalid smoking durations.
  3. **Environmental & Medical:** `air_pollution_index` (AQI 0–300), `occupational_exposure`, `radon_exposure`, `family_history_cancer`, `asthma`, and `previous_tb`.
  4. **Symptoms & Lifestyle:** First-line symptoms (`chronic_cough`, `chest_pain`, `shortness_of_breath`, `fatigue`) alongside `exercise_hours_per_week`, `diet_quality` (1–4), `alcohol_units_per_week`, and `healthcare_access` (1–4).
* **Color-Coded Clinical Triage:** Once the clinician clicks **"Assess Risk"**, the UI translates the raw probability score into actionable clinical guidance:
  * **Low Risk (`st.success` & `st.info`):** Renders a green banner with the exact probability (e.g., `Assessment: LOW RISK (16.4%)`) and a blue recommendation box (`Recommendation: Standard preventative care`), protecting healthy patients from unnecessary radiation and keeping imaging departments from becoming overloaded.
  * **High Risk (`st.error` & `st.warning`):** Renders a red alert banner (e.g., `Assessment: HIGH RISK (75.1%)`) paired with an orange warning (`Recommendation: Immediate clinical follow-up and imaging advised`), prompting the primary care physician to immediately refer the patient for Low-Dose CT (LDCT) screening.
* **Network Fault Tolerance:** The UI catches `requests.exceptions.ConnectionError` to display a clear message if the FastAPI backend is offline, and surfaces exact server error codes if a non-200 HTTP response occurs.

---

### 5. Containerization & Microservice Orchestration (Docker & Docker Compose)
Hospital IT environments often suffer from "dependency drift" (differences in OS, Python interpreters, or library versions) and strict security constraints that make manual software installation impractical for medical staff. To ensure 100% computational reproducibility, the application was containerized into isolated microservices using **Docker**:

* **Backend Container (`Dockerfile.api`):** Built on a lightweight `python:3.9-slim` Linux image. It installs dependencies with `--no-cache-dir` to minimize image footprint, copies only `api.py` and `xgboost_pipeline.joblib` into `/app`, exposes port `8000`, and launches the Uvicorn server (`0.0.0.0:8000`).
* **Frontend Container (`Dockerfile.app`):** Also based on `python:3.9-slim`, isolating `app.py` in its own container environment, exposing Streamlit's default port `8501`, and binding the server to `0.0.0.0`.
* **Orchestration (`docker-compose.yml`):** Connects and manages both containers via **Docker Compose**:
  * **Startup Ordering (`depends_on`):** Ensures the `frontend` service waits until the `api` container is initialized before launching.
  * **Internal Bridge Network (`dss-network`):** Establishes an isolated virtual bridge network. The Streamlit container communicates with the FastAPI container securely via Docker's internal DNS (`http://api:8000/predict`) rather than hardcoded IP addresses, keeping the ML engine shielded from unauthorized external network access while exposing the UI on port `8501`.
  * **Zero-Configuration Deployment:** Enables any clinic or hospital workstation to spin up the entire three-tier system with a single command (`docker compose up --build`).
---

## 🚀 How to Run Instructions

### Option 1: Run with Docker Compose (Recommended)
Docker Compose builds and orchestrates both the FastAPI backend and Streamlit frontend in isolated containers with zero manual configuration.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/lung-cancer-cdss.git
   cd lung-cancer-cdss
   ```
2. **Build and start the microservices:**
   ```bash
   docker compose up --build
   ```
3. **Access the services:**
   * **Clinical Web Interface (Streamlit):** [http://localhost:8501](http://localhost:8501)
   * **Backend REST API & Swagger Docs (FastAPI):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Run Locally (Python Environment)
If you prefer to run the services directly on your host machine:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the FastAPI Backend Server (Terminal 1):**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
3. **Launch the Streamlit Frontend (Terminal 2):**
   *(Note: If running outside Docker, ensure `API_URL` in `app.py` is set to `http://localhost:8000/predict`)*
   ```bash
   streamlit run app.py --server.port=8501
   ```

---

## 📚 Citations & References
* **[1] National Health Service (NHS)** (2022). *Lung cancer - Overview*.
* **[2] World Health Organization (WHO)** (2024). *Lung cancer - Key facts and overview*.
* **[3] American Cancer Society (ACS)** (2024). *What Is Lung Cancer?*.
* **[4] Centers for Disease Control and Prevention (CDC)** (2024). *What Are the Risk Factors for Lung Cancer?*.
* **[5] National Cancer Institute (NCI)** (2023). *Non-Small Cell Lung Cancer Treatment (PDQ®) – Patient Version*.
* **[6] International Agency for Research on Cancer (IARC) - WHO** (2024). *GLOBOCAN: Global Cancer Observatory - Lung Cancer Fact Sheet*.
* **[7] National Cancer Institute (NCI) - SEER Program** (2024). *Cancer Stat Facts: Lung and Bronchus Cancer*.
* **[8] Sutton, R. T., Pincock, D., Baumgart, D. C., Sadowski, D. C., Federau, R. N., & Dorcell, T. R.** (2020). *An overview of clinical decision support systems: benefits, risks, and strategies for success*. npj Digital Medicine, 3(1), 17.
* **[9] Bi, W. L., et al.** (2019). *Artificial intelligence in cancer imaging: Clinical challenges and applications*. CA: A Cancer Journal for Clinicians, 69(2), 127–157.
* **[10] Klarenbeek, S. E., Schuurbiers-Siebers, O. C. J., van den Heuvel, M. M., Prokop, M., & Tummers, M.** (2021). *Barriers and Facilitators for Implementation of a Computerized Clinical Decision Support System in Lung Cancer Multidisciplinary Team Meetings—A Qualitative Assessment*. Biology, 10(1), 9.
* **[11] Bisarwal, P., & Chauhan, A.** (2020). *Detection of Lung Cancer using Machine Learning Techniques Based on Routine Blood Indices*. In Proc. 2020 8th Int. Conf. Reliability, Infocom Technologies and Optimization (ICRITO), pp. 696–701.
* **[12] Çalışkan, M., & Tazaki, K.** (2023). *AI/ML advances in non-small cell lung cancer biomarker discovery*. Frontiers in Oncology, 13, 1260374.
* **[13] Li, Y., Wu, X., Yang, P., Jiang, G., & Luo, Y.** (2022). *Machine Learning for Lung Cancer Diagnosis, Treatment, and Prognosis*. Genomics, Proteomics & Bioinformatics, 20(5), 850–866.
* **[14] Cortés-Ibáñez, P. O., et al.** (2023). *Serum-based biomarkers associated with lung cancer risk and cause-specific mortality in the German randomized Lung Cancer Screening Intervention (LUSI) trial*. Translational Lung Cancer Research, 12(12), 2419–2433.
* **[15] Lundberg, S. M., & Lee, S.-I.** (2017). *A Unified Approach to Interpreting Model Predictions*. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 4765–4774.
* **[16] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O.** (2012). *Leakage in data mining: Formulation, detection, and avoidance*. ACM Transactions on Knowledge Discovery from Data (TKDD), 6(4), 15.
* **[17] Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*. In Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD '16), pp. 785–794.
* **[18] Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
* **[19] Gundersen, O. E., & Kjensmo, S.** (2018). *State of the Art: Reproducibility in Artificial Intelligence*. In Proc. 32nd AAAI Conf. Artificial Intelligence (AAAI-18), pp. 1644–1651.
* **[20] Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
* **[21] Sokolova, M., & Lapalme, G.** (2009). *A systematic analysis of performance measures for classification tasks*. Information Processing & Management, 45(4), 427–437.
* **[22] Kelly, C. J., Karthikesalingam, A., Suleyman, M., Corrado, G., & King, D.** (2019). *Key challenges for delivering clinical impact with artificial intelligence*. BMC Medicine, 17(1), 195.
* **[23] He, H., & Garcia, E. A.** (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.
* **[24] Dritsas, E., & Trigka, M.** (2022). *Lung Cancer Risk Prediction with Machine Learning Models*. Big Data and Cognitive Computing, 6(4), 139.
* **[25] Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G.** (2015). *Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement*. Annals of Internal Medicine, 162(1), 55–63.
* **[26] National Lung Screening Trial Research Team** (2011). *Reduced lung-cancer mortality with low-dose computed tomographic screening*. New England Journal of Medicine, 365(5), 395–409.
* **[27] Kapoor, S., & Narayanan, A.** (2023). *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science*. Patterns, 4(9), 100804.
* **[28] Lundberg, S. M., et al.** (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2(1), 56–67.
* **[29] Shapley, L. S.** (1953). *A Value for n-Person Games*. In Contributions to the Theory of Games (AM-28), Vol. II, pp. 307–318. Princeton University Press.
* **[30] Moons, K. G., et al.** (2015). *Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): explanation and elaboration*. Annals of Internal Medicine, 162(1), W1–W73.
* **[31] Friedman, J. H.** (2001). *Greedy function approximation: a gradient boosting machine*. The Annals of Statistics, 29(5), 1189–1232.
* **[32] Quinlan, J. R.** (1986). *Induction of decision trees*. Machine Learning, 1(1), 81–106.
* **[33] Martin, R. C.** (2017). *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall.
* **[34] Fowler, M.** (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley Professional.
* **[35] Streamlit Inc.** (2024). *Streamlit: A faster way to build and share data apps* (Documentation).
* **[36] Ramírez, S.** (2024). *FastAPI: Modern, Fast (High-Performance) Web Framework for Building APIs with Python* (Documentation).
* **[37] Colvin, S., et al.** (2024). *Pydantic: Data validation using Python type hints* (Documentation).
* **[38] Joblib Development Team** (2024). *Joblib: Running Python functions as pipeline jobs* (Documentation).
* **[39] Boettiger, C.** (2015). *An introduction to Docker for reproducible research*. ACM SIGOPS Operating Systems Review, 49(1), 71–79.
* **[40] Gruendner, J., et al.** (2019). *KETOS: Clinical decision support and machine learning as a service – A training and deployment platform based on Docker, OMOP-CDM, and FHIR Web Services*. PLOS ONE, 14(10), e0223010.
* **[41] Merkel, D.** (2014). *Docker: lightweight Linux containers for consistent execution*. Linux Journal, 2014(239), 2.
* **[42] Docker Inc.** (2024). *Docker Documentation: Accelerated container application development*.
D. Merkel, "Docker: lightweight Linux containers for consistent execution," Linux Journal, vol. 2014, no. 239, p. 2, 2014.
