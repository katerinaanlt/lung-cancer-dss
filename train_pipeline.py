import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Load the initial dataset
df = pd.read_csv('lung_cancer.csv')

# Remove columns that cause Data Leakage
leakage_cols = ['xray_abnormal', 'copd', 'crp_level', 'fev1_x10']
df = df.drop(columns=leakage_cols)

# Separate features (X) and target variable (y)
X = df.drop(columns=['lung_cancer_risk'])
y = df['lung_cancer_risk']

# Define categorical and numerical columns
categorical_cols = [
    'gender', 'income_level', 'smoker', 'passive_smoking',
    'occupational_exposure', 'radon_exposure', 'family_history_cancer',
    'asthma', 'previous_tb', 'chronic_cough', 'chest_pain',
    'shortness_of_breath', 'fatigue', 'diet_quality', 'healthcare_access'
]

# Dynamically identify the remaining columns as numerical
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# Build the unified Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
])

# Train the pipeline on the full dataset (5,000 samples)
pipeline.fit(X, y)

# Serialize and export the final artifact
export_filename = 'xgboost_pipeline.joblib'
joblib.dump(pipeline, export_filename)
print(f"Model successfully exported to {export_filename}")
