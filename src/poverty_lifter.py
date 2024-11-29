# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('data/raw/train_data.csv')
submission_df = pd.read_csv('data/raw/submission_data.csv')
template_df = pd.read_csv('data/processed/submission_template.csv')

# Replace placeholders with NaN
train_df.replace([-1, -1.0, '-1', '-1.0'], np.nan, inplace=True)
submission_df.replace([-1, -1.0, '-1', '-1.0'], np.nan, inplace=True)

# Convert date columns to datetime
date_cols = ["launch_date", "ind_launch_date", "date"]
for col in date_cols:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce')
    submission_df[col] = pd.to_datetime(submission_df[col], errors='coerce')

# Feature Engineering Functions
def extract_date_features(df):
    df['launch_year'] = df['launch_date'].dt.year
    df['launch_month'] = df['launch_date'].dt.month
    df['launch_day'] = df['launch_date'].dt.day
    df['ind_launch_year'] = df['ind_launch_date'].dt.year
    df['ind_launch_month'] = df['ind_launch_date'].dt.month
    df['ind_launch_day'] = df['ind_launch_date'].dt.day
    df['date_year'] = df['date'].dt.year
    df['date_month'] = df['date'].dt.month
    df['date_day'] = df['date'].dt.day
    # Time since launch
    # Calculate months_since_launch
    df['months_since_launch'] = (
        (df['date'].dt.year - df['launch_date'].dt.year) * 12 +
        (df['date'].dt.month - df['launch_date'].dt.month)
    ).fillna(0)
    df['months_since_ind_launch'] = (
        (df['date'].dt.year - df['ind_launch_date'].dt.year) * 12 +
        (df['date'].dt.month - df['ind_launch_date'].dt.month)
    ).fillna(0)   
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['date_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date_month'] / 12)
    return df

def create_market_features(df):
    df['market_size'] = df['population'] * df['prev_perc']
    df['affordability_index'] = df['che_pc_usd'] / df['price_month']
    df['price_insurance_interaction'] = df['price_month'] * (1 - df['insurance_perc_che'])
    return df

# Apply feature engineering functions
train_df = extract_date_features(train_df)
train_df = create_market_features(train_df)
submission_df = extract_date_features(submission_df)
submission_df = create_market_features(submission_df)

# Handle 'indication' feature (multi-label encoding)
def process_indication(df):
    df['indication'] = df['indication'].apply(lambda x: eval(x) if pd.notnull(x) else [])
    return df

train_df = process_indication(train_df)
submission_df = process_indication(submission_df)

# Split the data into training and testing sets based on launch_date
perc_train_samples = 0.8
launches = train_df.groupby('cluster_nl')['launch_date'].first().reset_index()
launches = launches.sort_values('launch_date')
cutoff = int(len(launches) * perc_train_samples)
cutoff_launch_date = launches.iloc[cutoff]['launch_date']
train_cluster_nls = launches.iloc[:cutoff]['cluster_nl']
test_cluster_nls = launches.iloc[cutoff:]['cluster_nl']

train_data = train_df.loc[train_df['cluster_nl'].isin(train_cluster_nls)]
test_data = train_df.loc[train_df['cluster_nl'].isin(test_cluster_nls)]

# Separate features and target
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

# Save cluster_nl and date for metric computation
metric_df = test_data[['cluster_nl', 'date', 'target']].copy()

# Define custom transformer for target encoding
class TargetEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoders = {}
        
    def fit(self, X, y):
        for col in self.cols:
            encoder = TargetEncoder()
            encoder.fit(X[col], y)
            self.encoders[col] = encoder
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols:
            X_transformed[col] = self.encoders[col].transform(X[col])
        return X_transformed

# Define columns
numeric_features = [
    'che_pc_usd', 'che_perc_gdp', 'insurance_perc_che', 'population', 'prev_perc', 
    'price_month', 'price_unit', 'public_perc_che', 'months_since_launch', 
    'months_since_ind_launch', 'market_size', 'affordability_index', 
    'price_insurance_interaction', 'month_sin', 'month_cos'
]

categorical_features_low = ['corporation', 'country', 'therapeutic_area']
categorical_features_high = ['brand', 'drug_id']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer_low = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

categorical_transformer_high = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("target_encoder", TargetEncodingTransformer(cols=categorical_features_high))
])

# Process 'indication' with MultiLabelBinarizer
class IndicationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        
    def fit(self, X, y=None):
        self.mlb.fit(X['indication'])
        return self
    
    def transform(self, X):
        indication_encoded = self.mlb.transform(X['indication'])
        indication_df = pd.DataFrame(indication_encoded, columns=self.mlb.classes_, index=X.index)
        X = X.join(indication_df)
        X = X.drop(columns=['indication'])
        return X

# Combine all preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat_low", categorical_transformer_low, categorical_features_low),
        ("cat_high", 'passthrough', categorical_features_high)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("indication_transformer", IndicationTransformer()),
    ("preprocessor", preprocessor),
    ("target_encoder", TargetEncodingTransformer(cols=categorical_features_high)),
    ("scaler", StandardScaler()),
    ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])

# Custom CYME metric
def cyme_metric(y_true, y_pred):
    ape = np.abs((y_true - y_pred) / y_true)
    ape = ape.replace([np.inf, -np.inf], np.nan).dropna()
    median_ape = np.median(ape)
    return median_ape

cyme_scorer = make_scorer(cyme_metric, greater_is_better=False)

# Hyperparameter tuning
param_dist = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__learning_rate": [0.01, 0.03, 0.1],
    "regressor__max_depth": [5, 7, 9],
    "regressor__subsample": [0.7, 0.9, 1.0],
    "regressor__colsample_bytree": [0.7, 0.9, 1.0]
}

# Time-based cross-validation
tscv = TimeSeriesSplit(n_splits=3)

random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=tscv, 
    scoring=cyme_scorer, 
    verbose=2, 
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Evaluate the model
def predict_and_measure_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE:  {mse}")
    # Assuming unaveraged_CYME is defined appropriately
    cyme = cyme_metric(y_test, y_pred)
    print(f"CYME: {cyme}")
    return mse, cyme

mse, cyme = predict_and_measure_performance(best_model, X_test, y_test)

# Feature Importance using SHAP
xgb_regressor = best_model.named_steps["regressor"]
# Prepare data for SHAP
X_test_transformed = best_model.named_steps["preprocessor"].transform(X_test)
explainer = shap.TreeExplainer(xgb_regressor)
shap_values = explainer.shap_values(X_test_transformed)

# Plot summary plot
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

# Prepare submission
# Apply the same preprocessing to the submission data
submission_df = submission_df.drop(columns=['target'], errors='ignore')
submission_df = best_model.named_steps["indication_transformer"].transform(submission_df)
submission_df = best_model.named_steps["target_encoder"].transform(submission_df)
X_submission = submission_df.copy()

# Predict on submission data
predictions = best_model.predict(X_submission)

# Merge predictions with template
submission_df['prediction'] = predictions
submission_df['date'] = pd.to_datetime(submission_df['date'], errors='coerce')

final_submission = template_df.merge(
    submission_df[['cluster_nl', 'date', 'prediction']],
    on=['cluster_nl', 'date'],
    how='left'
)

# Fill missing predictions with zeros or appropriate values
final_submission['prediction'] = final_submission['prediction'].fillna(0)

# Save the submission file
final_submission[['cluster_nl', 'date', 'prediction']].to_csv('../data/outputs/final_submission.csv', index=False)
