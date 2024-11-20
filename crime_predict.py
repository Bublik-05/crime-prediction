import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
import numpy as np

# Suppress warnings for readability
warnings.filterwarnings("ignore")
pd.set_option('display.max.columns', None)

# Load and preprocess data
crime_data = pd.read_csv('crimedata.csv')
code_cols = ['countyCode', 'communityCode']
for col in code_cols:
    crime_data[col] = crime_data[col].fillna(value=0).astype('int')

suffixes = ['township', 'city', 'borough']
for suffix in suffixes:
    crime_data['communityName'] = crime_data['communityName'].str.replace(suffix, '', regex=True)

# Define feature and target columns
feature_columns = [
    'population', 'racepctblack', 'racePctWhite', 'agePct12t21', 'agePct65up',
    'medIncome', 'PctPopUnderPov', 'PctUnemployed', 'PctEmplManu', 'PctEmploy',
    'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'LemasSwornFT',
    'PolicReqPerOffic', 'PolicBudgPerPop'
]
target_column = 'murders'

# Prepare training and testing sets
X = crime_data[feature_columns].fillna(0)
y = crime_data[target_column].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best hyperparameters from previous tuning
best_rf_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10}
best_gb_params = {'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_depth': 7, 'learning_rate': 0.1}

# Initialize models with tuned parameters
rf = RandomForestRegressor(random_state=42, **best_rf_params)
gb = GradientBoostingRegressor(random_state=42, **best_gb_params)

# Train models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Ensemble by averaging predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
ensemble_pred = (rf_pred + gb_pred) / 2

# Evaluate ensemble
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print("Ensemble Model - Mean Squared Error:", ensemble_mse)
print("Ensemble Model - Mean Absolute Error:", ensemble_mae)

joblib.dump(rf, 'rf_model.joblib')
joblib.dump(gb, 'gb_model.joblib')
