import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load('model.joblib')
    le = joblib.load('label_encoder.joblib')
    # Load small test data (assume data/test.csv exists)
    df_test = pd.read_csv('./data/sample_input.csv')
    features = [
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
        'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
        'koi_steff', 'koi_slogg', 'koi_smass', 'koi_srad'
    ]
    X_test = df_test[features]
    y_test = le.transform(df_test['koi_disposition'])
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.9