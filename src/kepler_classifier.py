import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Ensure models directory exists
os.makedirs('./models', exist_ok=True)

# Load Kepler CSV file from local /data directory
df = pd.read_csv('data/kepler.csv', on_bad_lines='warn', engine='python')

# Features (Kepler-specific transit parameters)
features = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_steff', 'koi_slogg', 'koi_srad'
]

target = 'koi_disposition'

# Preprocess: Drop rows with missing target/features, fill others with median
df = df.dropna(subset=[target] + features)
for col in features:
    df[col] = df[col].fillna(df[col].median())

X = df[features]
y = df[target]

# Encode labels: FALSE POSITIVE=0, CANDIDATE=1, CONFIRMED=2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Kepler Test Accuracy: {acc:.4f}")
print("Kepler Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model, label encoder, and scaler in ./models directory
dump(model, './models/kepler_model.joblib')
dump(le, './models/kepler_label_encoder.joblib')
dump(scaler, './models/kepler_scaler.joblib')
print("Kepler model, label encoder, and scaler saved in ./models/")
