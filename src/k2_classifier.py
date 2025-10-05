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

# Load K2 CSV file from local /data directory
df = pd.read_csv('./data/k2.csv')

# Features (K2-specific parameters)
features = [
    'pl_orbper', 'pl_rade', 'pl_eqt', 'pl_insol',
    'st_teff', 'st_logg', 'st_rad'
]

target = 'disposition'

# Standardize disposition labels
df[target] = df[target].str.upper().replace({
    'FALSE': 'FALSE POSITIVE',
    'NOT DISPOSITIONED': 'CANDIDATE',
    'UNCONFIRMED': 'CANDIDATE'
})

# Preprocess: Drop rows with missing target/features, fill others with median
df = df.dropna(subset=[target] + features)
for col in features:
    df[col] = df[col].fillna(df[col].median())

X = df[features]
y = df[target]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Debug: Print unique classes
print("Unique disposition classes:", le.classes_)

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
print(f"K2 Test Accuracy: {acc:.4f}")
print("K2 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, labels=range(len(le.classes_))))

# Save model, label encoder, and scaler in ./models directory
dump(model, './models/k2_model.joblib')
dump(le, './models/k2_label_encoder.joblib')
dump(scaler, './models/k2_scaler.joblib')
print("K2 model, label encoder, and scaler saved in ./models/")
