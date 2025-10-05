import streamlit as st
import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Apply custom CSS for a cosmic background
st.markdown(
    """
    <style>
    /* Main content background */
    .main {
        background-image: linear-gradient(rgba(0, 0, 50, 0.7), rgba(50, 0, 100, 0.7)), 
                         url('https://images.unsplash.com/photo-1464802686167-b939a6910659?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        color: white;
    }
    /* Sidebar background */
    .css-1x8cf1d, .sidebar-content {
        background-image: linear-gradient(rgba(0, 0, 50, 0.8), rgba(50, 0, 100, 0.8)), 
                         url('https://images.unsplash.com/photo-1464802686167-b939a6910659?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        color: white;
    }
    /* Ensure text readability */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }
    /* Style buttons for better contrast */
    .stButton>button {
        background-color: #4B0082;
        color: white;
        border: 1px solid #ffffff;
    }
    .stButton>button:hover {
        background-color: #6A0DAD;
        color: white;
    }
    /* Style sliders for visibility */
    .stSlider [data-baseweb="slider"] {
        background-color: transparent;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #4B0082;
    }
    /* Ensure dataframe and other widgets are readable */
    .stDataFrame, .stTable {
        background-color: rgba(0, 0, 50, 0.8);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Model-specific configurations
MODEL_CONFIGS = {
    'Kepler': {
        'model_file': 'models/kepler_model.joblib',
        'encoder_file': 'models/kepler_label_encoder.joblib',
        'scaler_file': 'models/kepler_scaler.joblib',
        'features': [
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad'
        ],
        'log_features': ['koi_period', 'koi_insol', 'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_depth'],
        'target': 'koi_disposition',
        'data_file': 'data/kepler.csv'
    },
    'K2': {
        'model_file': 'models/k2_model.joblib',
        'encoder_file': 'models/k2_label_encoder.joblib',
        'scaler_file': 'models/k2_scaler.joblib',
        'features': [
            'pl_orbper', 'pl_rade', 'pl_eqt', 'pl_insol',
            'st_teff', 'st_logg', 'st_rad'
        ],
        'log_features': ['pl_orbper', 'pl_insol', 'pl_rade', 'pl_eqt'],
        'target': 'disposition',
        'data_file': 'data/k2.csv'
    },
    'TESS': {
        'model_file': 'models/tess_model.joblib',
        'encoder_file': 'models/tess_label_encoder.joblib',
        'scaler_file': 'models/tess_scaler.joblib',
        'features': [
            'pl_orbper', 'pl_tranmid', 'pl_trandurh', 'pl_trandep',
            'pl_rade', 'pl_eqt', 'pl_insol', 'st_teff', 'st_logg', 'st_rad'
        ],
        'log_features': ['pl_orbper', 'pl_insol', 'pl_rade', 'pl_eqt', 'pl_trandep'],
        'target': 'tfopwg_disp',
        'data_file': 'data/tess.csv'
    }
}

# Function to get data
def get_data(model_type):
    config = MODEL_CONFIGS[model_type]
    try:
        df = pd.read_csv(config['data_file'], engine='python', on_bad_lines='warn')
    except FileNotFoundError:
        st.error(f"Data file {config['data_file']} not found in ./data directory.")
        return None
    return df

# Load model/encoder/scaler
@st.cache_resource
def load_model(model_type):
    config = MODEL_CONFIGS[model_type]
    try:
        model = load(config['model_file'])
        le = load(config['encoder_file'])
        scaler = load(config['scaler_file'])
        return model, le, scaler
    except FileNotFoundError:
        st.warning(f"No {model_type} model found in ./models directory. Run the corresponding classifier script first.")
        return None, None, None

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(model_type):
    config = MODEL_CONFIGS[model_type]
    df = get_data(model_type)
    if df is None:
        return None, None, None, None
    features = config['features']
    log_features = config['log_features']
    target = config['target']
    
    # Standardize disposition labels for K2 and TESS
    if model_type in ['K2', 'TESS']:
        df[target] = df[target].str.upper().replace({
            'FALSE': 'FALSE POSITIVE',
            'NOT DISPOSITIONED': 'CANDIDATE',
            'UNCONFIRMED': 'CANDIDATE'
        })
    
    df = df.dropna(subset=[target] + features)
    for col in features:
        df[col] = df[col].fillna(df[col].median())
    for col in log_features:
        df[col] = np.log1p(df[col])
    X = df[features]
    y = df[target]
    le_data = LabelEncoder()
    y_encoded = le_data.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    return X_test, y_test, le_data.classes_, features

def plot_confusion_matrix(model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig

st.title("🌌 Exoplanet Hunter: Sauron's Eye")
st.markdown("Upload transit data to classify as CONFIRMED, CANDIDATE, or FALSE POSITIVE. Select a model for prediction and retraining.")

# Model selection dropdown
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox("Select Model", options=list(MODEL_CONFIGS.keys()), index=0)
model, le, scaler = load_model(model_type)

# Model Stats Section
if model is not None:
    with st.expander(f"View {model_type} Model Statistics & Confusion Matrix"):
        X_test, y_test, classes, features = load_and_preprocess_data(model_type)
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            fig = plot_confusion_matrix(model, X_test_scaled, y_test, classes)
            st.pyplot(fig)
            y_pred_test = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred_test)
            st.metric("Test Accuracy", f"{acc:.4f}")
            st.text(classification_report(y_test, y_pred_test, target_names=classes, labels=range(len(classes))))
            if 'computed_acc' not in st.session_state:
                st.session_state.computed_acc = acc

# Sidebar for hyperparams with clickable info buttons
st.sidebar.header("Hyperparameter Tuning")

# Initialize session state for toggling parameter info
if 'show_n_estimators' not in st.session_state:
    st.session_state.show_n_estimators = False
if 'show_max_depth' not in st.session_state:
    st.session_state.show_max_depth = False
if 'show_learning_rate' not in st.session_state:
    st.session_state.show_learning_rate = False
if 'show_gamma' not in st.session_state:
    st.session_state.show_gamma = False

# n_estimators
col1, col2 = st.sidebar.columns([4, 1])
with col1:
    st.markdown("**n_estimators**")
with col2:
    if st.button("ℹ️", key="n_estimators_info"):
        st.session_state.show_n_estimators = not st.session_state.show_n_estimators
if st.session_state.show_n_estimators:
    st.sidebar.markdown(
        "Number of boosting rounds. Higher values can improve accuracy but may lead to overfitting and longer training times. Typical range: 100–1000."
    )
n_estimators = st.sidebar.slider("Number of estimators", 100, 1000, 300, key="n_estimators", label_visibility="collapsed")

# max_depth
col1, col2 = st.sidebar.columns([4, 1])
with col1:
    st.markdown("**max_depth**")
with col2:
    if st.button("ℹ️", key="max_depth_info"):
        st.session_state.show_max_depth = not st.session_state.show_max_depth
if st.session_state.show_max_depth:
    st.sidebar.markdown(
        "Maximum depth of each tree. Controls model complexity; shallower trees (lower values) reduce overfitting but may underfit if too low. Typical range: 3–15."
    )
max_depth = st.sidebar.slider("Maximum depth", 3, 15, 6, key="max_depth", label_visibility="collapsed")

# learning_rate
col1, col2 = st.sidebar.columns([4, 1])
with col1:
    st.markdown("**learning_rate**")
with col2:
    if st.button("ℹ️", key="learning_rate_info"):
        st.session_state.show_learning_rate = not st.session_state.show_learning_rate
if st.session_state.show_learning_rate:
    st.sidebar.markdown(
        "Step size shrinkage used in updates to prevent overfitting. Lower values require more trees but can improve generalization. Typical range: 0.05–0.3."
    )
learning_rate = st.sidebar.slider("Learning rate", 0.05, 0.3, 0.1, key="learning_rate", label_visibility="collapsed")

# gamma
col1, col2 = st.sidebar.columns([4, 1])
with col1:
    st.markdown("**gamma**")
with col2:
    if st.button("ℹ️", key="gamma_info"):
        st.session_state.show_gamma = not st.session_state.show_gamma
if st.session_state.show_gamma:
    st.sidebar.markdown(
        "Minimum loss reduction required to make a further partition on a leaf node. Lower values allow more nuanced splits, while higher values make the model more conservative. Typical range: 0.0–0.5."
    )
gamma = st.sidebar.slider("Gamma", 0.0, 0.5, 0.1, key="gamma", label_visibility="collapsed")

retrain = st.sidebar.button("Retrain Model")

if retrain and model is not None:
    config = MODEL_CONFIGS[model_type]
    df = get_data(model_type)
    if df is not None:
        features = config['features']
        log_features = config['log_features']
        target = config['target']
        
        if model_type in ['K2', 'TESS']:
            df[target] = df[target].str.upper().replace({
                'FALSE': 'FALSE POSITIVE',
                'NOT DISPOSITIONED': 'CANDIDATE',
                'UNCONFIRMED': 'CANDIDATE'
            })
        
        df = df.dropna(subset=[target] + features)
        for col in features:
            df[col] = df[col].fillna(df[col].median())
        for col in log_features:
            df[col] = np.log1p(df[col])
        X = df[features]
        y = df[target]
        le_new = LabelEncoder()
        y_encoded = le_new.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Dynamically adjust k_neighbors for SMOTE based on minority class size
        class_counts = pd.Series(y_train).value_counts()
        min_class_size = min(class_counts)
        k_neighbors = min(5, max(1, min_class_size - 1))  # Use 5 or fewer, ensuring at least 1
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except ValueError as e:
            st.sidebar.error(
                f"Failed to retrain {model_type} model due to insufficient samples in one or more classes. "
                f"Minimum class size: {min_class_size}. Please add more data for the minority class(es)."
            )
            st.stop()
        
        scaler_new = StandardScaler()
        X_train_res_scaled = scaler_new.fit_transform(X_train_res)
        X_test_scaled = scaler_new.transform(X_test)
        
        new_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            gamma=gamma,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        new_model.fit(X_train_res_scaled, y_train_res)
        y_pred = new_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.sidebar.success(f"Retrained {model_type} Model! New Accuracy: {acc:.4f}")
        dump(new_model, config['model_file'])
        dump(le_new, config['encoder_file'])
        dump(scaler_new, config['scaler_file'])
        st.session_state.computed_acc = acc
        st.rerun()
        model, le, scaler = load_model(model_type)

# Upload section
config = MODEL_CONFIGS[model_type]
features = config['features']
log_features = config['log_features']
target = config['target']
uploaded_file = st.file_uploader(f"Upload CSV for {model_type} (columns: " + ", ".join(features[:5]) + "...)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='warn')
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())
    
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}. Add them or use sample {model_type} data.")
    else:
        df_filled = df.copy()
        for col in features:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        for col in log_features:
            if col in df_filled.columns:
                df_filled[col] = np.log1p(df_filled[col])
        X_new = df_filled[features]
        
        if model is not None:
            X_new_scaled = scaler.transform(X_new)
            probs = model.predict_proba(X_new_scaled)
            confidences = np.max(probs, axis=1)
            preds = np.argmax(probs, axis=1)
            
            threshold = 0.5
            fp_label = le.transform(['FALSE POSITIVE'])[0]
            adjusted_preds = preds.copy()
            low_conf_mask = confidences < threshold
            adjusted_preds[low_conf_mask] = fp_label
            df_filled['predicted_disposition'] = le.inverse_transform(adjusted_preds)
            df_filled['confidence'] = confidences
            
            st.subheader("Predictions")
            display_cols = ['predicted_disposition', 'confidence']
            if target in df_filled.columns:
                display_cols = [target] + display_cols
            st.dataframe(df_filled[display_cols])
            
            if model_type == 'Kepler':
                st.subheader("FP Flag Summary (Top Contributors to Predictions)")
                flag_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
                flag_cols = [col for col in flag_cols if col in df_filled.columns]
                if flag_cols:
                    flag_summary = df_filled[flag_cols].sum().to_frame('Count Set to 1')
                    st.dataframe(flag_summary)
            
            csv = df_filled.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
            
            st.subheader("Model Statistics")
            unique_preds, counts = np.unique(adjusted_preds, return_counts=True)
            st.write(pd.DataFrame({'Disposition': le.inverse_transform(unique_preds), 'Count': counts}))
            
            computed_acc = st.session_state.get('computed_acc', 0.0)
            st.info(f"Overall Accuracy (from {model_type} training): {acc:.4f}")
            
            if target in df.columns:
                ingest = st.checkbox("Ingest this labeled data to update model?")
                if ingest:
                    current_df = get_data(model_type)
                    if current_df is not None:
                        new_data = df_filled[features + [target]]
                        combined = pd.concat([current_df, new_data], ignore_index=True)
                        if model_type in ['K2', 'TESS']:
                            combined[target] = combined[target].str.upper().replace({
                                'FALSE': 'FALSE POSITIVE',
                                'NOT DISPOSITIONED': 'CANDIDATE',
                                'UNCONFIRMED': 'CANDIDATE'
                            })
                        combined = combined.dropna(subset=[target] + features)
                        for col in features:
                            combined[col] = combined[col].fillna(combined[col].median())
                        for col in log_features:
                            if col in combined.columns:
                                combined[col] = np.log1p(combined[col])
                        combined.to_csv(f'updated_{model_type.lower()}_data.csv', index=False)
                        X = combined[features]
                        y = combined[target]
                        le_new = LabelEncoder()
                        y_encoded = le_new.fit_transform(y)
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
                        
                        # Dynamically adjust k_neighbors for SMOTE
                        class_counts = pd.Series(y_train).value_counts()
                        min_class_size = min(class_counts)
                        k_neighbors = min(5, max(1, min_class_size - 1))  # Use 5 or fewer, ensuring at least 1
                        try:
                            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                        except ValueError as e:
                            st.error(
                                f"Failed to ingest and retrain {model_type} model due to insufficient samples in one or more classes. "
                                f"Minimum class size: {min_class_size}. Please add more data for the minority class(es)."
                            )
                            st.stop()
                        
                        scaler_new = StandardScaler()
                        X_train_res_scaled = scaler_new.fit_transform(X_train_res)
                        X_test_scaled = scaler_new.transform(X_test)
                        
                        new_model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            gamma=gamma,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            eval_metric='mlogloss'
                        )
                        new_model.fit(X_train_res_scaled, y_train_res)
                        y_pred = new_model.predict(X_test_scaled)
                        acc = accuracy_score(y_test, y_pred)
                        st.success(f"Data ingested and {model_type} model updated! New Accuracy: {acc:.4f}")
                        dump(new_model, config['model_file'])
                        dump(le_new, config['encoder_file'])
                        dump(scaler_new, config['scaler_file'])
                        st.session_state.computed_acc = acc
                        updated_csv = combined.to_csv(index=False)
                        st.download_button(
                            label=f"Download Updated {model_type} Dataset as CSV",
                            data=updated_csv,
                            file_name=f'updated_{model_type.lower()}_data.csv',
                            mime='text/csv'
                        )
                        st.rerun()
                        model, le, scaler = load_model(model_type)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Built for NASA Space Apps 2025. Data: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)")
st.markdown("🚀 **Exoplanet Hunter: Sauron's Eye**")