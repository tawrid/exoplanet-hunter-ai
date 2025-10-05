# Exoplanet Hunter AI - Sauron's Eye

## Setup (Python 3.13 on macOS)
1. Install Python 3.13 from python.org.
2. Clone repo: `git clone https://github.com/tawrid/exoplanet-hunter-ai.git`
3. Create venv: `python -m venv venv` (activate: `source venv/bin/activate`).
4. Install: `pip install -r requirements.txt` (updated for 3.13; no build errors).
5. Train: `python train.py` (updated TAP API for full columns like koi_smass; expect ~0.99 accuracy).
6. Run app: `streamlit run app.py`

## Deployment
- **GCP App Engine**: Set GCP_PROJECT_ID and GCP_SA_KEY in GitHub secrets. Push to main triggers deploy (uses Python 3.13 runtime).
- **Azure App Service**: Use Azure CLI: `az webapp up --runtime "PYTHON|3.13" --sku B1`.
- **Local/GitHub Pages**: Run locally as above. For static demo, see /static branch (future: export to HTML via streamlit-export).

## Usage
- Novices: Select the model from left navigation → Upload unlabeled CSV → Get predictions.
- Researchers: Select the model from left navigation → Tweak params → Retrain → Ingest new labeled data. View confusion matrix in expander.

## Screenshot
![Screenshot](images/screenshot_saurons_eye.png)


## Recorded Video Link
https://github.com/tawrid/exoplanet-hunter-ai/tree/main/videos/video_record.mp4


Questions? Open issue!