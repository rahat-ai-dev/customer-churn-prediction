# Customer Churn Prediction — Ensemble ML Project

End-to-end churn prediction pipeline using an **ensemble of models** (Random
Forest + XGBoost + LightGBM combined via Stacking / Voting), trained on the
real-world **Telco Customer Churn** dataset from Kaggle. Includes a
**Streamlit web app** so anyone can fill in a customer's details and get a
churn prediction instantly — no code required.

**Live demo:** _add your Streamlit Cloud link here after deploying (see
Section 7 below), e.g. `https://your-app-name.streamlit.app`_

Real results on this dataset (7,043 customers): **ROC-AUC ≈ 0.845** with the
voting ensemble — see `models/metrics_report.json` for the full comparison
across all 6 models.

## 1. Dataset

Source (Kaggle): `blastchar/telco-customer-churn`
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

7,043 telecom customers, 21 columns (demographics, account info, services
subscribed, monthly/total charges, and the target `Churn`: Yes/No).

### Download it (pick one)

**Option A — Kaggle API (recommended)**
```bash
pip install kaggle
# put your kaggle.json (API token) in ~/.kaggle/kaggle.json
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

**Option B — Manual**
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle page above and
place it at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

> Note: this container has no internet access to kaggle.com, so the dataset
> must be downloaded on your own machine and dropped into `data/raw/`
> (or run the Kaggle API command locally before pushing to wherever you train).

## 2. Project structure

```
churn-ensemble/
├── data/
│   ├── raw/                 # original Kaggle CSV goes here
│   └── processed/           # cleaned / feature-engineered data (auto-generated)
├── src/
│   ├── config.py            # paths, constants, hyperparams
│   ├── data_loader.py       # load + validate raw CSV
│   ├── preprocessing.py     # cleaning, encoding, scaling, split
│   ├── feature_engineering.py
│   ├── models.py            # base learners + ensemble (voting/stacking) definitions
│   ├── train.py             # trains all models, saves best ensemble
│   ├── evaluate.py          # metrics, confusion matrix, ROC-AUC, SHAP-style importances
│   └── predict.py           # load saved model, predict on new data
├── api/
│   └── app.py                # FastAPI serving endpoint (for programmatic access)
├── models/                   # trained model artifacts (.pkl) saved here
├── notebooks/
│   └── churn_eda_and_modeling.ipynb   # ⭐ full EDA + training + evaluation, step by step
├── .streamlit/
│   └── config.toml           # UI theme for the Streamlit app
├── streamlit_app.py          # ⭐ the web UI — this is what you deploy
├── main.py                   # runs the full training pipeline end-to-end
├── requirements.txt
├── .gitignore
└── README.md
```

**Two ways to see how the model was built:**
- `main.py` — runs everything silently, script-style, prints a results table at the end.
- `notebooks/churn_eda_and_modeling.ipynb` — the same pipeline, but broken into
  cells with visualizations at every step: missing-value checks, churn
  distribution, churn rate by contract/tenure/charges, correlation heatmap,
  feature importance, ROC curves, confusion matrix. Open this if you want to
  **see and understand** the data and results, not just run a script.

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run the full pipeline

```bash
python main.py
```

This will:
1. Load `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. Clean + encode + engineer features → save to `data/processed/`
3. Train base learners: Logistic Regression, Random Forest, XGBoost, LightGBM
4. Build a **Stacking Ensemble** (meta-learner: Logistic Regression) and a
   **Soft Voting Ensemble**
5. Evaluate all models on a held-out test set (Accuracy, Precision, Recall,
   F1, ROC-AUC), print a comparison table
6. Save the best-performing model to `models/churn_ensemble.pkl`

## 5. Try the web app locally

```bash
streamlit run streamlit_app.py
```
Opens a form in your browser — fill in a customer's details, click
**Predict Churn**, see the probability instantly.

## 6. Serve predictions via API (for developers)

```bash
uvicorn api.app:app --reload --port 8000
```
Then `POST /predict` with a JSON body of customer features → returns churn
probability + label. Use this if you want another app/script to call the
model programmatically. The Streamlit app in Section 5 is the one to share
with non-technical people.

## 7. Deploy it with a permanent public link (free)

This is how you get a link that opens the live app directly — the kind you'd
put in your LinkedIn post or GitHub README, so anyone can click and use it
without installing anything.

**One-time setup:**

1. **Push this project to GitHub** (make the repo public):
   ```bash
   cd churn-ensemble
   git init
   git add .
   git commit -m "Customer churn prediction ensemble app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   > The trained model files in `models/` (~13 MB) are committed too, so the
   > deployed app works instantly — it does NOT need to retrain on startup.

2. **Deploy on Streamlit Community Cloud** (free, made for exactly this):
   - Go to **https://share.streamlit.io**
   - Sign in with your GitHub account
   - Click **"New app"**
   - Repository: `<your-username>/<repo-name>`, Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click **Deploy**
   - Wait ~1-2 minutes → you get a permanent URL like:
     `https://your-app-name.streamlit.app`

3. **Share that link anywhere** — LinkedIn post, GitHub README badge, resume,
   portfolio site. One click takes anyone straight into the working app.

**Add a badge to the top of your GitHub README** so visitors see a big
"Open App" button:
```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
```

**Keeping it updated:** any time you `git push` a change to `main`, Streamlit
Cloud automatically redeploys the app within a minute or two — no manual
redeploy step needed.

## 8. Why ensembling here

Telco churn data is tabular, moderately imbalanced (~26.5% churn), with a mix
of categorical and numeric features and non-linear interactions (e.g. contract
type × tenure × monthly charges). No single model wins outright:
- **Logistic Regression** — strong linear baseline, interpretable
- **Random Forest** — captures non-linear interactions, robust to outliers
- **XGBoost / LightGBM** — gradient boosting, typically best raw performance
  on this dataset, handles class imbalance well via `scale_pos_weight`

Stacking/voting these together usually nets a **1-3% ROC-AUC lift** over the
best single model, and reduces variance from any one model overfitting to
particular feature interactions.
