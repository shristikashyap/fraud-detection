# A Deep Learning-Based Context-Aware Fraud Detection System Using Behavioral Analysis

This repository is being developed in phases. Phase 1 prepares the public
transaction data. Phase 2 trains and evaluates an initial classifier for
cold-start transactions. Phase 3 adds synthetic user behavior and trusted
behavioral baselines. Phase 4 adds an experimental LSTM sequence signal.
Phase 5 adds a separate experimental sequence autoencoder. Phase 6 combines
available classifier and behavioral signals into an explainable triage score.
Phase 7 exposes the models through Flask, and Phase 8 provides a React
dashboard for monitoring API results.

## Project layout

- `backend/`: Flask service and the Python data pipeline.
- `frontend/`: React/Vite dashboard and API client.
- `backend/data/raw/` and `backend/data/processed/`: local data files; ignored by Git.
- `backend/data/behavioral/`: generated synthetic behavioral events; ignored by Git.

## Phase 1 setup

The workspace Python environment is `.venv`. Install the Phase 1 backend
dependencies and the frontend dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\backend[dev]
Set-Location .\frontend
npm install
Set-Location ..
```

Download and prepare the public OpenML Credit Card Fraud dataset:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.data.prepare_dataset --download
```

The pipeline writes chronological train, validation, and test Parquet files
under `backend/data/processed/` and a JSON preparation report. The source data
is approximately 70 MB and is downloaded only when this command is run.
Prepared data is not committed to the repository.

To run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests
```

## Phase 2: Initial fraud classifier

Train Logistic Regression and Random Forest candidates, tune each decision
threshold on validation data only, then select using validation PR-AUC,
followed by F1, recall, and precision as tie-breakers. Evaluate the selected
model once on the held-out test split:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.classification.training
```

The fitted pipeline is saved to
`backend/models/initial_fraud_classifier.joblib`; actual train, validation,
candidate comparison, final test metrics, confusion matrices, and
classification reports are written to
`backend/reports/phase2_evaluation.json`. The model file is local/ignored;
the JSON report is retained. The test split is not read for candidate or
threshold selection. No oversampling is used; candidates use training-only
class weighting. Logistic Regression and Random Forest are always compared.
XGBoost is optional and is included only when installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.\backend[xgboost]'
```

Use the saved cold-start model from Python:

```python
from pathlib import Path
from fraud_detection.classification.prediction import load_predictor

predictor = load_predictor(
    Path("backend/models/initial_fraud_classifier.joblib")
)
result = predictor.predict({"Time": 1000, "Amount": 42.50, "V1": 0.1})
```

Prediction input uses the dataset's `Time`, `Amount`, and any available
`V1`–`V28` fields. Missing PCA components are imputed using values learned
from the training split. The model probability threshold is selected from
validation to maximize F1 (precision breaks ties). Risk categories use
configurable `medium_risk_fraction` on `FraudPredictor`; by default the low /
medium boundary is half the selected fraud threshold, and high risk begins at
that threshold. Weighted classifiers may not produce calibrated real-world
probabilities; the output is a model score interpreted as a probability, not a
guaranteed likelihood.

## Phase 3: Behavioral learning and baselines

Generate a deterministic, explicitly synthetic behavioral dataset:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.behavioral.data
```

By default, the generator creates 21 days of fictional activities for six
users, plus a labeled multiple-anomaly demonstration per user. CSV and Parquet
output are supported with `--output`. Rows include the requested user,
timestamp, login, transaction, device, location, activity, and status fields,
plus `is_synthetic`, `scenario`, and `synthetic_suspicious_label` metadata.
These fictional values are for exercising the software only; they are not
observations, a training corpus for production behavior, or performance
evidence. The analyzer intentionally ignores scenario and synthetic labels.

`fraud_detection.behavioral.BehavioralLearningService` accepts the existing
Phase 2 `FraudPredictor` (load it with `load_predictor`) and a configurable
`BaselineConfig`. A user's state is NEW USER with no trusted activity, LEARNING
USER until both the configured minimum trusted event count and elapsed-history
duration are met, and ESTABLISHED USER after both conditions are met. Login
time tolerance, frequency deviation, robust amount tolerance, sequence support,
and the number of deviations that request additional verification are
configurable.

Profiles are updated only when the caller explicitly supplies `verified=True`,
the event has no detected behavioral deviation, and the Phase 2 classifier
returns LOW RISK for a transaction. The application must only set `verified`
after its own trusted review/authorization; this library does not establish
identity or transaction truth. Deviations are explanatory signals, not fraud
predictions. A single deviation does not automatically request verification;
the configurable multiple-signal threshold controls that advisory. For
transaction-less login events, the initial transaction classifier is reported
as not applicable; explicitly verified login events can still contribute to
login-time and activity-sequence baselines.

The bridge maps wall-clock event timestamps to seconds since UTC midnight for
the Phase 2 classifier's time input, rather than passing Unix-epoch values.
This only approximates the original dataset's elapsed-seconds feature and is a
distribution mismatch; Phase 2 model probabilities on behavioral events should
be treated as advisory until validated on representative transaction data.

Profiles can be persisted with `BaselineStore.save(path)` and restored with
`BaselineStore.load(path)`. The service response includes the baseline summary,
anomaly codes and observed/expected values, sequence context, initial-classifier
output, readiness state, and optional sequential-model output.

## Phase 4: LSTM sequential behavior analysis

Install the optional TensorFlow dependency and train the separate LSTM model:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.\backend[tensorflow]'
.\.venv\Scripts\python.exe -m fraud_detection.behavioral.sequential.training
```

The trainer reads the Phase 3 synthetic activity CSV but does not modify it.
It sorts events per user, creates causal, left-padded sequence windows, and
holds out complete users (the current six-user dataset yields four train, one
validation, and one test user). Scaling statistics and category vocabularies
are fit only from training-user events. Inputs contain transaction amount and
amount change, elapsed interval, cyclic login hour, transaction frequency,
device/location changes and novelty, transaction/activity type, and activity
transition. User IDs, device/location identifiers, scenario names, synthetic
metadata, transaction status, and suspicious labels are not model inputs.
Synthetic scenario labels are used only as supervised training/evaluation
targets.

Training saves a separate Keras model to
`backend/models/behavioral_lstm.keras`, preprocessing configuration to
`backend/models/behavioral_lstm_preprocessing.json`, and actual split metrics
to `backend/reports/phase4_lstm_evaluation.json`. The model, preprocessing
configuration, and generated activity file are local artifacts ignored by
Git. The report lists the held-out users, label counts, per-split metrics, and
the evaluation limitations. With only six fictional users and one suspicious
scenario per user, these metrics are unstable demonstrations and provide no
evidence of real-world fraud performance.

Load the saved sequence analyzer:

```python
from pathlib import Path
from fraud_detection.behavioral.sequential.prediction import load_sequence_predictor

lstm = load_sequence_predictor(
    Path("backend/models/behavioral_lstm.keras"),
    Path("backend/models/behavioral_lstm_preprocessing.json"),
)
sequence_result = lstm.predict(chronological_user_activities)
```

For service integration, pass the analyzer as `lstm_analyzer` to
`BehavioralLearningService` and pass preceding chronological events using
`recent_activities` when assessing an established user. The service appends
the current activity and returns `lstm_sequence_analysis` separately from
`initial_fraud_classifier`; it does not let the sequence score replace or
change the classifier result or make a fraud decision. The LSTM returns an
advisory synthetic-sequence anomaly score and requires sufficient context.
Do not treat its signal as confirmed fraud.

## Phase 5: Autoencoder anomaly detection

Train a separate sequence autoencoder on normal activities from training
users:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.behavioral.autoencoder.training
```

The autoencoder reuses Phase 4's causal sequence features, user-aware split,
and training-fitted categorical/numeric preprocessing. Its model and
preprocessor are fit using **normal training-user events/windows only**.
Early stopping and the default 95th-percentile reconstruction-error threshold
use normal validation-user windows only. Test-user events and synthetic labels
are used only for the final held-out evaluation. The quantile is configurable
with `--validation-quantile`.

The model is saved separately as
`backend/models/behavioral_autoencoder.keras`; its feature preprocessing and
threshold are saved in
`backend/models/behavioral_autoencoder_preprocessing.json`; actual metrics and
threshold facts are saved in
`backend/reports/phase5_autoencoder_evaluation.json`. These files do not
overwrite the Random Forest or LSTM artifacts.

Load the autoencoder signal:

```python
from pathlib import Path
from fraud_detection.behavioral.autoencoder.prediction import (
    load_autoencoder_predictor,
)

autoencoder = load_autoencoder_predictor(
    Path("backend/models/behavioral_autoencoder.keras"),
    Path("backend/models/behavioral_autoencoder_preprocessing.json"),
)
result = autoencoder.predict(chronological_user_activities)
```

To expose it through the Phase 3 service, pass it as `autoencoder_analyzer`
alongside the existing `lstm_analyzer`. The response contains a separate
`autoencoder_analysis` object with reconstruction error, threshold, and an
advisory behavioral-anomaly flag. It never replaces the initial classifier,
changes the LSTM result, or decides fraud. This phase does not combine signals
or change the contextual baseline's review behavior.

Evaluation targets are the synthetic suspicious-scenario labels, not verified
fraud labels. The six-user dataset has one suspicious scenario per user, so
the held-out test contains just one positive example. Classification and
ranking metrics from that test are therefore highly unstable and cannot show
real-world fraud-detection performance. The normal-validation quantile is not
a guaranteed future false-positive rate; this model must be retrained and
validated on representative, consented real activity before operational use.

## Phase 6: Combined context-aware risk assessment

`fraud_detection.behavioral.CombinedRiskAssessor` consumes the structured
result from `BehavioralLearningService` and reports the initial classifier,
contextual baseline, LSTM, and autoencoder signals separately. It is a
standalone backend module; this phase does not change the Flask API or React
dashboard.

Each available signal is normalized independently: the classifier and LSTM
scores are mapped relative to their own decision/anomaly thresholds using
`score / (score + threshold)` (and remain explicitly uncalibrated advisory
scores), reconstruction error is mapped as
`error / (error + validation_threshold)`, and contextual deviations are
scaled by a configurable saturation count. Signals without the threshold
needed for normalization are excluded rather than combined as raw values.
Configurable default weights are
0.45, 0.20, 0.20, and 0.15 respectively. Weights are renormalized over only
available, positively weighted signals. An unavailable or inactive model is
reported with an exclusion reason and contributes neither risk nor low-risk
evidence. If no usable signal exists, the assessment has no score/category
and recommends manual review instead of assigning LOW RISK.

The default score boundaries are LOW RISK below 30, MEDIUM RISK from 30, and
HIGH RISK from 70. MEDIUM and HIGH risk recommend additional verification;
the Phase 3 baseline's own verification request is also preserved. These
weights and thresholds are configurable policy defaults, not learned or
empirically calibrated values. The combined 0–100 score is a triage score,
not a fraud probability, and the assessor never confirms fraud. Individual
normalized scores, configured and normalized weights, contribution points,
explanations, unavailable reasons, and limitations appear in the result.

New and learning users retain the Phase 2 classifier signal while personalized
context, LSTM, or autoencoder signals are excluded unless their outputs are
actually available. An anomaly is a behavioral deviation, not proof of fraud.
Phase 4 and 5 models were trained/evaluated on a six-user synthetic dataset;
their signals and the combined defaults are not validated on representative
real-world behavior and must not be used as operationally calibrated risk.
Use the module independently:

```python
from fraud_detection.behavioral import CombinedRiskAssessor

assessment = behavioral_service.assess(activity, recent_activities=history)
risk = CombinedRiskAssessor().assess(assessment)
```

## Phase 7: Flask REST API

The Flask application loads the Phase 2 classifier and, independently, the
Phase 4 LSTM and Phase 5 autoencoder from fixed filenames in
`backend/models/`. Each model's preprocessing configuration is loaded by its
existing predictor. Missing or invalid artifacts are reported by `/api/health`;
the API does not retrain models or accept artifact paths from requests.

Start the API from the workspace root:

```powershell
.\.venv\Scripts\python.exe -m flask --app fraud_detection.api.app:create_app run
```

The default CORS allowlist permits `http://localhost:5173` and
`http://127.0.0.1:5173`. Embedding applications may configure
`CORS_ALLOWED_ORIGINS` when calling `create_app`; credentials are not enabled.
Configure `MODEL_DIR` and `BASELINE_STORE_PATH` through server-side Flask
configuration only, never through an API request.

Endpoints:

- `GET /api/health`: API status and per-model availability without filesystem paths.
- `POST /api/analyze`: analyze one activity and optional preceding events.
- `GET /api/users`: list trusted behavioral profile summaries.
- `GET /api/users/<user_id>`: return one minimized profile summary.
- `GET /api/transactions?page=1&per_page=20`: paginated transaction/activity history.
- `GET /api/alerts?page=1&per_page=50`: paginated assessments requiring review (`limit` remains a supported alias for `per_page`).
- `GET /api/reports`: summarize persistent assessments.

Example request:

```json
{
  "activity": {
    "user_id": "user-001",
    "timestamp": "2026-01-08T03:00:00+00:00",
    "login_time": "03:00",
    "transaction_amount": 125.5,
    "transaction_frequency": 2,
    "device_type": "mobile",
    "device_id": "device-17",
    "location": "Boston",
    "transaction_type": "online",
    "activity_type": "login_and_transaction",
    "transaction_status": "completed"
  },
  "recent_activities": []
}
```

`activity` requires a user ID, timezone-aware ISO-8601 timestamp, and activity
type. Transaction, device, and location fields are validated when supplied.
Optional `recent_activities` must belong to the same user, be chronological,
and precede the current event. The API rejects untrusted labels and any
`verified` flag. Incoming events and client-supplied sequence context are used
only for this assessment: the API always calls the behavioral service with
`verified=False` and never writes them into trusted profiles.

Trusted profiles are loaded from
MongoDB at startup. If the legacy server-owned
`backend/data/behavioral/trusted_profiles.json` exists, profiles absent from
MongoDB are imported without replacing any existing MongoDB profile. The API
provides no automatic trust or profile-update endpoint; analyzed activities
remain `verified=False`. User endpoints omit raw transaction histories,
device IDs, and location values. Paginated transaction history keeps deviation
reasons but omits raw device and location values. The API has no authentication
and is intended for local/demo use, not exposure to an untrusted network. A
missing usable model signal produces an unavailable risk score rather than a
fabricated low-risk result. Risk remains triage guidance, not confirmed fraud.

## Phase 9: MongoDB persistence

The backend uses one reusable PyMongo client and the database selected by
`MONGO_DB_NAME` (default: `fraud_detection`). Configure the Atlas URI in the
backend environment before starting Flask. Example PowerShell configuration:

```powershell
$env:MONGO_URI = "[MongoDB connection string redacted]
$env:MONGO_DB_NAME = "fraud_detection"
.\.venv\Scripts\python.exe -m flask --app fraud_detection.api.app:create_app run
```

In MongoDB Atlas, create a database user with only the required database
read/write privileges and allow the development machine in Network Access.
Never commit the URI or put it in a `VITE_*` variable; Vite variables are
browser-visible. The root `.gitignore` excludes `.env` and `.env.*` files.
The Flask process reads environment variables directly; if using a local
`.env` file, export its values in the shell or use a trusted environment
loader outside source control.

When MongoDB is not configured or cannot be reached, Flask remains available
in degraded mode and `/api/health` reports `backend_available` and a
credential-free `database` status separately from model availability. The
classifier may still produce an analysis, but the response explicitly reports
`persistence.status: unavailable`; behavioral profile readiness is also marked
unavailable if profiles could not be loaded. Database-backed list/report
endpoints return HTTP 503 rather than silently presenting empty persistent
history. Health never returns `MONGO_URI`.

Collections and their primary stored content:

- `transactions`: request/transaction identifiers, UTC event and creation
  timestamps, activity fields, status, risk score/category, and the complete
  API analysis response.
- `risk_assessments`: classifier, contextual, LSTM, and autoencoder signals,
  combined score/category, explanations, recommendation, and UTC creation time.
- `alerts`: alert and transaction identifiers, risk details, explanation,
  recommendation, UTC creation time, and open status.
- `behavioral_profiles`: trusted baseline aggregates, `user_status`,
  `last_updated`, and ordered history values used to rebuild the baseline.

Unique and query indexes are created for transaction IDs/idempotency keys,
user/time transaction history, risk-assessment transaction IDs and dates,
alert IDs/status/dates, and profile user IDs. Timestamps are persisted as UTC
BSON dates. A repeated request with the same `Idempotency-Key` and payload
returns the original analysis; reusing that key with a different payload
returns HTTP 409. The dashboard sends an idempotency key for each analysis.
The API does not accept a client-supplied `verified` flag. Profile writes only
occur through the existing explicit trusted-learning path, which continues to
require `verified=True`, no contextual deviations, and an allowed classifier
result; compare-and-replace checks the previously loaded activity count before
updating a stored profile.

The tests use isolated `mongomock` databases; they do not connect to or alter
production Atlas data. Transaction and alert routes accept `page` and
`per_page` (1–100), and return total counts and page counts. The React
dashboard reads these paginated routes so history and alerts persist across
browser and backend restarts.

To build the frontend:

```powershell
Set-Location .\frontend
npm run build
```

For local development, run the Flask API and Vite dashboard in separate
terminals from the workspace root. The Vite proxy forwards `/api` requests to
the local Flask service, so no frontend environment variable is needed for
the default setup:

```powershell
.\.venv\Scripts\python.exe -m flask --app fraud_detection.api.app:create_app run
```

```powershell
Set-Location .\frontend
npm run dev
```

Open the Vite URL printed in the frontend terminal (normally
`http://localhost:5173`). To point the dashboard directly at a different API,
set `VITE_API_BASE_URL` before starting Vite; the client appends `/api` unless
the value already ends with `/api`:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:5000"
npm run dev
```

The dashboard includes Overview, Transaction Monitoring, Behaviour Analysis,
Fraud Detection, Risk Analysis, Alerts, and Reports views. It displays
classifier and anomaly-model availability separately, shows explanations and
signal contributions returned by the API, and leaves absent scores
unavailable rather than treating them as zero. Submitted activity and optional
prior sequence events are inference-only; the dashboard does not mark them as
trusted or update behavioral baselines.

Profile views show only the minimized summaries exposed by the API. MongoDB
transactions, reports, and alerts persist, but the demo API still has no
authentication or authorization layer; do not expose it to an untrusted
network or store production financial/identity data in this capstone setup.

## Dataset and limitations

Phase 1 uses OpenML dataset 1597, `creditcard`, whose public record describes
284,807 European card transactions from September 2013, including 492 frauds.
It provides `Time`, `Amount`, PCA-transformed `V1`–`V28`, and the binary `Class`
label. It does **not** contain customer identifiers, device, geographic
location, login, or other user-activity history. This dataset can support
initial transaction classification research, but cannot train or validate
personalized behavioral models by itself. Those later phases will require a
separate suitable behavioral dataset or consented, appropriately protected
application activity data.

Dataset record and citation:

- OpenML: https://www.openml.org/d/1597
- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi,
  “Calibrating Probability with Undersampling for Unbalanced Classification,”
  IEEE Symposium on Computational Intelligence and Data Mining, 2015.

The source dataset is highly imbalanced. Model evaluation preserves the
chronological holdout and reports class-sensitive metrics; accuracy alone
would be misleading. The Phase 1 preparation report contains dataset/split
facts only; Phase 2 metrics are in the separate evaluation report.

The cyclic time features are derived from elapsed seconds modulo one day. They
are not local clock-time observations and must not be interpreted as learned
login-time or user behavior.

## Planned phases

1. Dataset ingestion, preprocessing, and feature engineering (complete).
2. Initial fraud classifier and evaluation on train/validation/test data (complete).
3. User activity collection and trusted-activity behavioral baselines (complete).
4. LSTM sequential behavior model (complete).
5. Autoencoder anomaly model (complete).
6. Configurable, explainable combined risk assessment and cold-start handling (complete).
7. Flask inference/API integration (complete).
8. React dashboard integration with the Phase 7 API (complete; history is database-backed).
9. MongoDB persistence for analyses, alerts, reports, and trusted profiles (complete).

TensorFlow is an optional dependency, installed for Phase 4 and Phase 5 model
training and inference.
