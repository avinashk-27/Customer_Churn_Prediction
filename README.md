<h1>Overview</h1>

This project builds a machine learning pipeline to identify customers at risk of churning based on historical behavioral, transactional, and demographic data. The goal is to give the retention/marketing team a ranked list of at-risk customers along with the key drivers behind each prediction.

<h3> Five classifiers are trained and tuned via grid search, evaluated on held-out test accuracy, and the best-performing model is serialized to disk for reuse.</h3>


| Model | Best Params | Accuracy |

| Logistic Regression | (default) | 0.87 |

| K-Nearest Neighbors | `n_neighbors=12`, `weights=distance` | 0.86 |

| **SVM (best model)** | `C=0.01`, `kernel=linear` | **0.875** |

| Decision Tree | `criterion=entropy`, `max_depth=20`, `min_samples_leaf=4`, `min_samples_split=2`, `splitter=random` | 0.85 |

| Random Forest | `bootstrap=True`, `max_features=2`, `n_estimators=256` | 0.87 |

The tuned **SVM** (linear kernel, `C=0.01`) had the highest test accuracy and is the model saved as `model.pkl`.

<h3>Pipeline Details</h3>

1. Load & Inspect

 Reads the CSV, checks shape/structure with .head(), .tail(), .info(), .describe(), .isna().sum()

2. Clean

  Fills missing InternetService values with an empty string

3. Explore (EDA)

Correlation matrix across numeric columns

Churn distribution (pie chart)

Average MonthlyCharges by Churn, and by Churn + Gender

Average Tenure and Age by Churn

Average MonthlyCharges by ContractType (bar chart)

Distribution histograms for MonthlyCharges and Tenure


4. Prepare Features

Selects Age, Gender, Tenure, MonthlyCharges as features (x); Churn as target (y)

Encodes Gender (1 = Female, 0 = Other) and Churn (1 = Yes, 0 = No) as binary

80/20 train/test split via train_test_split

Scales features with StandardScaler (fit on train, applied to both — see note below); scaler saved to scaler.pkl


5. Train & Tune Models Each model (except Logistic Regression) is tuned with GridSearchCV (5-fold CV), then scored on test accuracy via a shared modelperformance() helper:

Logistic Regression

K-Nearest Neighbors (n_neighbors, weights)

SVM (C, kernel)

Decision Tree (criterion, splitter, max_depth, min_samples_split, min_samples_leaf)

Random Forest (n_estimators, max_features, bootstrap)


6. Save Best Model

The tuned SVM (gridsvc.best_estimator_) is saved as model.pkl via joblib

<h3>Website Link</h3>

https://customerchurnprediction-vz53hmappppamd7qgbchfzym.streamlit.app/
