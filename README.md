Overview

This project builds a machine learning pipeline to identify customers at risk of churning based on historical behavioral, transactional, and demographic data. The goal is to give the retention/marketing team a ranked list of at-risk customers along with the key drivers behind each prediction.

Five classifiers are trained and tuned via grid search, evaluated on held-out test accuracy, and the best-performing model is serialized to disk for reuse.

| Model | Best Params | Accuracy |

| Logistic Regression | (default) | 0.87 |

| K-Nearest Neighbors | `n_neighbors=12`, `weights=distance` | 0.86 |

| **SVM (best model)** | `C=0.01`, `kernel=linear` | **0.875** |

| Decision Tree | `criterion=entropy`, `max_depth=20`, `min_samples_leaf=4`, `min_samples_split=2`, `splitter=random` | 0.85 |

| Random Forest | `bootstrap=True`, `max_features=2`, `n_estimators=256` | 0.87 |

The tuned **SVM** (linear kernel, `C=0.01`) had the highest test accuracy and is the model saved as `model.pkl`.
