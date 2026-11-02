# Recruitment Matching using Machine Learning

A machine learning study for predicting whether a **recruitment–applicant pair** is matched (`TARGET = 1`) or unmatched (`TARGET = 0`). The project compares optimized numerical classifiers with models enhanced by multilingual semantic-similarity features derived from Japanese text.

> **Scope:** Experimental analysis on a **synthetic dataset**, not a production hiring system or a substitute for human recruitment decisions.

## Project overview

The study covers two tasks:

- **Task 1 — Numerical matching:** Train and optimize models using **10 numerical matching attributes**.
- **Task 2 — Numerical + semantic matching:** Add **6 engineered semantic-similarity features** from Japanese-language text, for **16 total model inputs**.

Four classifiers are evaluated: **Logistic Regression, Random Forest, Gradient Boosting, and LightGBM**. The primary metric is **positive-class F1-score**, with precision, recall, and ROC-AUC used for additional evaluation.

## Dataset

The supplied synthetic dataset has **3,000 recruitment–applicant pairs**:

| Target | Meaning | Records | Share |
|---|---|---:|---:|
| `0` | Unmatched | 2,850 | 95% |
| `1` | Matched | 150 | 5% |

Because the classes are highly imbalanced, accuracy alone is misleading: an always-unmatched classifier would achieve 95% accuracy while identifying no matches.

### Task 1: Numerical features

1. Communication skill score
2. Applicant required-criteria match ratio
3. Job/work-history Jaccard score
4. Job-history count (last 3 years)
5. Job experience match score
6. Job required-criteria match ratio
7. Job-experience BM25 score
8. Total job-history count
9. Job changes per career year
10. Skill match score

Identifiers, raw text columns, and the target are excluded from Task 1 predictors.

### Task 2: Semantic features

A frozen pretrained **`paraphrase-multilingual-MiniLM-L12-v2`** encoder produces **384-dimensional embeddings** of three Japanese text fields: **Job**, **Choice Point**, and **Skills**. The encoder is **not fine-tuned**.

Three pairwise cosine similarities are computed:

- Job ↔ Skills
- Job ↔ Choice Point
- Choice Point ↔ Skills

Their **mean, maximum, and minimum** provide three further features. Only these **6 scalar similarities** are added to the 10 numerical inputs; the raw embedding vectors are not concatenated with tabular features.

## Methodology

1. Audit data types, labels, duplicates, and missing values.
2. Split data into training and held-out test partitions **before** training-dependent preprocessing or oversampling.
3. Use median imputation; standardize inputs for Logistic Regression only.
4. Address class imbalance with training-fold-only `RandomOverSampler` (none, 1:2, or 1:1 balancing, as applicable).
5. Optimize classifiers through cross-validation and hyperparameter search.
6. Select probability thresholds using **training-only out-of-fold predictions** to maximize positive-class F1.
7. Evaluate selected configurations on the held-out test set.

For the latest Task 1 experiment, optimization sampled **8 hyperparameter configurations × 3 balancing strategies × 5 folds = 120 fits per classifier**, or **480 fits** across the four algorithms.

### Important split distinction

The results below come from **different experimental partitions**:

| Experiment | Train | Test | Split strategy |
|---|---:|---:|---|
| Latest Task 1 | 2,400 | 600 | Stratified row-level split |
| Earlier Task 2 | 2,424 | 576 | `Recruitment_ID`-disjoint groups |

Their headline scores are **not a controlled head-to-head comparison**.

## Results

### Task 1 — Numerical features

| Classifier | Baseline F1 | Optimized F1 | Optimized precision | Optimized recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6374 | **0.8667** | 0.8667 | 0.8667 |
| Random Forest | 0.7500 | 0.8276 | 0.8571 | 0.8000 |
| Gradient Boosting | 0.7606 | 0.8308 | 0.7714 | 0.9000 |
| LightGBM | 0.7647 | 0.8000 | 0.8800 | 0.7333 |

The highest **observed** Task 1 test F1 was **86.67%** for optimized Logistic Regression, using 1:1 oversampling and a **0.92** decision threshold. Compared with its baseline, the number of false positives declined from **32 to 4**, while false negatives increased from **1 to 4**.

### Task 2 — Numerical + semantic features

These figures are from an **earlier executed grouped experiment**, not a new execution of the later Task 2 Colab notebook.

| Classifier | Test F1 | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.8077 | 0.9545 | 0.7000 |
| Random Forest | 0.8333 | 0.8333 | 0.8333 |
| Gradient Boosting | 0.7925 | 0.9130 | 0.7000 |
| LightGBM | **0.8364** | 0.9200 | 0.7667 |

The highest observed Task 2 F1 was **83.64%** for LightGBM, with a 1:2 positive-to-negative training oversampling ratio and **0.73** decision threshold. It detected **23 of 30** true matches, with **2 false positives** and **7 false negatives**.

### Paired comparison on the same grouped holdout

The earlier experiment also evaluated both feature sets on the **same 576-record grouped test partition**:

| Classifier | Numerical F1 | Numerical + semantic F1 |
|---|---:|---:|
| Logistic Regression | 0.8077 | 0.8077 |
| Random Forest | **0.8462** | 0.8333 |
| Gradient Boosting | 0.8077 | 0.7925 |
| LightGBM | 0.8364 | **0.8364** |

**Finding:** Adding six semantic features did **not consistently improve F1** across the four classifiers on this synthetic dataset. Further unified evaluation is needed before drawing general conclusions.

## Limitations

- **Synthetic data:** The reported performance does not establish real-world recruitment accuracy or fairness.
- **Only 30 positive cases per test set:** Small changes in classification outcomes can significantly affect F1.
- **Possible target construction overlap:** Engineered match scores may correlate with the synthetic-label generation process; this needs verification.
- **Identity overlap:** The latest Task 1 row split can share recruitment/applicant identities across partitions. The earlier grouped experiment separates recruitment IDs but may still share applicant IDs.
- **Selection uncertainty:** Reported highest test scores should be interpreted as exploratory, not as confirmed performance of a preselected model on new data.

## Future work

- Run a **unified group-disjoint benchmark** for both feature sets.
- Check the synthetic target-generation procedure for potential leakage.
- Validate with independent, representative real-world data and previously unseen applicant/job identities.
- Evaluate model calibration, error trade-offs, robustness, and fairness before any operational use.

## Tools and techniques

Python; scikit-learn classifiers and preprocessing; LightGBM; imbalanced-learn `RandomOverSampler`; sentence-transformer multilingual MiniLM embeddings; cosine similarity; cross-validation; hyperparameter and threshold optimization.

## Reproducing the analysis

This README documents the methodology and results in the accompanying technical report. **The report does not specify a verified repository structure, notebook filenames, dependency lockfile, or executable commands.** Once the notebooks and data are committed, add their actual paths and environment setup instructions here. Do not commit private applicant data or credentials.

## Reference

*Recruitment Matching Analysis: Optimized Numerical and Semantic Machine Learning* — Internship Technical Report, Tasks 1 and 2.

## Responsible use

This repository is intended for research and learning. Any practical recruitment use would require external validation, privacy protections, fairness assessment, and meaningful human oversight.
