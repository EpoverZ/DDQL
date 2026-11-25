# DQDLz Linting Rules & Checks

This document lists every rule and check executed by the DQDLz engine, along with its category, machine-readable code, natural language explanation, and exact thresholds as defined directly in `core/checks.py`.

---

## 1. Data Quality Checks

### Missing Values Exceeded
- **Code**: `missing_values_threshold_exceeded`
- **Category**: `data_quality`
- **Level**: `warn`
- **Threshold**: `threshold = 30.0` (percentage)
- **Explanation**: Flags any individual column where the proportion of missing (`NaN`/`None`) values exceeds 30.0% of total rows.
- **Formula**: `(df[col].isna().sum() / len(df)) * 100 > threshold`

### Duplicate Rows Detected
- **Code**: `duplicate_rows_detected`
- **Category**: `data_quality`
- **Level**: `warn`
- **Threshold**: `threshold = 5.0` (percentage)
- **Explanation**: Flags datasets where exact duplicate rows account for more than 5.0% of total samples.
- **Formula**: `(df.duplicated().sum() / len(df)) * 100 > threshold`

### Constant Column
- **Code**: `constant_column`
- **Category**: `data_quality`
- **Level**: `info`
- **Threshold**: `unique_count == 1`
- **Explanation**: Identifies zero-variance columns containing only a single unique value across all rows.
- **Formula**: `df[col].nunique() == 1`

### Small Dataset
- **Code**: `small_dataset`
- **Category**: `data_quality`
- **Level**: `info`
- **Threshold**: `len(df) < 100`
- **Explanation**: Flags datasets with fewer than 100 rows, indicating statistical instability for model training.

### Features Exceed Samples
- **Code**: `features_exceed_samples`
- **Category**: `data_quality`
- **Level**: `warn`
- **Threshold**: `cols > rows`
- **Explanation**: Identifies the "curse of dimensionality" where the number of features exceeds the number of observations (`df.shape[1] > df.shape[0]`).

---

## 2. Target & Label Checks

### Label Dominance
- **Code**: `label_dominance`
- **Category**: `target`
- **Level**: `warn`
- **Threshold**: `dominance_threshold = 90.0` (percentage)
- **Explanation**: Flags classification targets where a single majority class accounts for more than 90.0% of total samples.
- **Formula**: `(class_count / total_samples) * 100 > 90.0`

### Rare Class
- **Code**: `rare_class`
- **Category**: `target`
- **Level**: `warn`
- **Threshold**: `rare_count_threshold = 20` (sample count)
- **Explanation**: Flags any target class label that has strictly fewer than 20 observations, posing extreme few-shot risk during cross-validation.
- **Formula**: `class_count < 20`

---

## 3. Data Leakage Checks

### High Target Correlation
- **Code**: `high_target_correlation`
- **Category**: `leakage`
- **Level**: `warn`
- **Threshold**: `corr_threshold = 0.9` (Pearson correlation)
- **Explanation**: Flags numeric features with an absolute linear correlation exceeding 0.90 with a numeric target variable.
- **Formula**: `abs(X[col].corr(y)) > 0.9`

### ID-Like Column
- **Code**: `id_like_column`
- **Category**: `leakage`
- **Level**: `warn`
- **Threshold**: `id_ratio = 0.95` OR name keyword in `['id', 'uuid', 'index']`
- **Explanation**: Identifies features where distinct values exceed 95.0% of total rows or the feature name contains identifier keywords.
- **Formula**: `(nunique / len(df) > 0.95) or any(k in col.lower() for k in ['id', 'uuid', 'index'])`

### Strong Leakage Risk
- **Code**: `strong_leakage_risk`
- **Category**: `leakage`
- **Level**: `warn`
- **Threshold**: Intersection of `high_corr` and `id_like`
- **Explanation**: Flags critical leakage when a feature simultaneously matches ID-like criteria and exhibits high correlation with the target.

---

## 4. Bias & Fairness Signals

### Under-represented Demographic Group
- **Code**: `group_underrepresented`
- **Category**: `bias`
- **Level**: `warn`
- **Threshold**: `min_group_pct = 10.0` (percentage)
- **Keywords**: `['gender', 'sex', 'race', 'age', 'region', 'income']`
- **Explanation**: Evaluates non-numeric features matching demographic keywords and flags any categorical group whose representation is below 10.0%.
- **Formula**: `(group_count / total_samples) * 100 < 10.0`

---

## 5. Trainability Checks

### Unhandled Missing Values
- **Code**: `missing_values_unhandled`
- **Category**: `trainability`
- **Level**: `info`
- **Threshold**: `total_missing > 0`
- **Explanation**: Signals that the dataset contains missing values that require imputation or dropping before scikit-learn/PyTorch estimators can accept it.
- **Formula**: `df.isna().sum().sum() > 0`

### No Numeric Features
- **Code**: `no_numeric_features`
- **Category**: `trainability`
- **Level**: `warn`
- **Threshold**: `len(numeric_columns) == 0`
- **Explanation**: Flags datasets with zero numeric columns, indicating encoding or feature engineering is required.
