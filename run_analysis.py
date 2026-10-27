"""
CA3 Mini Project: Healthcare Analytics
Title: Identifying Diabetes Risk Factors
Python implementation reproducing all analytics and visualizations from the R notebook.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})
os.makedirs("outputs", exist_ok=True)

# ==============================================================================
# CELL 1: Load Dataset & Initial Inspection
# ==============================================================================
print("=" * 70)
print("CELL 1: DATASET INSPECTION")
print("=" * 70)

url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
diabetes = pd.read_csv(url)

print("First 6 Records:")
print(diabetes.head(6))

print("\nDataset Dimensions:")
print(f"[{diabetes.shape[0]}, {diabetes.shape[1]}]")

print("\nColumn Names:")
print(list(diabetes.columns))

print("\nDataset Structure:")
print(diabetes.info())

print("\nSummary Statistics:")
print(diabetes.describe().T[['min', '25%', '50%', 'mean', '75%', 'max']])

# ==============================================================================
# CELL 2: Missing Values & Data Quality Check
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 2: DATA QUALITY & OUTCOME DISTRIBUTION")
print("=" * 70)

print("Missing Values in Each Column (NaN):")
print(diabetes.isna().sum())

medical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero Values in Medical Columns (Clinically Invalid):")
zero_counts = {col: (diabetes[col] == 0).sum() for col in medical_columns}
for col, count in zero_counts.items():
    print(f"  {col}: {count}")

print("\nDiabetes Outcome Distribution:")
outcome_counts = diabetes['Outcome'].value_counts().sort_index()
print(outcome_counts)

print("\nOutcome Percentages:")
print((diabetes['Outcome'].value_counts(normalize=True).sort_index() * 100).round(2))

# ==============================================================================
# CELL 3: Data Cleaning & Median Imputation
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 3: DATA CLEANING (MEDIAN IMPUTATION)")
print("=" * 70)

diabetes_clean = diabetes.copy()

# Replace invalid zero measurements with NaN
for col in medical_columns:
    diabetes_clean.loc[diabetes_clean[col] == 0, col] = np.nan

print("Missing values after zero-value correction:")
print(diabetes_clean.isna().sum())

# Replace missing measurements with each column's median
for col in medical_columns:
    col_median = diabetes_clean[col].median()
    diabetes_clean[col] = diabetes_clean[col].fillna(col_median)

print("\nMissing values after cleaning:")
print(diabetes_clean.isna().sum())

print(f"\nCleaned dataset dimensions: {diabetes_clean.shape}")

print("\nSummary of cleaned medical measurements:")
print(diabetes_clean[medical_columns].describe().T[['min', '25%', '50%', 'mean', '75%', 'max']].round(2))

# Convert outcome to readable labels
diabetes_clean['Outcome_Label'] = diabetes_clean['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# ==============================================================================
# CELL 4: Exploratory Data Visualizations
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Diabetes outcome distribution
sns.countplot(x='Outcome_Label', data=diabetes_clean, ax=axes[0, 0], palette=['#4C72B0', '#DD8452'], hue='Outcome_Label', legend=False)
axes[0, 0].set_title("Distribution of Diabetes Outcomes", fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel("Patient Group")
axes[0, 0].set_ylabel("Number of Patients")

# 2. Average glucose level by outcome
sns.barplot(x='Outcome_Label', y='Glucose', data=diabetes_clean, ax=axes[0, 1], palette=['#4C72B0', '#DD8452'], errorbar=None, hue='Outcome_Label', legend=False)
axes[0, 1].set_title("Average Glucose Level by Diabetes Outcome", fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel("Patient Group")
axes[0, 1].set_ylabel("Average Glucose Level")

# 3. BMI distribution by outcome
sns.boxplot(x='Outcome_Label', y='BMI', data=diabetes_clean, ax=axes[1, 0], palette=['#4C72B0', '#DD8452'], hue='Outcome_Label', legend=False)
axes[1, 0].set_title("BMI Comparison by Diabetes Outcome", fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel("Patient Group")
axes[1, 0].set_ylabel("Body Mass Index (BMI)")

# 4. Age distribution by outcome
sns.histplot(data=diabetes_clean, x='Age', hue='Outcome_Label', bins=20, alpha=0.6, ax=axes[1, 1], palette=['#4C72B0', '#DD8452'])
axes[1, 1].set_title("Age Distribution by Diabetes Outcome", fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel("Age (years)")
axes[1, 1].set_ylabel("Number of Patients")

plt.tight_layout()
plot_path_4 = "outputs/01_exploratory_distributions.png"
plt.savefig(plot_path_4, dpi=300)
plt.close()
print(f"\n[Saved Plot]: {plot_path_4}")

# ==============================================================================
# CELL 5: Group Means & Correlation with Outcome
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 5: GROUP MEANS & FEATURE CORRELATIONS")
print("=" * 70)

print("AVERAGE MEDICAL MEASUREMENTS BY DIABETES OUTCOME:\n")
group_means = diabetes_clean.groupby('Outcome_Label')[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']].mean().round(2)
print(group_means)

print("\nCORRELATION WITH DIABETES OUTCOME:\n")
features = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

correlations = diabetes_clean[features].apply(lambda col: diabetes_clean['Outcome'].corr(col))
corr_df = pd.DataFrame({"Variable": features, "Correlation": correlations.round(3)})
corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index).reset_index(drop=True)
print(corr_df.to_string(index=False))

# Plot correlations with Outcome
plt.figure(figsize=(10, 6))
sns.barplot(data=corr_df, x='Correlation', y='Variable', hue='Correlation', palette='Blues_r', legend=False)
plt.title("Correlation of Patient Characteristics with Diabetes Outcome", fontsize=13, fontweight='bold')
plt.xlabel("Correlation Coefficient")
plt.ylabel("Patient Characteristic")
plt.tight_layout()
plot_path_5 = "outputs/02_correlation_with_outcome.png"
plt.savefig(plot_path_5, dpi=300)
plt.close()
print(f"[Saved Plot]: {plot_path_5}")

# ==============================================================================
# CELL 6: Correlation Heatmap
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 6: CORRELATION HEATMAP")
print("=" * 70)

num_cols = diabetes_clean.select_dtypes(include=[np.number]).drop(columns=['Outcome'], errors='ignore')
num_cols['Outcome'] = diabetes_clean['Outcome']
cor_matrix = num_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Heatmap of Diabetes Dataset", fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path_6 = "outputs/03_correlation_heatmap.png"
plt.savefig(plot_path_6, dpi=300)
plt.close()
print(f"[Saved Plot]: {plot_path_6}")

# ==============================================================================
# CELL 7 & 8: Hypothesis Testing (Welch's Two-Sample t-test on Glucose)
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 7 & 8: STATISTICAL HYPOTHESIS TESTING (WELCH'S T-TEST)")
print("=" * 70)

glucose_no_diabetes = diabetes_clean.loc[diabetes_clean['Outcome'] == 0, 'Glucose']
glucose_diabetes = diabetes_clean.loc[diabetes_clean['Outcome'] == 1, 'Glucose']

# Welch's t-test (equal_var=False)
ttest_res = stats.ttest_ind(glucose_no_diabetes, glucose_diabetes, equal_var=False)

# Compute 95% CI for difference in means (No Diabetes - Diabetes)
mean_diff = glucose_no_diabetes.mean() - glucose_diabetes.mean()
se_diff = np.sqrt(glucose_no_diabetes.var(ddof=1)/len(glucose_no_diabetes) + glucose_diabetes.var(ddof=1)/len(glucose_diabetes))
# Satterthwaite degrees of freedom
dof = (glucose_no_diabetes.var(ddof=1)/len(glucose_no_diabetes) + glucose_diabetes.var(ddof=1)/len(glucose_diabetes))**2 / (
    (glucose_no_diabetes.var(ddof=1)/len(glucose_no_diabetes))**2 / (len(glucose_no_diabetes)-1) +
    (glucose_diabetes.var(ddof=1)/len(glucose_diabetes))**2 / (len(glucose_diabetes)-1)
)
ci_lower = mean_diff - stats.t.ppf(0.975, dof) * se_diff
ci_upper = mean_diff + stats.t.ppf(0.975, dof) * se_diff

print("GLUCOSE LEVEL COMPARISON\n")
print(f"Mean glucose - No Diabetes : {glucose_no_diabetes.mean():.2f}")
print(f"Mean glucose - Diabetes    : {glucose_diabetes.mean():.2f}")
print(f"T-statistic                : {ttest_res.statistic:.3f}")
print(f"Degrees of freedom         : {dof:.2f}")
print(f"P-value                    : {ttest_res.pvalue:.4e} (p < 2.2e-16)")
print(f"95% Confidence Interval    : [{ci_lower:.2f}, {ci_upper:.2f}]")

# Glucose Boxplot
plt.figure(figsize=(7, 6))
sns.boxplot(x='Outcome_Label', y='Glucose', data=diabetes_clean, palette=['#4C72B0', '#DD8452'], hue='Outcome_Label', legend=False)
plt.title("Glucose Levels by Diabetes Outcome", fontsize=13, fontweight='bold')
plt.xlabel("Patient Group")
plt.ylabel("Glucose Level")
plt.tight_layout()
plot_path_7 = "outputs/04_glucose_boxplot.png"
plt.savefig(plot_path_7, dpi=300)
plt.close()
print(f"[Saved Plot]: {plot_path_7}")

# ==============================================================================
# CELL 9: Logistic Regression Model
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 9: LOGISTIC REGRESSION MODEL TRAINING & EVALUATION")
print("=" * 70)

# 80/20 train/test split matching the seed 123
np.random.seed(123)
n = len(diabetes_clean)
indices = np.arange(n)
train_size = int(np.floor(0.8 * n))
train_indices = np.random.choice(indices, size=train_size, replace=False)
test_indices = np.setdiff1d(indices, train_indices)

train_data = diabetes_clean.iloc[train_indices].copy()
test_data = diabetes_clean.iloc[test_indices].copy()

feature_cols = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

X_train = sm.add_constant(train_data[feature_cols])
y_train = train_data['Outcome']

logit_model = sm.Logit(y_train, X_train).fit(disp=False)

print("LOGISTIC REGRESSION MODEL SUMMARY:\n")
print(logit_model.summary())

# Test set predictions
X_test = sm.add_constant(test_data[feature_cols])
y_test = test_data['Outcome']
predicted_prob = logit_model.predict(X_test)
predicted_class = (predicted_prob >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, predicted_class)
cm_df = pd.DataFrame(cm, index=['Actual: 0 (No Diabetes)', 'Actual: 1 (Diabetes)'],
                         columns=['Predicted: 0', 'Predicted: 1'])

print("\nCONFUSION MATRIX:")
print(cm_df)

acc = accuracy_score(y_test, predicted_class) * 100
print(f"\nModel Accuracy: {acc:.2f}%")

# ==============================================================================
# CELL 10: Model Performance Evaluation Metrics
# ==============================================================================
print("\n" + "=" * 70)
print("CELL 10: MODEL PERFORMANCE EVALUATION")
print("=" * 70)

tn, fp, fn, tp = cm.ravel()
precision = (tp / (tp + fp)) * 100
recall = (tp / (tp + fn)) * 100
specificity = (tn / (tn + fp)) * 100
f1 = (2 * precision * recall / (precision + recall))

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score"],
    "Value": [round(acc, 2), round(precision, 2), round(recall, 2), round(specificity, 2), round(f1, 2)]
})

print(metrics_df.to_string(index=False))

# Plot performance metrics
plt.figure(figsize=(9, 6))
bars = plt.bar(metrics_df['Metric'], metrics_df['Value'], color='#4682B4', width=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 110)
plt.title("Logistic Regression Model Performance", fontsize=14, fontweight='bold')
plt.xlabel("Evaluation Metric")
plt.ylabel("Score (%)")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plot_path_10 = "outputs/05_model_performance.png"
plt.savefig(plot_path_10, dpi=300)
plt.close()
print(f"\n[Saved Plot]: {plot_path_10}")

print("\n" + "=" * 70)
print("ANALYSIS EXECUTION COMPLETE! ALL OUTPUTS AND PLOTS GENERATED SUCCESSFULLY.")
print("=" * 70)
