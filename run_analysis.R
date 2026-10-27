# ==============================================================================
# CA3 Mini Project: Healthcare Analytics
# Title: Identifying Diabetes Risk Factors Using R
# ==============================================================================

# Ensure outputs directory exists
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

# ------------------------------------------------------------------------------
# 1. Load the dataset & Initial Inspection
# ------------------------------------------------------------------------------
url <- "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
diabetes <- read.csv(url)

cat("First 6 Records:\n")
print(head(diabetes))

cat("\nDataset Dimensions:\n")
print(dim(diabetes))

cat("\nColumn Names:\n")
print(names(diabetes))

cat("\nDataset Structure:\n")
str(diabetes)

cat("\nSummary Statistics:\n")
print(summary(diabetes))

# ------------------------------------------------------------------------------
# 2. Check for missing values & zero-value anomalies
# ------------------------------------------------------------------------------
cat("\nMissing Values in Each Column:\n")
print(colSums(is.na(diabetes)))

medical_columns <- c(
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI"
)

cat("\nZero Values in Medical Columns (Clinically Invalid):\n")
print(sapply(diabetes[medical_columns], function(x) sum(x == 0)))

cat("\nDiabetes Outcome Distribution:\n")
print(table(diabetes$Outcome))

cat("\nOutcome Percentages:\n")
print(round(prop.table(table(diabetes$Outcome)) * 100, 2))

# ------------------------------------------------------------------------------
# 3. Data Cleaning (Median Imputation for biologically impossible zeros)
# ------------------------------------------------------------------------------
diabetes_clean <- diabetes

for (col in medical_columns) {
  diabetes_clean[[col]][diabetes_clean[[col]] == 0] <- NA
}

cat("\nMissing values after zero-value correction:\n")
print(colSums(is.na(diabetes_clean)))

for (col in medical_columns) {
  diabetes_clean[[col]][is.na(diabetes_clean[[col]])] <-
    median(diabetes_clean[[col]], na.rm = TRUE)
}

cat("\nMissing values after cleaning:\n")
print(colSums(is.na(diabetes_clean)))

cat("\nCleaned dataset dimensions:\n")
print(dim(diabetes_clean))

cat("\nSummary of cleaned medical measurements:\n")
print(summary(diabetes_clean[medical_columns]))

# ------------------------------------------------------------------------------
# 4. Exploratory Visualizations
# ------------------------------------------------------------------------------
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
}
library(ggplot2)

diabetes_clean$Outcome_Label <- factor(
  diabetes_clean$Outcome,
  levels = c(0, 1),
  labels = c("No Diabetes", "Diabetes")
)

# 1. Diabetes outcome distribution
p1 <- ggplot(diabetes_clean, aes(x = Outcome_Label, fill = Outcome_Label)) +
  geom_bar() +
  labs(title = "Distribution of Diabetes Outcomes", x = "Patient Group", y = "Number of Patients") +
  theme_minimal() + theme(legend.position = "none")

# 2. Average glucose level by diabetes outcome
p2 <- ggplot(diabetes_clean, aes(x = Outcome_Label, y = Glucose, fill = Outcome_Label)) +
  stat_summary(fun = mean, geom = "col", width = 0.6) +
  labs(title = "Average Glucose Level by Diabetes Outcome", x = "Patient Group", y = "Average Glucose Level") +
  theme_minimal() + theme(legend.position = "none")

# 3. BMI distribution by diabetes outcome
p3 <- ggplot(diabetes_clean, aes(x = Outcome_Label, y = BMI, fill = Outcome_Label)) +
  geom_boxplot() +
  labs(title = "BMI Comparison by Diabetes Outcome", x = "Patient Group", y = "Body Mass Index (BMI)") +
  theme_minimal() + theme(legend.position = "none")

# 4. Age distribution by diabetes outcome
p4 <- ggplot(diabetes_clean, aes(x = Age, fill = Outcome_Label)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 20) +
  labs(title = "Age Distribution by Diabetes Outcome", x = "Age (years)", y = "Number of Patients", fill = "Patient Group") +
  theme_minimal()

# ------------------------------------------------------------------------------
# 5. Average Measurements & Feature Correlations
# ------------------------------------------------------------------------------
cat("\nAVERAGE MEDICAL MEASUREMENTS BY DIABETES OUTCOME:\n\n")
group_means <- aggregate(
  cbind(Glucose, BloodPressure, Insulin, BMI, Age) ~ Outcome_Label,
  data = diabetes_clean,
  FUN = mean
)
group_means[, -1] <- round(group_means[, -1], 2)
print(group_means)

cat("\nCORRELATION WITH DIABETES OUTCOME:\n\n")
variables <- c(
  "Pregnancies", "Glucose", "BloodPressure",
  "SkinThickness", "Insulin", "BMI",
  "DiabetesPedigreeFunction", "Age"
)

correlations <- sapply(variables, function(variable) {
  cor(diabetes_clean[[variable]], diabetes_clean$Outcome)
})

correlation_results <- data.frame(
  Variable = variables,
  Correlation = round(as.numeric(correlations), 3)
)
correlation_results <- correlation_results[order(-abs(correlation_results$Correlation)), ]
print(correlation_results, row.names = FALSE)

# ------------------------------------------------------------------------------
# 6. Correlation Heatmap
# ------------------------------------------------------------------------------
numeric_data <- diabetes_clean[, sapply(diabetes_clean, is.numeric)]
cor_matrix <- cor(numeric_data)
cor_data <- as.data.frame(as.table(cor_matrix))
names(cor_data) <- c("Variable1", "Variable2", "Correlation")

# ------------------------------------------------------------------------------
# 7 & 8. Hypothesis Testing: Welch's Two-Sample t-test
# ------------------------------------------------------------------------------
cat("\nGLUCOSE LEVEL COMPARISON (WELCH'S T-TEST):\n\n")
glucose_test <- t.test(Glucose ~ Outcome_Label, data = diabetes_clean)
print(glucose_test)

cat("\nMean glucose - No Diabetes:", round(mean(diabetes_clean$Glucose[diabetes_clean$Outcome == 0]), 2), "\n")
cat("Mean glucose - Diabetes:", round(mean(diabetes_clean$Glucose[diabetes_clean$Outcome == 1]), 2), "\n")
cat("T-statistic:", round(unname(glucose_test$statistic), 3), "\n")
cat("P-value:", format.pval(glucose_test$p.value, digits = 4), "\n")
cat("95% Confidence Interval:\n")
print(round(glucose_test$conf.int, 2))

# ------------------------------------------------------------------------------
# 9. Logistic Regression Model
# ------------------------------------------------------------------------------
set.seed(123)
train_index <- sample(seq_len(nrow(diabetes_clean)), size = floor(0.8 * nrow(diabetes_clean)))
train_data <- diabetes_clean[train_index, ]
test_data <- diabetes_clean[-train_index, ]

diabetes_model <- glm(
  Outcome ~ Pregnancies + Glucose + BloodPressure +
    SkinThickness + Insulin + BMI +
    DiabetesPedigreeFunction + Age,
  data = train_data,
  family = binomial
)

cat("\nLOGISTIC REGRESSION MODEL SUMMARY:\n\n")
print(summary(diabetes_model))

predicted_prob <- predict(diabetes_model, newdata = test_data, type = "response")
predicted_class <- ifelse(predicted_prob >= 0.5, 1, 0)

confusion_matrix <- table(Actual = test_data$Outcome, Predicted = predicted_class)
cat("\nCONFUSION MATRIX:\n")
print(confusion_matrix)

accuracy <- mean(predicted_class == test_data$Outcome) * 100
cat("\nModel Accuracy:", round(accuracy, 2), "%\n")

# ------------------------------------------------------------------------------
# 10. Model Performance Metrics
# ------------------------------------------------------------------------------
TP <- sum(test_data$Outcome == 1 & predicted_class == 1)
TN <- sum(test_data$Outcome == 0 & predicted_class == 0)
FP <- sum(test_data$Outcome == 0 & predicted_class == 1)
FN <- sum(test_data$Outcome == 1 & predicted_class == 0)

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
specificity <- TN / (TN + FP)
f1_score <- 2 * precision * recall / (precision + recall)
accuracy <- (TP + TN) / (TP + TN + FP + FN)

metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score"),
  Value = round(c(accuracy, precision, recall, specificity, f1_score) * 100, 2)
)

cat("\nMODEL PERFORMANCE EVALUATION:\n\n")
print(metrics, row.names = FALSE)
