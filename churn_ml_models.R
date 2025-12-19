# ============================================================================
# MODULE 5 WORKSHOP - POINT 3: CHURN CLASSIFICATION
# Clean and fully functional version (English)
# ============================================================================

# INSTALL PACKAGES (run only once if needed):
# install.packages("e1071")
# install.packages("caret")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("randomForest")

# 1. LOAD LIBRARIES ---------------------------------------------------------
library(e1071)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

# 2. LOAD DATA --------------------------------------------------------------

# Adjust the path if needed
Churn <- read.csv(
  "data/churn_dataset.csv foler",
  sep = "|",
  header = TRUE
)

# Alternatively:
# Churn <- read.csv(file.choose(), sep="|", header=TRUE)

# Preview data
head(Churn)
str(Churn)

cat("\n========== DATASET INFORMATION ==========\n")
cat("Dimensions:", nrow(Churn), "rows x", ncol(Churn), "columns\n")

# 3. DATA PREPROCESSING -----------------------------------------------------

# Convert categorical variables to factors
Churn$area_code <- as.factor(Churn$area_code)
Churn$international_plan <- as.factor(Churn$international_plan)
Churn$voice_mail_plan <- as.factor(Churn$voice_mail_plan)
Churn$churn <- as.factor(Churn$churn)

# 4. EXPLORATORY DATA ANALYSIS (EDA) ----------------------------------------

cat("\n========== EXPLORATORY DATA ANALYSIS ==========\n")
cat("\nChurn distribution:\n")
print(table(Churn$churn))
print(prop.table(table(Churn$churn)) * 100)

# Visualizations
par(mfrow = c(1, 2))

plot(
  Churn$churn,
  col = c("green", "red"),
  main = "Customer Distribution",
  ylab = "Frequency"
)

percentages <- round(prop.table(table(Churn$churn)) * 100, 2)
labels <- paste(c("No", "Yes"), "\n", percentages, "%")

pie(
  table(Churn$churn),
  col = c("green", "red"),
  labels = labels,
  main = "Churn Distribution"
)

par(mfrow = c(1, 1))

cat("\nMajority class (No Churn):", percentages[1], "%\n")
cat("Minority class (Churn):", percentages[2], "%\n")

# 5. FEATURE SELECTION & DATA SPLIT -----------------------------------------

churn_model <- Churn[, c(
  "international_plan",
  "total_day_minutes",
  "total_day_calls",
  "total_intl_minutes",
  "number_customer_service_calls",
  "total_intl_calls",
  "churn"
)]

colnames(churn_model) <- c(
  "International_Plan",
  "Day_Minutes",
  "Day_Calls",
  "International_Minutes",
  "Customer_Service_Calls",
  "International_Calls",
  "Churn"
)

# Train/Test split: 80% / 20%
set.seed(123)
partition <- sample(2, nrow(churn_model), replace = TRUE, prob = c(0.8, 0.2))
train_data <- churn_model[partition == 1, ]
test_data <- churn_model[partition == 2, ]

cat("\n========== DATA SPLIT ==========\n")
cat("Training set:", nrow(train_data), "rows\n")
cat("Test set:", nrow(test_data), "rows\n")

# 6. MODEL 1: NAIVE BAYES ----------------------------------------------------

nb_model <- naiveBayes(Churn ~ ., data = train_data)
nb_pred <- predict(nb_model, test_data)

cm_nb <- confusionMatrix(nb_pred, test_data$Churn, positive = "yes")
print(cm_nb)

# 7. MODEL 2: LOGISTIC REGRESSION -------------------------------------------

logit_model <- glm(Churn ~ ., data = train_data, family = binomial)

logit_prob <- predict(logit_model, test_data, type = "response")
logit_pred <- factor(
  ifelse(logit_prob > 0.5, "yes", "no"),
  levels = c("no", "yes")
)

cm_logit <- confusionMatrix(logit_pred, test_data$Churn, positive = "yes")
print(cm_logit)

# 8. MODEL 3: DECISION TREE -------------------------------------------------

tree_model <- rpart(Churn ~ ., data = train_data, method = "class")

rpart.plot(
  tree_model,
  main = "Decision Tree - Churn",
  extra = 104,
  box.palette = "RdYlGn",
  shadow.col = "gray"
)

tree_pred <- predict(tree_model, test_data, type = "class")

cm_tree <- confusionMatrix(tree_pred, test_data$Churn, positive = "yes")
print(cm_tree)

# 9. MODEL 4: RANDOM FOREST -------------------------------------------------

set.seed(123)
rf_model <- randomForest(
  Churn ~ .,
  data = train_data,
  ntree = 100,
  importance = TRUE
)

rf_pred <- predict(rf_model, test_data)

cm_rf <- confusionMatrix(rf_pred, test_data$Churn, positive = "yes")
print(cm_rf)

cat("\nVariable Importance:\n")
print(importance(rf_model))
varImpPlot(rf_model, main = "Variable Importance - Random Forest")

# 10. MODEL COMPARISON ------------------------------------------------------

# Helper function to extract metrics
extract_metrics <- function(cm) {
  list(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Pos Pred Value"]),
    F1 = as.numeric(cm$byClass["F1"])
  )
}

metrics_nb <- extract_metrics(cm_nb)
metrics_logit <- extract_metrics(cm_logit)
metrics_tree <- extract_metrics(cm_tree)
metrics_rf <- extract_metrics(cm_rf)

comparison <- data.frame(
  Model = c("Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(metrics_nb$Accuracy, metrics_logit$Accuracy,
               metrics_tree$Accuracy, metrics_rf$Accuracy),
  Sensitivity = c(metrics_nb$Sensitivity, metrics_logit$Sensitivity,
                  metrics_tree$Sensitivity, metrics_rf$Sensitivity),
  Specificity = c(metrics_nb$Specificity, metrics_logit$Specificity,
                  metrics_tree$Specificity, metrics_rf$Specificity),
  Precision = c(metrics_nb$Precision, metrics_logit$Precision,
                metrics_tree$Precision, metrics_rf$Precision),
  F1_Score = c(metrics_nb$F1, metrics_logit$F1,
               metrics_tree$F1, metrics_rf$F1)
)

comparison[, 2:6] <- round(comparison[, 2:6], 4)

cat("\n*** MODEL COMPARISON TABLE ***\n")
print(comparison)

# Visual comparison
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

barplot(comparison$Accuracy, names.arg = c("NB","Logit","Tree","RF"),
        main = "Accuracy", ylim = c(0,1))
abline(h = max(comparison$Accuracy), col = "red", lty = 2)

barplot(comparison$Sensitivity, names.arg = c("NB","Logit","Tree","RF"),
        main = "Sensitivity (Recall)", ylim = c(0,1))
abline(h = max(comparison$Sensitivity), col = "red", lty = 2)

barplot(comparison$Specificity, names.arg = c("NB","Logit","Tree","RF"),
        main = "Specificity", ylim = c(0,1))
abline(h = max(comparison$Specificity), col = "red", lty = 2)

barplot(comparison$F1_Score, names.arg = c("NB","Logit","Tree","RF"),
        main = "F1-Score", ylim = c(0,1))
abline(h = max(comparison$F1_Score), col = "red", lty = 2)

par(mfrow = c(1, 1))

# 11. BEST MODEL SELECTION --------------------------------------------------

best_sensitivity <- which.max(comparison$Sensitivity)
best_f1 <- which.max(comparison$F1_Score)

cat("\nWhy is SENSITIVITY critical for churn prediction?\n")
cat("• Detecting customers who WILL churn is the priority\n")
cat("• False Negatives = lost customers (high cost)\n")
cat("• False Positives = retention effort (lower cost)\n")

cat("\n*** BEST MODEL BASED ON SENSITIVITY ***\n")
cat("→", comparison$Model[best_sensitivity], "\n")
cat("  Sensitivity:", comparison$Sensitivity[best_sensitivity], "\n")

cat("\n*** BEST MODEL BASED ON F1-SCORE ***\n")
cat("→", comparison$Model[best_f1], "\n")
cat("  F1-Score:", comparison$F1_Score[best_f1], "\n")

# 12. FINAL CONCLUSIONS -----------------------------------------------------

cat("\n1. PROBLEM:\n")
cat("   Imbalanced dataset (", round(percentages[1], 1), "% non-churn)\n")

cat("\n2. MODELS COMPARED: 4\n")
cat("   • Naive Bayes\n")
cat("   • Logistic Regression\n")
cat("   • Decision Tree\n")
cat("   • Random Forest\n")

cat("\n3. KEY METRIC: SENSITIVITY\n")
cat("   Critical for detecting churn customers\n")

cat("\n4. BEST MODEL:\n")
cat("   ✓", comparison$Model[best_sensitivity], "\n")
cat("   Justification: Highest churn detection rate\n")

cat("\n5. BUSINESS RECOMMENDATIONS:\n")
cat("   • Implement retention strategies\n")
cat("   • Prioritize high-risk customers\n")
cat("   • Monitor customer service interactions\n")

