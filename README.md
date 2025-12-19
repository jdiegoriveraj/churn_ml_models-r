# churn_ml_models-r
Customer churn classification in R using Naive Bayes, Logistic Regression, Decision Trees, and Random Forest.
# Customer Churn Classification Analysis (R)

This project presents a complete customer churn classification analysis using supervised machine learning techniques in R.  
The objective is to identify customers likely to cancel a service and compare different classification models based on key performance metrics.

## Dataset
- Customer churn dataset
- CSV format separated by pipe (`|`)
- Target variable: `churn` (Yes / No)

## Preprocessing
- Conversion of categorical variables to factors
- Train/Test split (80% / 20%)
- Feature selection based on business relevance

## Models Implemented
- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Sensitivity (Recall) *(main metric for churn)*
- Specificity
- Precision
- F1-Score

## Model Comparison
A comparative analysis is performed to identify the best model, prioritizing **Sensitivity**, since false negatives (lost customers) are more costly than false positives.

## Key Findings
- The dataset is imbalanced (majority non-churn customers)
- Random Forest shows the best overall performance
- Customer service calls and usage patterns are among the most important predictors

## Technologies
- R
- Packages: `caret`, `e1071`, `rpart`, `rpart.plot`, `randomForest`

## Notes
- This project was developed as part of an academic data science module.
- The script is fully reproducible.

