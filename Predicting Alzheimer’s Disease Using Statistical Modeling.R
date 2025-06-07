# Import Packages ---------------------------------------------------------
library(tidyverse)
library(dplyr)
library(glmnet)
library(randomForest)
library(MASS)
library(gridExtra)
library(caret)
library(tree)
library(class)
library(e1071)
library(kernlab)
library(vtable)
library(corrplot)
library(lattice)
library(patchwork)
library(data.table)
library(Hmisc)
library(pROC)
library(ROCR)
library(leaps)
library(boot)
library(bestglm)
library(PredPsych)
library(gbm)
library(xgboost)
library(doParallel)
library(parallel)
library(MLmetrics)
library(rlang)

# Import and Preview Data-------------------------------------------------------------
df = read_csv("Desktop/UC Davis/STA 160/Alzheimers Data/train.csv")
head(df)

# Data Cleaning -----------------------------------------------------------
df = df %>% dplyr::select(-PatientID, -DoctorInCharge) # Remove unneeded columns

# Separate reponse and continuous/discrete variables
age = df[, 1]

cont_var_list = c(5, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 27)

continuous_vars_df = df[, cont_var_list]
discrete_vars = df[, -cont_var_list]
discrete_vars = discrete_vars[, -1] # Drop age

discrete_vars_list = colnames(discrete_vars)
discrete_vars_list = discrete_vars_list[-18] # Remove diagnosis

# EDA ---------------------------------------------------------------------
summary(continuous_vars_df) #Summary Statistics

# Get Counts of each value for discrete variables
value_counts_df = discrete_vars %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value") %>%
  dplyr::count(Feature, Value) %>%
  pivot_wider(names_from = Value, values_from = n, values_fill = 0)

# Grid of histograms for continuous variables
dev.new()
hist.data.frame(continuous_vars_df)

# Check skewness of continuous variables
sapply(continuous_vars_df, skewness)

# Histograms of skewed variables
ggplot(df, aes(x = DiastolicBP)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Diastolic Blood Pressure",
       x = "Diastolic BP", y = "Frequency") +
  theme_minimal()

ggplot(df, aes(x = CholesterolTriglycerides)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Cholesterol Triglycerides",
       x = "CholesterolTriglycerides", y = "Frequency") +
  theme_minimal()

ggplot(df, aes(x = CholesterolLDL)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of CholesterolLDL",
       x = "CholesterolLDL", y = "Frequency") +
  theme_minimal()

ggplot(df, aes(x = CholesterolHDL)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of CholesteroHLDL",
       x = "CholesterolHDL", y = "Frequency") +
  theme_minimal()

ggplot(df, aes(x = SystolicBP)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Systolic Blood Pressure",
       x = "SystolicBP", y = "Frequency") +
  theme_minimal()

ggplot(df, aes(x = AlcoholConsumption)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Alcohol Consumption",
       x = "Alcohol Consumption", y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(df, aes(x = SleepQuality)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Sleep Quality",
       x = "Sleep Quality", y = "Frequency") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))

# Box Plot Grid for Continuous Variables
continuous_vars_long = pivot_longer(continuous_vars_df, 
                                     everything(), 
                                     names_to = "Variable", 
                                     values_to = "Value")

ggplot(continuous_vars_long, aes(x = "", y = Value)) +
  geom_boxplot(fill = "skyblue") +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(x = NULL, y = "Value", title = "Boxplots for Predictor Variables") +
  theme(
    strip.text = element_text(size = 12),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

#Density Curve Grid

ggplot(continuous_vars_long, aes(x = Value, fill = Variable)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(x = "Value", y = "Density", title = "Density Curves by Variable") +
  theme(strip.text = element_text(size = 12),
        legend.position = "none")

# Correlations

corrplot(cor(continuous_vars_df), method = "circle") # Correlation plot for each continuous variable
cor(x = continuous_vars_df, y = as.numeric(df$Diagnosis)) # Checking correlation with diagnosis and continuous predictors

# Graph correlations between continuous predictors and diagnosis
cors = sapply(continuous_vars_df, function(x) cor(x, df$Diagnosis, use = "complete.obs"))
cor_df = data.frame(Variable = names(cors), Correlation = cors) # Create correlation data frame for plotting

# Plot correlations
ggplot(cor_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Correlation with Diagnosis", x = "Variable", y = "Correlation") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Check the top 5 most positive and negative correlations with diagnosis
top_negative = cor_df[order(cor_df$Correlation), ][1:5, ]
top_positive = cor_df[order(-cor_df$Correlation), ][1:5, ]
top_correlated = rbind(
  Top_Positive = top_positive,
  Top_Negative = top_negative
)

# Violin Plots
p1 = ggplot(df, aes(x = factor(Diagnosis), y = MMSE)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Diagnosis", y = "MMSE") +
  theme_minimal() # Clear difference between classes

p2 = ggplot(df, aes(x = factor(Diagnosis), y = FunctionalAssessment)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Diagnosis", y = "Functional Assessment") +
  theme_minimal() # Clear difference between classes

p3 = ggplot(df, aes(x = factor(Diagnosis), y = ADL)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Diagnosis", y = "ADL") +
  theme_minimal() # Clear difference between classes

(p1 | p2) / (p3 | plot_spacer()) +
plot_annotation(
  title = "Violin Plots of MMSE, Functional Assessment, and ADL by Alzheimer's Diagnosis",
  theme = theme(plot.title = element_text(hjust = 0.5)))

# Violin plots with no clear pattern

ggplot(df, aes(x = factor(Diagnosis), y = Age)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)", y = "Age") +
  theme_minimal()

ggplot(df, aes(x = factor(Diagnosis), y = AlcoholConsumption)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)", y = "Alcohol Consumption") +
  theme_minimal()

ggplot(df, aes(x = factor(Diagnosis), y = PhysicalActivity)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)", y = "Physical Acitvity") +
  theme_minimal()

ggplot(df, aes(x = factor(Diagnosis), y = BMI)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)", y = "BMI") +
  theme_minimal()

ggplot(df, aes(x = factor(Diagnosis), y = SleepQuality)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)", y = "Sleep Quality") +
  theme_minimal()

ggplot(df, aes(x = factor(Diagnosis), y = CholesterolHDL)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)", y = "SystolicBP") +
  theme_minimal()

# Plot that is interesting because people with Alzheimer's have better diet quality
ggplot(df, aes(x = factor(Diagnosis), y = DietQuality)) +
  geom_violin(trim = FALSE, fill = "skyblue") +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(title = "Violin Plot of Diet Quality by Diagnosis",
       x = "Alzheimer's Diagnosis (0 = No, 1 = Yes)",
       y = "Diet Quality") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Check diet quality differences between patients with a positive and negative diagnosis
df %>%
  dplyr::group_by(Diagnosis) %>%
  dplyr::summarise(
    mean_diet = mean(DietQuality, na.rm = TRUE),
    median_diet = median(DietQuality, na.rm = TRUE),
    count = dplyr::n()
  )

# Check diagnosis patterns by gender
gender_labels = c("Male", "Female")         
diagnosis_labels = c("No Alzheimer's", "Alzheimer's") 

prop.table(table(Gender = factor(df$Gender, labels = gender_labels),
                 Diagnosis = factor(df$Diagnosis, labels = diagnosis_labels)),
           margin = 1)

# Check proportions of diagnoses based on BehavioralProglems, Memory Complaints, and Ethnicity
p1_disc = ggplot(df, aes(x = factor(BehavioralProblems), fill = factor(Diagnosis))) +
    geom_bar(position = "fill") +
    labs(x = "Behavioral Problems", y = "Proportion", fill = "Diagnosis") +
    ggtitle("Proportion of Diagnosis\nbased on Behavioral Problems") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))

p2_disc = ggplot(df, aes(x = factor(MemoryComplaints), fill = factor(Diagnosis))) +
  geom_bar(position = "fill") +
  labs(x = "Memory Complaints", y = "Proportion", fill = "Diagnosis") +
  ggtitle("Proportion of Diagnosis\nbased on Memory Complaints") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

p3_disc = ggplot(df, aes(x = factor(Ethnicity), fill = factor(Diagnosis))) +
  geom_bar(position = "fill") +
  labs(x = "Ethnicity", y = "Proportion", fill = "Diagnosis") +
  ggtitle("Proportion of Diagnosis\nbased on Ethnicity") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
p1_disc + p2_disc + p3_disc

# Create table of proportions of patients diagnosed with Alzheimer's based on ethnicity
ethnicity_labels = c("Caucasian", "African American", "Asian", "Other")

ethnicity_factor = factor(df$Ethnicity, levels = 0:3, labels = ethnicity_labels)
diagnosis_factor = factor(df$Diagnosis, levels = 0:1, labels = diagnosis_labels)

prop.table(table(Ethnicity = ethnicity_factor, Diagnosis = diagnosis_factor), margin = 1)

table(`Family History` = df$FamilyHistoryAlzheimers,
      `Diagnosis` = df$Diagnosis) # Confusion matrix of family history vs. Alzheimer's Diagnosis

# Check outliers using Q1 - 1.5IQR and Q3 + 1.5IQR
outlier_check = lapply(continuous_vars_df, function(x) {
  Q1 = quantile(x, 0.25, na.rm = TRUE)
  Q3 = quantile(x, 0.75, na.rm = TRUE)
  IQR = Q3 - Q1
  outliers = x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)
  sum(outliers, na.rm = TRUE)
})

# Check outliers based on z-scores (at least 3 standard devs)
z_outliers = lapply(continuous_vars_df, function(x) {
  z = scale(x)
  sum(abs(z) > 3, na.rm = TRUE)
})

# Print results
outlier_check
z_outliers 

# No outliers found using either test

# Modeling Set Up ----------------------------------------------------------------
set.seed(1) # Set a random seed to ensure consistent folds for cross-validation

df = df %>% dplyr::select(Diagnosis, everything()) # Move the diagnosis to the front

accuracy_df = data.frame(Accuracy = numeric(),
                         Precision = numeric(),
                         Recall = numeric(),
                         `F1 Score` = numeric()) # Data frame to save model accuracy metrics
# Models used:
  # Logistic
  # Logistic Best Subset
  # LASSO
  # Ridge
  # LDA
  # QDA
  # KNN
  # Decision Tree
  # Random Forest
  # Gradient Boosting
  # XGBoost
  # Used 10-fold CV to compare prediction accuracy

# Logistic Regression -----------------------------------------------------

folds = createFolds(df$Diagnosis, k = 10, list = TRUE) # Create folds for CV

# Initialize blank lists for accuracies
precision_list = recall_list = f1_list = accuracy_list = numeric(10)

# 10-fold CV for logistic regression
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  test_data = df[test_idx, ]
  train_data = df[-test_idx, ]
  
  # Fit model, predict, and get actual values of response
  model = glm(Diagnosis ~ ., data = train_data, family = binomial)
  probs = predict(model, test_data, type = "response")
  pred = ifelse(probs > 0.5, 1, 0)
  actual = test_data$Diagnosis
  
  # Make confusion matrix
  cm = confusionMatrix(factor(pred), factor(actual), positive = "1")
  
  # Calculate accuracy metrics
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
  accuracy_list[i]  = cm$overall["Accuracy"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_precision = mean(precision_list)
mean_recall = mean(recall_list)
mean_f1 = mean(f1_list)
mean_accuracy = mean(accuracy_list)

# Add accuracy metrics to data frame
accuracy_df["Logistic", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)

# Logistic Regression Best Subset -----------------------------------------

# Set random seed
set.seed(1)

# Using AIC, find best subset for logistic
stepAIC(glm(Diagnosis ~ ., data = train_data, family = binomial), direction = "both")

# Initialize blank lists for accuracies
precision_list = recall_list = f1_list = accuracy_list = numeric(10)

# 10-fold CV for logistic regression best subset
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  test_data = df[test_idx, ]
  train_data = df[-test_idx, ]
  
  # Fit model, predict, and get actual values of response
  model = glm(formula = Diagnosis ~ Age + EducationLevel + Smoking + CholesterolHDL + 
                 MMSE + FunctionalAssessment + MemoryComplaints + BehavioralProblems + 
                 ADL, family = binomial, data = train_data)
  probs = predict(model, test_data, type = "response")
  pred = ifelse(probs > 0.5, 1, 0)
  actual = test_data$Diagnosis
  
  # Make confusion matrix
  cm = confusionMatrix(factor(pred), factor(actual), positive = "1")
  
  # Calculate accuracy metrics
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
  accuracy_list[i]  = cm$overall["Accuracy"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_precision = mean(precision_list)
mean_recall = mean(recall_list)
mean_f1 = mean(f1_list)
mean_accuracy = mean(accuracy_list)

# Add accuracy metrics to data frame
accuracy_df["Best Subset Logistic", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)

# Forward Selection didn't eliminate enough variables, backward gave the same result as best subset

# LASSO ---------------------------------------------------------------------

# Set random seed
set.seed(1)

# Initialize blank lists for accuracies
accuracy_list = precision_list = recall_list = f1_list = numeric(10)

# 10-fold CV for the LASSO model
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  train_data = df[-test_idx, ]
  test_data = df[test_idx, ]
  
  # Make predictors into matrix compatible with glmnet function
  x_train = as.matrix(sapply(train_data[, -1], as.numeric))
  x_test  = as.matrix(sapply(test_data[, -1], as.numeric))
  
  # Create response variables
  y_train = as.numeric(train_data[[1]])
  y_test  = as.numeric(test_data[[1]])
  
  # Use CV to get best lambda value
  cv_lasso = cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
  lambda_min = cv_lasso$lambda.min
  
  # Fit model and get predictions 
  lasso_model = glmnet(x_train, y_train, alpha = 1, lambda = lambda_min, family = "binomial")
  probs = predict(lasso_model, newx = x_test, type = "response")
  preds = ifelse(probs > 0.5, 1, 0)
  preds_factor  = factor(preds, levels = c(0, 1))
  y_test_factor = factor(y_test, levels = c(0, 1))
  
  # Make confusion matrix
  cm = confusionMatrix(preds_factor, y_test_factor, positive = "1")
  
  # Calculate accuracy metrics
  accuracy_list[i]  = cm$overall["Accuracy"]
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_accuracy  = mean(accuracy_list)
mean_precision = mean(precision_list)
mean_recall    = mean(recall_list)
mean_f1        = mean(f1_list)

# Add accuracy metrics to data frame
accuracy_df["LASSO", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)

# Ridge -------------------------------------------------------------------

# Set random seed
set.seed(1)

# Initialize blank lists for accuracies
accuracy_list = precision_list = recall_list = f1_list = numeric(10)

# 10-fold CV for the Ridge model
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  train_data = df[-test_idx, ]
  test_data = df[test_idx, ]

  # Make predictors into matrix compatible with glmnet function
  x_train = as.matrix(sapply(train_data[, -1], as.numeric))
  x_test  = as.matrix(sapply(test_data[, -1], as.numeric))
  
  # Create response variables
  y_train = as.numeric(train_data[[1]])
  y_test  = as.numeric(test_data[[1]])
  
  # Use CV to get best lambda value
  cv_ridge = cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")
  lambda_min = cv_ridge$lambda.min
  
  # Fit model and get predictions 
  ridge_model = glmnet(x_train, y_train, alpha = 0, lambda = lambda_min, family = "binomial")
  probs = predict(lasso_model, newx = x_test, type = "response")
  preds = ifelse(probs > 0.5, 1, 0)
  preds_factor  = factor(preds, levels = c(0, 1))
  y_test_factor = factor(y_test, levels = c(0, 1))
  
  # Make confusion matrix
  cm = confusionMatrix(preds_factor, y_test_factor, positive = "1")
  
  # Calculate accuracy metrics
  accuracy_list[i]  = cm$overall["Accuracy"]
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_accuracy  = mean(accuracy_list)
mean_precision = mean(precision_list)
mean_recall    = mean(recall_list)
mean_f1        = mean(f1_list)

# Add accuracy metrics to data frame
accuracy_df["Ridge", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)


# LDA ---------------------------------------------------------------------

# Make the response a factor so it works with LDA
df$Diagnosis = factor(df$Diagnosis, levels = c(0, 1), labels = c("No", "Yes"))

# Set random seed
set.seed(1)

# Initialize blank lists for accuracies
accuracy_list = precision_list = recall_list = f1_list = numeric(10)

# 10-fold CV for the LDA model
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  train_data = df[-test_idx, ]
  test_data = df[test_idx, ]
  
  # Fit model and get predictions 
  lda_model = train(Diagnosis ~ ., data = train_data, method = "lda")
  preds = predict(lda_model, newdata = test_data)
  actual = test_data$Diagnosis
  preds = factor(preds, levels = c("No", "Yes"))
  actual = factor(actual, levels = c("No", "Yes"))
  
  # Make confusion matrix
  cm = confusionMatrix(preds, actual, positive = "Yes")
  
  # Calculate accuracy metrics
  accuracy_list[i]  = cm$overall["Accuracy"]
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_accuracy  = mean(accuracy_list)
mean_precision = mean(precision_list)
mean_recall    = mean(recall_list)
mean_f1        = mean(f1_list)

# Add accuracy metrics to data frame
accuracy_df["LDA", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)

# QDA ----------------------------------------------------------------------

# Set random seed
set.seed(1)

# Initialize blank lists for accuracies
accuracy_list = precision_list = recall_list = f1_list = numeric(10)

# 10-fold CV for the QDA model
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  train_data = df[-test_idx, ]
  test_data = df[test_idx, ]
  
  # Fit model and get predictions 
  qda_model = train(Diagnosis ~ ., data = train_data, method = "qda")
  preds = predict(qda_model, newdata = test_data)
  actual = test_data$Diagnosis
  preds = factor(preds, levels = c("No", "Yes"))
  actual = factor(actual, levels = c("No", "Yes"))
  
  # Make confusion matrix
  cm = confusionMatrix(preds, actual, positive = "Yes")
  
  # Calculate accuracy metrics
  accuracy_list[i]  = cm$overall["Accuracy"]
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_accuracy  = mean(accuracy_list)
mean_precision = mean(precision_list)
mean_recall    = mean(recall_list)
mean_f1        = mean(f1_list)

# Add accuracy metrics to data frame
accuracy_df["QDA", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)

# KNN ---------------------------------------------------------------------

# Re-import data and process to work with KNN
df = read_csv("Desktop/UC Davis/STA 160/Alzheimers Data/train.csv")
df = df %>% dplyr::select(-PatientID, -DoctorInCharge) # Remove unneeded columns
df = df %>% dplyr::select(Diagnosis, everything()) # Move the diagnosis to the front

# Make the response a factor so it works with KNN
df$Diagnosis = factor(df$Diagnosis, levels = c(0, 1), labels = c("No", "Yes"))

# Center and scale the predictors to work with KNN
preproc = preProcess(df[, -1], method = c("center", "scale"))
df_scaled = predict(preproc, df[, -1])
df_scaled$Diagnosis = df$Diagnosis  

# Set random seed
set.seed(1)

# Initialize blank lists for accuracies
accuracy_list = precision_list = recall_list = f1_list = numeric(10)

# 10-fold CV for the KNN model
for (i in 1:10) {
  # Split data into train and test
  test_idx = folds[[i]]
  train_data = df_scaled[-test_idx, ]
  test_data = df_scaled[test_idx, ]
  
  # Fit KNN with k = 3 and get predictions
  knn_model = train(Diagnosis ~ ., data = train_data,
                     method = "knn",
                     tuneGrid = data.frame(k = 3))
  preds = predict(knn_model, newdata = test_data)
  actual = test_data$Diagnosis
  preds = factor(preds, levels = c("No", "Yes"))
  actual = factor(actual, levels = c("No", "Yes"))
  
  # Make confusion matrix
  cm = confusionMatrix(preds, actual, positive = "Yes")
  
  # Calculate accuracy metrics
  accuracy_list[i]  = cm$overall["Accuracy"]
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_accuracy  = mean(accuracy_list)
mean_precision = mean(precision_list)
mean_recall    = mean(recall_list)
mean_f1        = mean(f1_list)

# Add accuracy metrics to data frame
accuracy_df["KNN", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)

# Decision Tree ----------------------------------------------------------

# Set random seed
set.seed(1)

# Initialize blank lists for accuracies
accuracy_list = precision_list = recall_list = f1_list = numeric(10)

# 10-fold CV for the decision tree model
for (i in 1:10) {
  # Split data into train and test
  train = df[-folds[[i]], ]
  test  = df[folds[[i]], ]
  
  # Fit model, predict, and get actual values of response
  tree_model = tree(Diagnosis ~ ., data = train)
  predictions = predict(tree_model, test, type = "class")
  predictions = factor(predictions, levels = c("No", "Yes"))
  actual = factor(test$Diagnosis, levels = c("No", "Yes"))
  
  # Make confusion matrix
  cm = confusionMatrix(predictions, actual, positive = "Yes")
  
  # Calculate accuracy metrics
  accuracy_list[i]  = cm$overall["Accuracy"]
  precision_list[i] = cm$byClass["Precision"]
  recall_list[i]    = cm$byClass["Recall"]
  f1_list[i]        = cm$byClass["F1"]
}

# Average the metrics for the 10 folds to get final accuracies
mean_accuracy  = mean(accuracy_list)
mean_precision = mean(precision_list)
mean_recall    = mean(recall_list)
mean_f1        = mean(f1_list)

# Add accuracy metrics to data frame
accuracy_df["Decision Tree", ] = c(mean_accuracy, mean_precision, mean_recall, mean_f1)


# Fit decision tree on full data and plot to understand important variables
tree = tree(Diagnosis ~ ., data = df)
plot(tree)
text(tree, pretty = 0)


# Random Forest ----------------------------------------------------------

# Create function for the four accuracy metrics
four_metrics = function(data, lev = NULL, model = NULL) {
  cm = confusionMatrix(data$pred, data$obs, positive = "Yes")
  c(Accuracy  = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall    = as.numeric(cm$byClass["Recall"]),
    F1        = as.numeric(cm$byClass["F1"]))
}

# Set random seed
set.seed(1)

# Create 10-fold CV object
ctrl = trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = four_metrics)

# 10-fold CV for the Random Forest model
rf_model = train(Diagnosis ~ ., 
                  data = df,
                  method = "rf",
                  trControl = ctrl,
                  tuneLength = 5,
                  metric = "F1")

# Examine results for various mtry values
rf_model$results[, c("mtry", "Accuracy", "Precision", "Recall", "F1")]

# Identify row with best F1 score and manually enter accuracy metrics
best_rf = rf_model$results[which.max(rf_model$results$F1), c("Accuracy", "Precision", "Recall", "F1")]
rf_accuracy = 0.9511247
rf_precision = 0.9469676
rf_recall = 0.9143169
rf_f1 = 0.9293707

# Add accuracy metrics to data frame
accuracy_df["Random Forest", ] = c(rf_accuracy, rf_precision, rf_recall, rf_f1)

# Create RF model on full data set and find/plot variable importance
rf_model = randomForest(Diagnosis ~ ., data = df, importance = TRUE)
importance(rf_model)
varImpPlot(rf_model)


# Gradient Boosting ----------------------------------------------------------

# Set random seed
set.seed(1)

# Create 10-fold CV object
ctrl = trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = four_metrics)

# 10-fold CV for the Gradient Boosting model
gbm_model = train(Diagnosis ~ ., 
                   data = df,
                   method = "gbm",
                   trControl = ctrl,
                   verbose = FALSE,       
                   tuneLength = 5,        
                   metric = "F1")

# Examine the Gradient Boosting Results
gbm_model$results[, c("n.trees", "interaction.depth", "shrinkage", "Accuracy", "Precision", "Recall", "F1")]

# Identify row with best F1 score and manually enter accuracy metrics
best_gbm = gbm_model$results[which.max(gbm_model$results$F1), c("Accuracy", "Precision", "Recall", "F1")]
gbm_accuracy = 0.9505399
gbm_precision = 0.9495308
gbm_recall = 0.9093716
gbm_f1 = 0.9283453

# Add accuracy metrics to data frame
accuracy_df["Gradient Boosting", ] = c(gbm_accuracy, gbm_precision, gbm_recall, gbm_f1)

# XGBoost ----------------------------------------------------------

# Set random seed
set.seed(1)

# Create 10-fold CV object
ctrl = trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = four_metrics)

# Enable parallel processing for faster model training
cl = makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# 10-fold CV for the XGBoost model
xgb_model = train(Diagnosis ~ ., 
                   data = df,
                   method = "xgbTree",
                   trControl = ctrl,
                   verbose = FALSE,
                   tuneLength = 5,  
                   metric = "F1")

# End parallel processing
stopCluster(cl)
registerDoSEQ()

# Identify row with best F1 score and manually enter accuracy metrics
best_xgb = xgb_model$results[which.max(xgb_model$results$F1), 
                              c("Accuracy", "Precision", "Recall", "F1")]
xgb_accuracy = 0.953447
xgb_precision = 0.9443295
xgb_recall = 0.9241803
xgb_f1 = 0.9333637

# Add accuracy metrics to data frame
accuracy_df["XGBoost", ] = c(xgb_accuracy, xgb_precision, xgb_recall, xgb_f1)

# Get variable importance
xgb_importance = varImp(xgb_model)
print(xgb_importance)

# Plot top variables
plot(xgb_importance, top = 20)
