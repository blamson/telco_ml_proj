library(ggplot2)    # Plotting
library(dplyr)      # For all general data work
library(tidymodels) # For everything models
library(yardstick)  # For model performance evaluation
library(rsample)    # For bootstrapping and sampling purposes
library(parsnip)    # For a standardized modeling interface
library(themis)     # To handle unbalanced data

tidymodels_prefer() # To prefer tidymodels functions given conflicts

# Read in data ----
telco_data <- readr::read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic data analysis ----
# Count number of customers who do and don't churn
telco_data %>%
    dplyr::group_by(Churn) %>%
    dplyr::summarise(
        count = n(),
        percent = n() / nrow(telco_data) * 100
    )

# Initial note: What we can see here is that only around a quarter of the customers in this data set churn.
# This is important, as certain metrics become less useful due to this. Accuracy, for instance,
# isn't particularly helpful for determining if we're capturing all of the customers that do churn
# as that number will be outweighed by the number of those who don't. 
# I'll need to be careful about which metrics to prioritize maximizing. 

# Update note: Note that this can be potentially side-stepped with the themis package which can help
# account for unbalanced data. I'll be utilizing that in this model.

telco_data %>%
    ggplot(aes(x = MonthlyCharges)) +
    geom_histogram(bins = 30, color = "white")

telco_data %>%
    ggplot(aes(x = TotalCharges)) +
    geom_histogram(bins = 30, color = "white")

telco_data %>%
    ggplot(aes(x = tenure)) +
    geom_histogram(bins = 30, color = "white")

    
# From the plots we can see very non-normal distributions for both types of charges.
# We'll want to log transform this data to more easily allow our model to work on it.

# Set seed for reproducibility ----
set.seed(500)

# Split data ----
# Strata is set to churn due to the disproportionate nature of the variable. 75% No, 25% Yes.
# I want to ensure that each sample gets enough of each value. 
telco_split <- initial_split(telco_data, prop = .8, strata = Churn)
telco_train <- training(telco_split)
telco_test <- testing(telco_split)

# Initial attempt! --------------------------------------------------------------------------------
# Set up recipe ----
telco_recipe_init <-
    recipe(Churn ~ ., data = telco_train) %>%
    update_role(customerID, new_role = "id") %>%
    step_mutate(SeniorCitizen = as.factor(SeniorCitizen)) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_corr(all_numeric_predictors(), threshold = 0.7)  # Remove highly correlated variables

# Set up model ----
telco_model <- 
    logistic_reg(mode = "classification") %>%
    set_engine("glm")

# Set up workflows ----
telco_workflow_init <- 
    workflow() %>%
    add_model(telco_model) %>%
    add_recipe(telco_recipe_init)

# Bootstrap the training data 5 times ----
telco_bstraps <-
    bootstraps(
        telco_train,
        times = 5,
        strata = Churn
    )

# Prediction on each of the 5 samples ----
telco_resamples_init <-
    telco_workflow_init %>%
    fit_resamples(
        resamples = telco_bstraps,
        # List off metrics I'm interested in capturing
        metrics = metric_set(accuracy, sensitivity, specificity, precision, recall, npv, ppv, f_meas),
        # Save predictions for confusion matrix
        control = control_resamples(save_pred = TRUE)
    )

# Overall metrics for the samples ----
telco_resamples_init %>%
    collect_metrics()

# Confusion matrix for re-samples ----
telco_resamples_init %>%
    conf_mat_resampled()

# Based on the metrics and matrix above, we have a big problem with the initial model. The most important metric
# for this problem is identifying customers who churn, which, according to specificity, is horrible. 
# The current model only correctly identifies around 55% of the customers who churn. 

# 2nd attempt! ------------------------------------------------------------------------------------

# Updated recipe ----
telco_recipe_updated <- 
    recipe(Churn ~ ., data = telco_train) %>%
    step_upsample(Churn, over_ratio = 1) %>%                  # Set number of samples for 'yes' and 'no' equal
    update_role(customerID, new_role = "id") %>%
    step_mutate(SeniorCitizen = as.factor(SeniorCitizen)) %>%
    step_naomit(everything(), skip = TRUE) %>%
    step_log(tenure, MonthlyCharges, TotalCharges) %>%        # log transform skewed numeric variables
    step_normalize(tenure, MonthlyCharges, TotalCharges) %>%  # z-standardize all numeric variables
    step_dummy(all_nominal_predictors()) %>%
    step_corr(all_numeric_predictors(), threshold = 0.7) %>%  # Remove highly correlated variables
    step_zv(all_numeric_predictors())                         # Remove numeric variables with zero variance

# Workflow using updated recipe ----
telco_workflow_updated <-
    workflow() %>%
    add_model(telco_model) %>%
    add_recipe(telco_recipe_updated)

telco_resamples_updated <-
    telco_workflow_updated %>%
    fit_resamples(
        resamples = telco_bstraps,
        # List off metrics I'm interested in capturing
        metrics = metric_set(accuracy, sensitivity, specificity, precision, recall, npv, ppv, f_meas),
        # Save predictions for confusion matrix
        control = control_resamples(save_pred = TRUE)
    )

# Overall metrics for the samples ----
telco_resamples_updated %>%
    collect_metrics()

# Confusion matrix for re-samples ----
telco_resamples_updated %>%
    conf_mat_resampled()

# Updating the recipe to better handle some of the quirks of the data set helps and hurts us a bit. 
# Comparing the two, we take a hit to accuracy, f_meas, npv, recall and sensitivity.
# On the other hand, everything else improves, especially specificity which we largely care about.
# We flip the type of error we're running into though. This model in general predicts that far more customers
# churn, which by the nature of that approach captures more of those that do. The issue is that we also
# capture many customers that don't. Within the 'yes' prediction, it's a near 50/50 split on whether the guess
# was correct or not. This should be improved. 
