library(ggplot2)
library(dplyr)
library(tidymodels)
library(yardstick)
library(rsample)
library(parsnip)
library(GGally)

tidymodels_prefer()

# Read in data ----
telco_data <- readr::read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Set seed for reproducibility ----
set.seed(500)

# Split data ----
telco_split <- initial_split(telco_data, prop = .8, strata = Churn)
telco_train <- training(telco_split)
telco_test <- testing(telco_split)

# Set up recipe ----
telco_recipe <-
    recipe(Churn ~ ., data = telco_train) %>%
    update_role(customerID, new_role = "id") %>%
    step_mutate(SeniorCitizen = as.factor(SeniorCitizen)) %>%
    step_dummy(all_nominal_predictors()) %>%
    # Remove highly correlated variables ----
    step_corr(all_numeric_predictors())

# Set up model ----
telco_model <- 
    logistic_reg(mode = "classification") %>%
    set_engine("glm")

# Set up workflow ----
telco_workflow <- 
    workflow() %>%
    add_model(telco_model) %>%
    add_recipe(telco_recipe)

# Bootstrap the training data 5 times ----
telco_bstraps <-
    bootstraps(
        telco_train,
        times = 5,
        strata = Churn
    )

# Prediction on each of the 5 samples ----
telco_resamples <-
    telco_workflow %>%
    fit_resamples(
        resamples = telco_bstraps,
        metrics = metric_set(accuracy)
    )

# Metrics for each sample ----
telco_resamples %>%
    collect_metrics()

