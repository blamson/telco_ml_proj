# In this script ill be exploring a random forest approach to the problem
# I'll be using a lot of advice from this chapter
# https://www.tmwr.org/resampling.html

load_libraries()

# Read in data ----
telco_data <- readr::read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Set initial seed for reproducibility ----
set.seed(501)

# Split data ----
telco_split <- initial_split(telco_data, prop = .8, strata = Churn)
telco_train <- training(telco_split)
telco_test <- testing(telco_split)

# For this recipe I'll do some pretty simple pre-processing. 
# The biggest part is the step_impute_mean. I figured this was a fairly harmless way to input
# missing data for what few missing values there are. A k-nearest-neighbors approach
# could also have been used but I want to avoid using techniques I don't understand when I can avoid it.
rf_recipe <-
    recipe(Churn ~ ., data = telco_train) %>%
    update_role(customerID, new_role = "id") %>%
    step_upsample(Churn, over_ratio = 1) %>%                  # Set number of samples for 'yes' and 'no' equal
    step_mutate(SeniorCitizen = as.factor(SeniorCitizen)) %>%
    step_impute_mean(TotalCharges) %>%                        # Use statistical imputation to handle missing values
    step_dummy(all_nominal_predictors())
    
rf_model <- 
    rand_forest(trees = 1000) %>%
    set_engine("ranger") %>%
    set_mode("classification")

rf_wflow <-
    workflow() %>%
    add_recipe(rf_recipe) %>%
    add_model(rf_model)

# Test using cross-validation
my_metrics <- metric_set(accuracy, sensitivity, specificity, precision, recall, npv, ppv, f_meas, roc_auc)
telco_folds <- vfold_cv(telco_train, v = 10)
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

# Test and show metrics
resample_and_metrics(my_wflow = rf_wflow, my_resample = telco_folds, my_control = keep_pred, my_metrics = my_metrics)

# What we can see is that we miss a LOT of the customers who churn. We also have the same problem as our logistic 
# model, in that we're guessing 'Yes' a bit too aggressively. Let's see what we can do about that.

