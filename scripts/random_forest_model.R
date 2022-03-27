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

# For this recipe I'll do some pretty simple pre-processing. Random Forests dont need much.
# According to the ranger docs, dummy variables aren't even required.
#   - ranger docs: https://parsnip.tidymodels.org/reference/details_rand_forest_ranger.html
# The biggest part is the step_impute_mean. I figured this was a fairly harmless way to input
# missing data for what few missing values there are. A k-nearest-neighbors approach
# could also have been used but I want to avoid using techniques I don't understand when I can avoid it.
rf_recipe <-
    recipe(Churn ~ ., data = telco_train) %>%
    update_role(customerID, new_role = "id") %>%
    themis::step_upsample(Churn, over_ratio = 1) %>%          # Set number of samples for 'yes' and 'no' equal
    step_impute_mean(TotalCharges)                            # Use statistical imputation to handle missing values
    
rf_model <- 
    rand_forest(trees = 100) %>%
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

# Model tuning ----
# First we need to know what our model hyper-parameters even are. 
show_model_info("rand_forest")

# Specify what model parameters to tune
tune_spec <-
    rand_forest(
        mtry = tune(),
        trees = tune(),
        min_n = tune()
    ) %>%
    # Set importance to permutation for variable importance plots later
    set_engine("ranger", importance = "permutation") %>%
    set_mode("classification")

# set up a grid for re sampling
tree_grid <- 
    grid_regular(
        mtry() %>% range_set(c(1,10)),
        trees() %>% range_set(c(50,300)),
        min_n() %>% range_set(c(2,10)),
        levels = 3
    )

tuned_wf <-
    workflow() %>%
    add_model(tune_spec) %>%
    add_recipe(rf_recipe)

tictoc::tic()
tree_results <-
    tuned_wf %>%
    tune_grid(
        resamples = telco_folds,
        grid = tree_grid,
        metrics = metric_set(f_meas, roc_auc, accuracy, specificity),
        control = control_resamples(save_pred = TRUE)
    )
tictoc::toc()

# Compare metrics based on number of trees and mtry
tree_results %>% 
    collect_metrics() %>%
    mutate(trees = as.factor(trees)) %>%
    ggplot(aes(mtry, mean, color = trees)) +
    geom_line(size = 1.5, alpha = 0.6) +
    geom_point(size = 2) +
    facet_wrap(~ .metric, nrow = 2) +
    scale_color_viridis_d(option = "plasma", begin = .9, end = 0)
    
# Based on the plot, it appears that the sweetspot to maximize f1 score without
# sacrificing everything else is to shoot for 5 mtry. Specificity appears to simply tank regardless
# as we increase the size of mtry.

# Based on observations from the plot, we'll choose the tree with the best f1 score.
# That appears to be the best balance.

best_tree <- tree_results %>%
    select_best("f_meas")

# Now we finalize, or update, the workflow 
final_wf <-
    tuned_wf %>%
    finalize_workflow(best_tree)

# This fits the model to the full training set and evaluates the 
# finalized model on the testing data
final_fit <-
    final_wf %>%
    last_fit(telco_split)

# Show collected metrics for the final fit
sink(file = "rf_res.txt")
final_fit %>%
    collect_metrics()
sink(file = NULL)

# Show predictions from the final fit
final_fit %>%
    collect_predictions() %>%
    head()


# Create an ROC curve for our final fit
final_fit %>%
    collect_predictions() %>%
    roc_curve(Churn, .pred_No) %>%
    autoplot()

# Based on the ROC we can see that our data doesn't over or under-fit the data.

# Variable Importance Plots
final_fit %>%
    extract_workflow() %>%
    extract_fit_parsnip() %>%
    vip::vip()
