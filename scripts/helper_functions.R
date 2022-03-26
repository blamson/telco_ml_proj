# Load in libraries -------------------------------------------------------------------------------
load_libraries <- function() {
    library(dplyr)      # For all general data work
    library(tidymodels) # For everything models
    library(yardstick)  # For model performance evaluation
    library(rsample)    # For bootstrapping and sampling purposes
    library(parsnip)    # For a standardized modeling interface
    library(themis)     # To handle unbalanced data
    library(ranger)     # for random forests
    
    tidymodels_prefer() # To prefer tidymodels functions given conflicts
}

# Using preferred re-sampling methods, provides metrics and confusion matrix ----------------------
resample_and_metrics <- function(my_wflow, my_resample, my_control, my_metrics) {
    test <-
        my_wflow %>%
        fit_resamples(
            resamples = my_resample, 
            control = my_control,
            metrics = my_metrics
        )
        
    test %>%
        collect_metrics() %>%
        print()
    
    test %>%
        conf_mat_resampled() %>%
        print()
}
