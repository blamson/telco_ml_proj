library(dplyr)
library(tidymodels)
library(yardstick)
library(rsample)
library(parsnip)

# Source of Guides --------------------------------------------------------------------------------
# This methodology strictly follows the advice provided in Chapter 10 of Modern R for Data Science
# Link: https://mdsr-book.github.io/mdsr2e/ch-modeling.html

# Vocabulary --------------------------------------------------------------------------------------
## Churn: Percentage of customers who stopped using a companies product

# Prepping Data -----------------------------------------------------------------------------------

# Read in data
# Convert majority of columns to factor for ease of use
telco_data <- readr::read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv") %>%
    dplyr::mutate(
        dplyr::across(
            where(is.character), as.factor
        ),
        customerID = customerID %>% as.character()
    )

# Set consistent seed
set.seed(364)

# Get the number of rows of data for future use
n <- nrow(telco_data)

# Split data into training and testing data sets using an 80/20 split
telco_parts <- telco_data %>%
    rsample::initial_split(prop = 0.8)

train <- telco_parts %>%
    rsample::training()

test <- telco_parts %>%
    rsample::testing()

# Compute estimated churn rate of customers -------------------------------------------------------
# I get the counts of each Churn factor level (yes / no)
# Afterwards I create a new variable, pct, that gets the percentage of each level
# I filter out only the row with the "Yes" level and pull out that percentage
# That pulled percentage is assigned to pi_bar.
pi_bar <- train %>%
    dplyr::count(Churn) %>%
    dplyr::mutate(pct = n / sum(n)) %>% 
    dplyr::filter(Churn == "Yes") %>%
    dplyr::pull(pct)

glue::glue("The estimated churn rate of the training sample is approximately {pi_bar %>% round(digits=3)}")

# Null Model ---------------------------------------------------
# Create the null model
mod_null <- logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(Churn ~ 1, data = train)

# Compute accuracy of null model for future comparisons
pred <- train %>%
    dplyr::bind_cols(
        predict(mod_null, new_data = train, type = "class")
    ) %>% 
    dplyr::rename(Churn_null = .pred_class)

# Method 1 ----
yardstick::accuracy(data = pred, Churn, Churn_null)

# Method 2 ----
confusion_null <- pred %>%
    yardstick::conf_mat(truth = Churn, estimate = Churn_null)
confusion_null

# Linear Model -------------------------------------------------

# Remove customerID from train dataset
train_no_id <- train %>%
    dplyr::select(-customerID)

# Create linear model using all columns as predictors
mod_log <- logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(Churn ~ ., data = train_no_id)

# Compute accuracy of model ----
pred <- pred %>%
    dplyr::bind_cols(
        predict(mod_log, new_data = train, type = "class")
    ) %>%
    dplyr::rename(Churn_log = .pred_class)

confusion_log <- pred %>%
    yardstick::conf_mat(truth = Churn, estimate = Churn_log)

confusion_log

yardstick::accuracy(data = pred, Churn, Churn_log)
