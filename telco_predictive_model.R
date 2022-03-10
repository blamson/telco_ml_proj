library(dplyr)
library(tidymodels)
library(yardstick)

# Source of Guides --------------------------------------------------------------------------------
# This methodology strictly follows the advice provided in Chapter 10 of Modern R for Data Science
# Link: https://mdsr-book.github.io/mdsr2e/ch-modeling.html

# Vocabulary --------------------------------------------------------------------------------------
## Churn: Percentage of customers who stopped using a companies product

# Prepping Data -----------------------------------------------------------------------------------

# Read in data
telco_data <- readr::read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv") %>%
    dplyr::mutate(Churn = as.factor(Churn))

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


# Create the null model
mod_null <- logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(Churn ~ 1, data = train)

# Create linear model

mod_log <- logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(Churn ~ ., data = train)