library(dplyr)
library(tidymodels)

# Vocabulary --------------------------------------------------------------------------------------
## Churn: Percentage of customers who stopped using a companies product

# Prepping Data -----------------------------------------------------------------------------------

# Read in data
telco_data <- readr::read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

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
# I first convert the churn column to boolean values
# I then get the counts of TRUE and FALSE before creating
# a column of TRUE FALSE proportions, pct. 
# pi_bar itself is assigned the proportion of TRUE values.
pi_bar <- train %>%
    dplyr::mutate(Churn = Churn == "Yes") %>%
    dplyr::count(Churn) %>%
    dplyr::mutate(pct = n / sum(n)) %>% 
    dplyr::filter(Churn == TRUE) %>%
    dplyr::pull(pct)

glue::glue("The estimated churn rate of the sample is approximately {pi_bar %>% round(digits=3)}")
