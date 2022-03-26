library(ggplot2)

# First we want to check out the data
skimr::skim(telco_data)

# We can see from this that only one column has missing values, TotalCharges. This will be important for the random
# forest method, as that requires some pre-processing strategies to handle NA's such as statistical imputation.
# The little histograms in the output also show a non-normal distribution for all of our numeric data
# We'll look at that more closely with the plots below. 

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