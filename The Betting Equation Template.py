# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:51:34 2023

@author: adamt
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t

# Define the logistic function
def logistic_function(x, a, b):
    return 1 / (1 + a * x**b)


# Read the CSV file
# Enter your file directory between the quotation marks below
file_path = r""
dataframe = pd.read_csv(file_path)

# Create a new column "closing_odds" based on the "B365CH" column
dataframe['closing odds'] = dataframe['B365CH'] - 1
X = dataframe['closing odds']

# Create a new column "Outcome" based on the conditions
dataframe['Outcome'] = np.where(dataframe['FTHG'] > dataframe['FTAG'], 1, 0)
Y = dataframe['Outcome']

# Save the modified DataFrame to a new CSV file using the file directory and file name of your choosing
dataframe.to_csv(r"", index=False)

# Fit the logistic function to the data
params, cov_matrix = curve_fit(logistic_function, X, Y)

# Estimated parameters
a, b = params
print(f"Estimated parameters: a = {a}, b = {b}")

# Calculate standard errors from the covariance matrix
std_errors = np.sqrt(np.diag(cov_matrix))
std_error_a, std_error_b = std_errors
print(f"Standard errors: a = {std_error_a}, b = {std_error_b}")

# Calculate t-statistics and p-values
null_a = 1
null_b = 1
t_stat_a = (a - null_a) / std_error_a
t_stat_b = (b - null_b) / std_error_b
df = len(X) - 2  # Degrees of freedom
p_value_a = 2 * (1 - t.cdf(np.abs(t_stat_a), df))
p_value_b = 2 * (1 - t.cdf(np.abs(t_stat_b), df))
print(f"t-statistics: a = {t_stat_a}, b = {t_stat_b}")
print(f"p-values: a = {p_value_a}, b = {p_value_b}")

# Calculate 95% confidence intervals
alpha = 0.05
t_critical = t.ppf(1 - alpha / 2, df)
ci_a = (a - t_critical * std_error_a, a + t_critical * std_error_a)
ci_b = (b - t_critical * std_error_b, b + t_critical * std_error_b)
print(f"95% confidence intervals:")
print(f"a: {ci_a}")
print(f"b: {ci_b}")


