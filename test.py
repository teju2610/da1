******One-Sample T-Test*********

from scipy.stats import ttest_1samp

# Sample data
scores = [86, 87, 88, 86, 87, 85, 90, 89]

t_stat, p_val = ttest_1samp(scores, 85)

print("One-Sample T-Test")
print("T-Statistic:", t_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Result: Mean is significantly different from 85.")
else:
    print("Result: No significant difference.")

******Two-Sample T-Test (Unpaired / Independent)******

from scipy.stats import ttest_ind

# Sample data for two groups
group1 = [82, 85, 88, 90, 86]
group2 = [75, 80, 78, 74, 76]

# Unpaired T-test
t_stat, p_val = ttest_ind(group1, group2)

print("\nTwo-Sample T-Test (Unpaired)")
print("T-Statistic:", t_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Result: Significant difference between the two groups.")
else:
    print("Result: No significant difference.")

**********Paired T-Test (Dependent)*********

from scipy.stats import ttest_rel

# Sample before and after scores of same students
before = [72, 75, 78, 79, 80]
after  = [74, 78, 79, 82, 85]

# Paired T-test
t_stat, p_val = ttest_rel(before, after)

print("\nPaired T-Test")
print("T-Statistic:", t_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Result: Significant difference before and after.")
else:
    print("Result: No significant difference.")

_______________________________________________________________
6)

**********Chi-Square Test of Independence********
import pandas as pd
from scipy.stats import chi2_contingency

# Sample contingency table (Gender vs Preference)
data = {'Tea': [30, 10], 'Coffee': [20, 40]}
table = pd.DataFrame(data, index=['Male', 'Female'])

chi2, p, dof, expected = chi2_contingency(table)

print("Chi-Square Test of Independence")
print("Chi2 Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

if p < 0.05:
    print("Result: Variables are dependent.")
else:
    print("Result: Variables are independent.")

************Chi-Square Goodness of Fit**********

from scipy.stats import chisquare

# Observed frequencies (e.g., dice rolls)
observed = [18, 22, 20, 19, 21, 20]

# Expected frequencies (assuming fair dice)
expected = [20] * 6

chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)

print("\nChi-Square Goodness of Fit Test")
print("Chi2 Statistic:", chi2_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("Result: Observed data does NOT fit expected distribution.")
else:
    print("Result: Data fits expected distribution.")