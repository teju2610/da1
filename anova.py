************One-Way ANOVA****************

import pandas as pd
from scipy.stats import f_oneway
group_A = [23, 25, 27, 24, 22]
group_B = [30, 31, 29, 32, 28]
group_C = [35, 34, 36, 33, 37]
f_stat, p_val = f_oneway(group_A, group_B, group_C)
print("One-Way ANOVA")
print("F-Statistic:", f_stat)
print("P-Value:", p_val)
if p_val < 0.05:
    print("Result: Significant difference between group means.")
else:
    print("Result: No significant difference.")

************* Two-Way ANOVA****************
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
data = {
    'Score': [85, 90, 88, 75, 78, 74, 92, 95, 93, 70, 68, 72],
    'Gender': ['M', 'M', 'M', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F'],
    'Course': ['Math', 'Math', 'Math', 'Math', 'Math', 'Math',
               'Bio', 'Bio', 'Bio', 'Bio', 'Bio', 'Bio']
}
df = pd.DataFrame(data)
model = ols('Score ~ C(Gender) + C(Course) + C(Gender):C(Course)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nTwo-Way ANOVA")
print(anova_table)