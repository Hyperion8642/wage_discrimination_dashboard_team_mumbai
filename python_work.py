"""
Q1 Code
"""
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.formula.api as smf

"""Reading & exploring null values: """
# df = pd.read_csv('salary.txt', header = True, sep='')
df = pd.read_csv('salary.txt', sep=r'\s+')
df.head()
# df = read.csv('salary.txt', header = TRUE, sep = '')

"""Exploring null values: """
df.isna().sum()

"""Create the side-by-side boxplot: """

plt.figure(figsize=(12, 6))

sns.boxplot(data=df_1995, x='field', y='salary', hue = 'sex', palette='viridis')

# Add titles and labels for clarity
plt.title('Salary Distribution by Field', fontsize=15)
plt.xlabel('Academic/Professional Field', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)

# If your field names are long, rotate the x-labels
plt.xticks(rotation=45)

plt.show()

"""
Data Filtering: 
"""
df_m = df[df['sex'] == 'M']
df_f = df[df['sex'] == 'F']


# Multiple Linear Regression (MLR) ANOVA: model with vs without sex
# predictors: degree, year, field, rank, admin
starting_df = starting_df.dropna(subset=['salary','deg','yrdeg','field','rank','admin','year'])

model_no_sex = ols('salary ~ deg + yrdeg + C(field) + C(rank) + admin + year', data=starting_df).fit()
model_with_sex = ols('salary ~ deg + yrdeg + C(field) + C(rank) + admin + year + C(sex)', data=starting_df).fit()

anova_results = sm.stats.anova_lm(model_no_sex, model_with_sex)
f_stat = anova_results['F'][1]
p_val_anova = anova_results['Pr(>F)'][1]

# Print results
print("\n=== Q1: Starting Salary Analysis ===")
print(f"Mean Salary (Men): {np.mean(men_start):.2f}")
print(f"Mean Salary (Women): {np.mean(women_start):.2f}")
print(f"Mean Difference (Men - Women): {mean_diff:.2f}")
print(f"Welch t-test: t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4e}")
print(f"\nANOVA (MLR: model with vs without sex): F-statistic = {f_stat:.4f}, p-value = {p_val_anova:.4f}")






"""
Q2 Code
"""









"""
Q3 Code
"""









"""Attempting to Find Confounders"""

# List of all potential confounders
controls = ['id','deg', 'yrdeg', 'field', 'startyr', 'year', 'rank', 'admin']

# 1. The "Gold Standard" Full Model
full_formula = 'salary ~ sex + ' + ' + '.join(controls)
m_full = smf.ols(full_formula, data=df).fit(cov_type='HC3')

# Get the 'sex' coefficient name (e.g., 'sex[T.Male]')
sex_param = [c for c in m_full.params.index if 'sex' in c][0]
full_beta = m_full.params[sex_param]

# 2. Iterate and "Leave One Out"
ovb_results = []

for var in controls:
    # Build formula excluding only the current variable
    reduced_controls = [c for c in controls if c != var]
    reduced_formula = 'salary ~ sex + ' + ' + '.join(reduced_controls)
    
    m_reduced = smf.ols(reduced_formula, data=df).fit(cov_type='HC3')
    reduced_beta = m_reduced.params[sex_param]
    
    # Calculate the "Bias" that this variable was correcting
    # If delta is large, this variable was a major confounder
    delta = full_beta - reduced_beta
    percent_impact = (delta / reduced_beta) * 100
    
    ovb_results.append({'Omitted Variable': var, 'Sex Beta (Without it)': reduced_beta, 'Shift (%)': percent_impact})

ovb_df = pd.DataFrame(ovb_results).sort_values(by='Shift (%)', ascending=False)
print(ovb_df)