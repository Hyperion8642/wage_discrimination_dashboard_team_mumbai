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
df_1995 = df[df['year'] == 95]
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

starting_df = df[df['year'] == df['startyr']]
df_same_salary = df[df['startyr'] == df['year']]


"""Multiple Linear Regression (MLR) ANOVA: model with vs without sex:
predictors: degree, year, field, rank, admin"""
starting_df = starting_df.dropna(subset=['salary','deg','yrdeg','field','rank','admin','year'])

model_no_sex = ols('salary ~ deg + yrdeg + C(field) + C(rank) + admin + year', data=starting_df).fit()
model_with_sex = ols('salary ~ deg + yrdeg + C(field) + C(rank) + admin + year + C(sex)', data=starting_df).fit()

anova_results = sm.stats.anova_lm(model_no_sex, model_with_sex)
f_stat = anova_results['F'][1]
p_val_anova = anova_results['Pr(>F)'][1]

men_start = starting_df[starting_df["sex"] == "M"]["salary"]
women_start = starting_df[starting_df["sex"] == "F"]["salary"]
mean_diff = np.mean(men_start) - np.mean(women_start)
t_stat, p_val_t = stats.ttest_ind(men_start, women_start, equal_var=False)

# Print results
print("\n=== Q1: Starting Salary Analysis ===")
print(f"Mean Salary (Men): {np.mean(men_start):.2f}")
print(f"Mean Salary (Women): {np.mean(women_start):.2f}")
print(f"Mean Difference (Men - Women): {mean_diff:.2f}")
print(f"Welch t-test: t-statistic = {t_stat:.4f}, p-value = {p_val_t:.4e}")
print(f"\nANOVA (MLR: model with vs without sex): F-statistic = {f_stat:.4f}, p-value = {p_val_anova:.4f}")

"""Wald Test"""
mls_model_same_year = smf.ols(formula='salary ~ sex + deg + yrdeg + field + startyr + ' \
'year + rank + admin', data=df_same_salary).fit(cov_type='HC3')
robust_wald_same_year = mls_model_same_year.wald_test_terms()
print(robust_wald_same_year)#


"""MLS Approach - Base Model: """
mls_model_same_year = smf.ols(formula='salary ~ sex + deg + yrdeg + field + ' \
'year + rank + admin', data=df_same_salary)
# Handle unequal variance
results_mls_same_year = mls_model_same_year.fit(cov_type='HC3')
print(results_mls_same_year.summary())

"""MLS Approach - Interaction Model: """
formula = 'salary ~ sex + sex * deg + sex * yrdeg + sex * field + sex * year + sex * rank + sex * admin'
mls_model_same_year_interact = smf.ols(formula=formula, data=df_same_salary)
# Handle unequal variance
results_mls_same_year_interact = mls_model_same_year_interact.fit(cov_type='HC3')
print(results_mls_same_year_interact.summary())


"""Permutation Test: """
df_start_year_sal = df[df['startyr'] == df['year']]
# 1. Calculate the actual observed difference
obs_mean_m = df_start_year_sal[df_start_year_sal['sex'] == 'M']['salary'].mean()
obs_mean_f = df_start_year_sal[df_start_year_sal['sex'] == 'F']['salary'].mean()
observed_diff = obs_mean_m - obs_mean_f

# 2. Setup for the simulation
n_iterations = 10000
perm_diffs = np.zeros(n_iterations)
salaries = df_start_year_sal['salary'].values
n_males = len(df_start_year_sal[df_start_year_sal['sex'] == 'M'])

# 3. The Loop
for i in range(n_iterations):
    # Shuffle the salaries
    shuffled_salaries = np.random.permutation(salaries)
    
    # Assign the first 'n_males' to Group A, the rest to Group B
    m_perm_mean = shuffled_salaries[:n_males].mean()
    f_perm_mean = shuffled_salaries[n_males:].mean()
    
    # Record the difference
    perm_diffs[i] = m_perm_mean - f_perm_mean
    # Create the histogram
plt.hist(perm_diffs, bins=30, color='skyblue', edgecolor='black')
plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label='Observed Diff')
plt.title('Permutation Distribution of Salary Differences')
plt.xlabel('Difference in Means (Male - Female)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculate the P-value
# (The proportion of random diffs that are as extreme or more extreme than ours)
p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_iterations
print(f"Permutation P-value: {p_value}")


"""
Q3 Code
"""
"""
Correlation Matrix: 
"""
df_salary_encoded = pd.get_dummies(df[['salary','sex', 'deg', 'yrdeg', 'id',
                                        'field', 'startyr','year', 'rank','admin']], 
                                        columns=['sex', 'deg', 'field', 'rank','admin'], 
                                        dtype=int,drop_first=True)
corr_matrix_full= df_salary_encoded.corr(numeric_only=True)
sns.heatmap(corr_matrix_full, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1);

"""MLR Models"""
"""Base Model"""
# Initial results failed to account for repititions resulting from id. Not much of a change after accounting for it though
mls_model_whole = smf.ols(formula='salary ~ sex + id + deg + yrdeg + field + startyr + ' \
'year + rank + admin', data=df)
# Handle unequal variance
results_mls_whole = mls_model_whole.fit(cov_type='HC3')
print(results_mls_whole.summary())

"""Interaction Model"""
formula = 'salary ~ sex * deg + sex * yrdeg + sex * field ' \
'+ sex * year + sex * startyr + sex * rank + sex * admin + sex * id'

mls_model_whole_interact = smf.ols(formula=formula, data=df)
# Handle unequal variance
results_mls_whole_interact = mls_model_whole_interact.fit(cov_type='HC3')
print(results_mls_whole_interact.summary())

"""ANOVA Model"""
mls_model_whole_no_sex = ols(formula='salary ~ id + deg + yrdeg + field + startyr + ' \
'year + rank + admin', data=df).fit()
mls_model_whole = ols(formula='salary ~ sex + id + deg + yrdeg + field + startyr + ' \
'year + rank + admin', data=df).fit()

# ANOVA: all factors vs sex
anova_results_whole = sm.stats.anova_lm(mls_model_whole_no_sex, mls_model_whole)
f_stat_anova_whole = anova_results_whole['F'][1]
p_val_anova_whole = anova_results_whole['Pr(>F)'][1]
print(f_stat_anova_whole)
print(p_val_anova_whole)
# print(anova_results.summary())

"""Wald Test"""
mls_model_whole = smf.ols(formula='salary ~ sex + id + deg + yrdeg + field + startyr + ' \
'year + rank + admin', data=df).fit(cov_type='HC3')
robust_anova_whole = mls_model_whole.wald_test_terms()

print(robust_anova_whole)

"""Attempting to Find Confounders: 
Looping over removing one variable at a time and seeing how the coefficient of sex shifts
"""

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