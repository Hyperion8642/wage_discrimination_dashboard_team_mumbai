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










"""
Q2 Code
"""









"""
Q3 Code
"""