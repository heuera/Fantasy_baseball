import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FOR OUTPUT FORMAT PYCHARM
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# EXPLORATORY ANALYSIS -------------------------------------------------------------------------------------------------

# Load in dataset
df_pitchers = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchers2000_2019.csv")

'''
# Look at the distributions
vars = ['ERA', 'CG', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'FBv', 'SIERA', 'WHIP']
for var in vars:
    var = str(var)
    plt.hist(df_pitchers[var], bins=50)
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.axvline(df_pitchers[var].mean(), color='k', linestyle='dashed', linewidth=1)
    #plt.show()

# Visualize variables by year
sns.set(style='whitegrid')
vars = ['W', 'ERA', 'IP', 'ER', 'HR', 'SO', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'WHIP', 'BABIP', 'SIERA', 'Hard%', 'FBv']
for var in vars:
    var = str(var)
    ax = sns.boxplot(x="Season", y=var, data=df_pitchers)
    #plt.show()

# Create correlation matrix for variables
corr_matrix = df_pitchers.corr()

# Look specifically at correlation between points and number of points scored in the following year (points_lead)
points_corr = df_pitchers['Points'].corr(df_pitchers['Points_lead'])
#print(points_corr)

# Is there evidence of correlation between how a player does in the current year and how they do in the following year?
    # null: pearson r = 0 (ie. no correlation)
pear_r = pearsonr(df_pitchers['Points'], df_pitchers['Points_lead'])
    # Pearson r = 0.59 and p < 0.05, so current year performance does seem to be strongly associated with the following year performance

# Visualize Points vs Points_lead
sns.scatterplot(data=df_pitchers, x="Points", y='Points_lead', hue="Season", s=10)
'''

df_keystats = df_pitchers.drop(['Name', 'Team', 'G', 'GS', 'CG', 'ShO', 'SV', 'HLD', 'BS', 'IBB', 'HBP',
       'WP', 'BK', 'LOB%', 'ERA-', 'FIP-', 'xFIP-', 'GB/FB', 'RS', 'RS/9',
       'Balls', 'Strikes', 'Pitches', 'Pull%', 'Cent%', 'Oppo%', 'SL%', 'SLv',
       'CT%', 'CTv', 'CB%', 'CBv', 'CH%', 'CHv', 'SF%', 'SFv', 'KN%', 'KNv',
       'XX%', 'wFB', 'wSL', 'wCT', 'wCB', 'wCH', 'wSF', 'wKN', 'wFB/C',
       'wSL/C', 'wCT/C', 'wCB/C', 'wCH/C', 'wSF/C', 'wKN/C'], axis=1)

#sns.pairplot(df_keystats, hue='Points_lead', size=1.5)
#plt.show()

# WILL ADD TO EXPLORATORY ANALYSIS


# PITCHER PERFORMANCE MODEL --------------------------------------------------------------------------------------------
# Will create new models as I go through ML course I'm auditing at Hopkins

# Drop missing values
df_keystats = df_keystats.dropna()
#print(len(df_keystats))                          # 3561 complete records in df

# Set index
df_keystats.set_index(['playerid', 'Season'], inplace=True)

# X is independent variables
X = df_keystats[df_keystats.columns[:-1]]

# y is the target (points_lead)
y = df_keystats['Points_lead'].astype('int')

# Load packages
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt

# Linear regression model with cross validation resampling
# Set number folds
cv = KFold(n_splits=5, random_state=1, shuffle=True)

# Build multiple linear regression model
model = LinearRegression()

# Use CV and evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)

# RMSE
print(sqrt(mean(absolute(scores))))              # 98.81 so not very good. Need to use a more flexible model









