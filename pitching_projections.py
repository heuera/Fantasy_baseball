import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

# FOR OUTPUT FORMAT PYCHARM
pd.set_option('display.max_columns', None)
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
# Will create new model(s) as I go through ML course I'm auditing at Hopkins

# Multiple regression model
# How many individual player-seasons are in the  dataset?
len(df_pitchers)                                                                                     # 4056 total player-years

'''
# Take a random sample from the dataframe
df_sample = df_pitchers.sample(frac=0.5, replace=False, random_state=1)                              # gives a sample of 2028 player-years

# Crude model (points_lead vs points)
crude = linear_model.LinearRegression().fit(df_sample[['Points']], df_sample[['Points_lead']])       # y = 0.64x + 57.47
#print(crude.coef_, crude.intercept_)
'''
# Using train_test_split
df_keystats = df_keystats.dropna()
df_keystats.set_index(['playerid', 'Season'], inplace=True)
#X = df_keystats[df_keystats.columns[:-1]]
X = df_keystats[['FIP', 'W']]
y = df_keystats['Points_lead'].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.5, random_state=1)

model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)
print(accuracy_score(y_test, y_model))

