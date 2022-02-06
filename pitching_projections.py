# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from numpy import mean
from numpy import absolute
from numpy import sqrt

# FOR OUTPUT FORMAT PYCHARM
#pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Load in dataset
df_pitchers = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchers2000_2019.csv")

# Feature Engineering --------------------------------------------------------------------------------------------------
def weight_stat(df, cols, weightcol):
    for i in cols:
        name = str("w" + i)
        df[name] = df[i] * df[weightcol]

weight_stat(df=df_pitchers, cols=['SIERA', 'BABIP', 'WHIP', 'FlyB%', 'K/9'], weightcol='weight')

df_pitchers['xHard%'] = df_pitchers['Hard%'] / (df_pitchers['Hard%'].mean()*100)      # Hard hit % standardized to mean
df_pitchers['Ks_and_Soft'] = df_pitchers['K%'] * df_pitchers['Soft%']

# Create a new df that only includes the variables I'm interested in
df_keystats = df_pitchers.drop(['Name', 'Team', 'HLD', 'CG', 'SL%', 'SLv', 'CT%', 'CTv', 'CB%', 'CBv', 'CH%',
                                'CHv', 'SF%', 'SFv', 'KN%', 'KNv', 'XX%', 'wFB', 'wSL', 'wCT', 'wCB', 'wCH', 'wSF',
                                'wKN', 'wFB/C', 'wSL/C', 'wCT/C', 'wCB/C', 'wCH/C', 'wSF/C', 'wKN/C', 'Balls',
                                'Strikes', 'Pitches', 'W', 'L', 'R', 'ER', 'BB', 'SO', 'H', 'GS', "BS",
                                "SV", 'IP', 'FIP-', 'xFIP-', 'FIP', 'xFIP', 'E-F', 'Hard%', 'ERA',
                                'K%', 'K/9', 'G', 'ShO', 'TBF', 'HR', 'IBB', 'HBP', 'WP', 'BK', 'K/9', 'BB/9', 'K/BB',
                                'K%', 'BB%', 'AVG', 'LOB%', 'ERA-', 'GB/FB', 'GB%', 'FlyB%', 'IFFB%', 'RS', 'Pull%',
                                'Cent%', 'Oppo%', 'Soft%', 'Med%', 'FastB%', 'HR/9', 'K-BB%', 'HR/FB', 'RS/9',
                                'num_seasons', 'sum_num', 'weight', 'SIERA', 'WHIP', 'BABIP', 'LD%', 'FBv'], axis=1)

# Drop missing values
df_keystats = df_keystats.dropna()

# Set index
df_keystats.set_index(['playerid', 'Season'], inplace=True)


# EXPLORATORY ANALYSIS -------------------------------------------------------------------------------------------------
'''
# Look at the distributions
vars = df_keystats.columns
for var in vars:
    var = str(var)
    plt.hist(df_pitchers[var], bins=50)
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.axvline(df_pitchers[var].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.show()

# Visualize variables by year
sns.set(style='whitegrid')
for var in vars:
    var = str(var)
    ax = sns.boxplot(x="Season", y=var, data=df_pitchers)
    #plt.show()

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

# Find the stats with the greatest correlation with next year's points
cors = df_pitchers.corrwith(df_pitchers['Points_lead'])
print(cors.sort_values())

# Create correlation matrix and generate a heatmap
corr_matrix = df_keystats.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, cmap='RdYlGn_r', xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
#plt.show()



# PITCHER PERFORMANCE MODELS -------------------------------------------------------------------------------------------

# X is independent variables
#X = df_keystats[df_keystats.columns[:-1]]
X = df_keystats.drop('Points_lead', axis=1)

# y is the target (points_lead)
y = df_keystats['Points_lead'].astype('int')

### Linear regression model with cross validation resampling ###
# Set number folds
cv = KFold(n_splits=5, random_state=1, shuffle=True)

# Build multiple linear regression model
model = LinearRegression()

# Use CV and evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)

# RMSE
print(sqrt(mean(absolute(scores))))  # RMSE = 98.03 so not very good. Need to use a more flexible model. Also, LR
        # assumes the effect of changes in a predictor on the response is independent of other predictors which
        # isn’t true in my case.


### K Nearest Neighbors regression model with cross validation resampling ###

# Build KNN regression model
KNNmodel = KNeighborsRegressor(n_neighbors=10)

# Use CV and evaluate model
KNNscores = cross_val_score(KNNmodel, X, y, scoring='neg_mean_squared_error', cv=cv)

# RMSE of KNN model
print(sqrt(mean(absolute(KNNscores))))           # RMSE = 101.83 too many dimensions



from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly = LinearRegression()
Polyscore = cross_val_score(poly, X_poly, y, scoring='neg_mean_squared_error', cv=5)
print(sqrt(mean(absolute(Polyscore))))

print(df_keystats.columns)








# To do: consider data transformation (partic. for skewed variables) and dim. reduction techniques/removing corr vars



