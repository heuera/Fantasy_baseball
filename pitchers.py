import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model

# Read in datasets
standard = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/standard.csv")
advanced = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/advanced.csv")
batted = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/battedball.csv")
statcast = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/statcast.csv")

# Drop columns that are duplicated
advanced = advanced.drop(['Name', 'Team', 'ERA'], axis=1)
batted = batted.drop(['Name', 'Team', 'BABIP'], axis=1)
statcast = statcast.drop(['Name', 'Team', 'IP', 'ERA'], axis=1)

# Merge datasets into a single dataframe
pitchers_1 = pd.merge(standard, advanced, on='playerid')
pitchers_2 = pd.merge(pitchers_1, batted, on='playerid')
pitchers = pd.merge(pitchers_2, statcast, on='playerid')

# Generate a points column based on fantasy scoring (fangraphs doesn't record a quality starts)
pitchers["points"] = (pitchers['IP']*3) + (pitchers['H']*-1) + (pitchers['ER']*-2) + (pitchers['BB']*-1) + (pitchers['HBP']*-1) + (pitchers['SO']*1) + (pitchers['W']*3) + (pitchers['L']*-3)

# Summary stats for each column
pd.set_option('display.max_columns', None)
#print(pitchers.describe())

# Regression model with basic stats against points
reg = linear_model.LinearRegression()
basic_variables = ['W', "L", "ERA", "IP", "H", "TBF", 'ER', 'BB', "HR", 'SO', 'K/9', 'BB/9', 'K/BB', 'HR/9']
reg.fit(pitchers[basic_variables], pitchers.points)
basic_variables = pd.Series(['W', "L", "ERA", "IP", "H", "TBF", 'ER', 'BB', "HR", 'SO', 'K/9', 'BB/9', 'K/BB', 'HR/9'])
reg_cos = pd.Series(reg.coef_)
basic = pd.concat([basic_variables, reg_cos], axis=1)
print(basic)


'''
fig, ax = plt.subplots()
pitchers.plot(kind='scatter',x='WHIP', y='points', c='blue', ax=ax)
ax.set_xlabel("WHIP")
ax.set_ylabel("Points")
plt.show()
'''