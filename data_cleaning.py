import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from scipy.stats import pearsonr

# FOR OUTPUT FORMAT PYCHARM
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# DATA CLEANING --------------------------------------------------------------------------------------------------------

# Load in dataframes
df_standard = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Standard.csv")
df_advanced = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Advanced.csv")
df_batted = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Battedball.csv")
df_ptype = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchtype.csv")
df_pvalue = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchvalue.csv")

# Merge the dataframes I'm interested in
df_pitchers = pd.merge(df_standard, df_advanced, how='left', on=['playerid', 'Season'], suffixes=['_x', "_y"])
df_pitchers = pd.merge(df_pitchers, df_batted, how='left', on=['playerid', 'Season'], suffixes=['_one', "_two"])
df_pitchers = pd.merge(df_pitchers, df_ptype, how='left', on=['playerid', 'Season'], suffixes=['_a', "_b"])
df_pitchers = pd.merge(df_pitchers, df_pvalue, how='left', on=['playerid', 'Season'], suffixes=['_c', "_d"])
df_pitchers.drop(['Name_y', 'Team_y', 'ERA_y', 'Name_a', 'Team_a', 'BABIP_two', 'Name_b', 'Team_b', 'Name', 'Team'], axis=1, inplace=True)
df_pitchers.rename(columns = {'Name_x':"Name", 'Team_x':'Team', 'ERA_x':'ERA', 'BABIP_one':'BABIP', 'FB%_b':'FastB%', 'FB%_a':"FlyB%"}, inplace = True)

# Make a number of seasons variable
df_pitchers['num_seasons'] = df_pitchers.sort_values('Season', ascending=True).groupby('playerid').cumcount()+1
df_pitchers['num_seasons'] = df_pitchers['num_seasons'].astype(int)

# Point column doesn't include quality starts because FanGraphs doesn't provide them
df_pitchers["Points"] = (df_pitchers['IP'] * 3) + (df_pitchers['H'] * -1) + (df_pitchers['ER'] * -2) + \
                        (df_pitchers['BB'] * -1) + (df_pitchers['HBP'] * -1) + (df_pitchers['SO'] * 1)

# Create a lead_points column which shows how many points the player had the the next year
df_pitchers['Points_lead'] = df_pitchers.sort_values(['playerid', 'Season']).groupby(['playerid'])['Points'].shift(-1)

# Filter out people with less than 60 IP (use 0 min to start to make sure I include each year in count regardless of IP)
df_pitchers = df_pitchers[df_pitchers.IP > 59]

# Filter out everything before 2000
df_pitchers = df_pitchers[df_pitchers.Season > 1999]

# Multi-level index of  player and then each season for that player
#df_pitchers = df_pitchers.set_index(['playerid', 'Season'])

# Get rid of % sign from all columns that are percentages and then divide by 100
cols = df_pitchers.columns
for head in cols:
    if "%" in head:
        df_pitchers[head] = (pd.to_numeric(df_pitchers[head].str[:-1]).div(100).mask(df_pitchers[head] == '%', 0))

# Because we want to use the previous year to predict performance, we need to drop all players who only played one year
df_pitchers = df_pitchers.groupby('playerid').filter(lambda x: len(x) > 1)

# Drop the last row within each group because it's the players last year and can't predict anything (and we already
# have that year's points total stored in the previous row)
last = df_pitchers.sort_values(['Season']).groupby(['playerid']).tail(1).index
df_pitchers = df_pitchers.drop(last)
df_pitchers = df_pitchers.set_index(['playerid', 'Season'])

# Export df_pitchers to csv
df_pitchers.to_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchers2000_2019.csv", index=True)