import numpy as np
import pandas as pd

# FOR OUTPUT FORMAT PYCHARM
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# DATA CLEANING -------------------------------------------------------------------------------------------------

# Load in dataframes
df_standard = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Standard.csv")
df_advanced = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Advanced.csv")
df_batted = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Battedball.csv")
df_ptype = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchtype.csv")
df_pvalue = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Pitchvalue.csv")

# Merge the dataframes I'm interested in
df_pitchers = pd.merge(df_standard, df_advanced, how='left', on=['playerid', 'Season'])
df_pitchers = pd.merge(df_pitchers, df_batted, how='left', on=['playerid', 'Season'])
df_pitchers = pd.merge(df_pitchers, df_ptype, how='left', on=['playerid', 'Season'])
df_pitchers = pd.merge(df_pitchers, df_pvalue, how='left', on=['playerid', 'Season'])
df_pitchers.drop(['Name_y', 'Team_y', 'ERA_y', 'BABIP_y', 'FB%_y', 'Name_x', 'Team_x'], axis=1, inplace=True)
df_pitchers.rename(columns = {'ERA_x':'ERA', 'BABIP_x':'BABIP', 'FB%_x':'FB%'}, inplace = True)

# Make a number of seasons variable
df_pitchers['num_seasons'] = df_pitchers.sort_values('Season', ascending=True).groupby('playerid').cumcount()+1       # orders by year, groups by player, then counts number of rows for each player
df_pitchers = df_pitchers[df_pitchers.IP > 59]                                                                        # filters out people with less than 60 IP (needs to use 0 min to start to make sure I got every year for number of season count)
df_pitchers = df_pitchers[df_pitchers.Season > 1999]                                                                  # filters out everything before 2000

# Multi-level index of  player and then each season for that player
df_pitchers = df_pitchers.set_index(['playerid', 'Season'])

# Point column doesnt include quality starts because FanGraphs doesn't provide them
df_pitchers["Points"] = (df_pitchers['IP'] * 3) + (df_pitchers['H'] * -1) + (df_pitchers['ER'] * -2) + (df_pitchers['BB'] * -1) + (df_pitchers['HBP'] * -1) + (df_pitchers['SO'] * 1)

# need to create a lead_points column where value is number of points player scored in next season (ie current row + 1)

# EXPLORATORY ANALYSIS ------------------------------------------------------------------------------------------


