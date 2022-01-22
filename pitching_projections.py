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

# Merge the dataframes I'm interested in
df_pitchers = pd.merge(df_standard, df_advanced, how='left', on=['playerId', 'Season'])
df_pitchers = pd.merge(df_pitchers, df_batted, how='left', on=['playerId', 'Season'])
df_pitchers.drop(['Name_y', 'Tm_x', 'TBF_y', 'AVG_y', 'IP_y', 'Name_x', 'Tm_y', 'TBF_x'], axis=1, inplace=True)
df_pitchers.rename(columns = {'AVG_x':'AVG', 'IP_x':'IP'}, inplace = True)

# Multi-level index of  player and then each season for that player
df_pitchers = df_pitchers.set_index(['playerId', 'Season'])

# EXPLORATORY ANALYSIS ------------------------------------------------------------------------------------------


