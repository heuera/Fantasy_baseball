import numpy as np
import pandas as pd

# FOR OUTPUT FORMAT PYCHARM
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Load in dataframes
df_standard = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Standard.csv")
df_advanced = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Advanced.csv")
df_batted = pd.read_csv("/Users/austinheuer/Desktop/Fantasy_baseball/data/Battedball.csv")


# Merge the dataframes I'm interested in
df_pitchers = pd.merge(df_standard, df_advanced, how='left', on=['ERA', 'playerid'])
df_pitchers.drop(['Name_y', 'Team_y'], axis=1, inplace=True)
df_pitchers = pd.merge(df_pitchers, df_batted, how='left', on=['BABIP', 'playerid'])
df_pitchers.drop(['Name_x', 'Team_x'], axis=1, inplace=True)

