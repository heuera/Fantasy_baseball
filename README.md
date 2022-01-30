# Fantasy Baseball Predictive Model

## Aim
The goal of this project is to use MLB pitcher data from 2000-2019 to predict pitcher performance in 2022. 

## Data
All data were taken from FanGraphs.com. Only pitchers who threw more than 60 innings were included in the analysis. Five csv files from FanGraphs provided data on standard, advanced, batted ball, pitch type, and pitch value. These files can be found in the data folder. 

Python code to clean and merge the csv files can be found in the data_cleaning.py file. Running this program results in the Pitchers2000_2019.csv found in the data folder. 

Cleaning included: merging the five FanGraphs csv files into one dataframe, dropping duplicate columns, generating a column for the number of seasons each player has been on a MLB team, generating a points column based on league scoring, generating a points_lead column that includes how many points the player scored in the following year, removing % from all percentage data, droping all players who only played one year (because they have nothing to predict for the following year), and droping each players last year (nothing to predict for the following year). 

## Exploratory Analysis

## Predictive Model

## Results

## Acknowledgments
I would like to thank FanGraphs.com for providing the data used in this project.
