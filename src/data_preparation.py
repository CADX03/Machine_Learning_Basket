import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def combine_yearly_team_data():
    data_dir = './data/year/'
    combined_dir = './data/year/combined/'

    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
    
    year_dfs = []
    
    for year in range(1, 12):
        file_path = os.path.join(data_dir, f'teams/yearTeams_{year}.csv')
        if os.path.exists(file_path):
            year_df = pd.read_csv(file_path)
            year_dfs.append(year_df)
    
    combined_df = pd.concat(year_dfs, ignore_index=True)

    columns_to_normalize = [
        'all_time_team_players_score',
        'team_players_score',
        'pre_season_team_score',
        'team_players_score1',
        'current_team_players_score'
    ]

    available_columns = [col for col in columns_to_normalize if col in combined_df.columns]
    
    # Apply Min-Max Normalization
    scaler = MinMaxScaler()
    combined_df[available_columns] = scaler.fit_transform(combined_df[available_columns])

    combined_df[available_columns] = combined_df[available_columns].round(4)
    
    combined_df.to_csv(os.path.join(combined_dir, 'combined_team_data.csv'), index=False)
    
    return combined_df