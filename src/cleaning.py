import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Method used for data cleaning
def get_clean_data(): 
    awards_players = pd.read_csv('./data/raw/awards_players.csv')
    coaches = pd.read_csv('./data/raw/coaches.csv')
    players_teams = pd.read_csv('./data/raw/players_teams.csv')
    players = pd.read_csv('./data/raw/players.csv')
    series_post = pd.read_csv('./data/raw/series_post.csv')
    teams_post = pd.read_csv('./data/raw/teams_post.csv')
    teams = pd.read_csv('./data/raw/teams.csv') 

    output_dir_cleaned_datasets = './data/clean'

    # Create folder 'clean' to save the cleaned datasets
    if not os.path.exists(output_dir_cleaned_datasets):
        os.makedirs(output_dir_cleaned_datasets)

    # Awards Players
    awards_players['award'] = awards_players['award'].replace('Kim Perrot Sportsmanship', 'Kim Perrot Sportsmanship Award')
    clean_awards_players = awards_players.drop('lgID', axis=1)
    clean_awards_players.drop_duplicates()
    #print(clean_awards_players.max()) # Check for inconsistent data
    #print(clean_awards_players.min()) # Check for inconsistent data
    clean_awards_players.to_csv(f'{output_dir_cleaned_datasets}/clean_awards_players.csv', index=False)


    # Coaches
    clean_coaches = coaches.drop('lgID', axis=1)
    clean_coaches.drop_duplicates()
    #print(clean_coaches.max()) # Check for inconsistent data
    #print(clean_coaches.min()) # Check for inconsistent data
    clean_coaches.to_csv(f'{output_dir_cleaned_datasets}/clean_coaches.csv', index=False)


    # Players Teams
    clean_players_teams = players_teams.drop('lgID', axis=1)
    clean_players_teams.drop_duplicates()
    #print(clean_players_teams.max()) # Check for inconsistent data
    #print(clean_players_teams.min()) # Check for inconsistent data
    clean_players_teams.to_csv(f'{output_dir_cleaned_datasets}/clean_players_teams.csv', index=False)


    # Players
    clean_players = players.drop('deathDate', axis=1)
    clean_players.drop_duplicates()
    clean_players = clean_players[clean_players['birthDate'] != "0000-00-00"] # Remove players that don't exist
    clean_players['college'] = clean_players['college'].fillna('X')
    clean_players['collegeOther'] = clean_players['collegeOther'].fillna('X')
    clean_players.loc[clean_players['college'] == 'none', 'college'] = 'X'

    clean_players['birthDate'] = pd.to_datetime(clean_players['birthDate'], errors='coerce')  # Convert to datetime format
    clean_players['age'] = (pd.to_datetime('today') - clean_players['birthDate']).dt.days // 365  # Convert birthDate to age

    # Encode 'pos' and 'college' columns using LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to 'pos' and 'college'
    clean_players['pos'] = label_encoder.fit_transform(clean_players['pos'])
    clean_players['college'] = label_encoder.fit_transform(clean_players['college'])

    # Proceed with your model training as before
    X = clean_players[['age', 'pos', 'college']]  # Features
    y_height = clean_players['height']  # Target variable (height)

    X_train, X_test, y_train, y_test = train_test_split(X, y_height, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for Height prediction: {mse}")

    # Predict missing height values
    missing_data = clean_players[(clean_players['height'] < 60.0) | (clean_players['height'] > 90.0)]
    predicted_heights = model.predict(missing_data[['age', 'pos', 'college']])

    clean_players.loc[(clean_players['height'] < 60.0) | (clean_players['height'] > 90.0), 'height'] = predicted_heights

    players_approved_weights = clean_players[(clean_players['weight'] >= 120.0) & (clean_players['weight'] <= 255.0)] # We used 87 and 62 since the highest registred WNBA player was 7.2ft tall and the smallest 5.2ft
    mean_weight = int(round(players_approved_weights['weight'].mean(), 1))
    clean_players.loc[(clean_players['weight'] < 120.0) | (clean_players['weight'] > 255.0), 'weight'] = mean_weight # Update players that have incorrect height inputs to the mean height of all players with correct values

    # Check for inconsistent data
    #numeric_columns = clean_players.select_dtypes(include=['number'])
    #print(numeric_columns.max())
    #print(numeric_columns.min())

    clean_players.to_csv(f'{output_dir_cleaned_datasets}/clean_players.csv', index=False)
    

    # Series Post
    clean_series_post = series_post.drop('lgIDWinner', axis=1)
    clean_series_post.drop_duplicates()
    #print(clean_series_post.max()) # Check for inconsistent data
    #print(clean_series_post.min()) # Check for inconsistent data
    clean_series_post = clean_series_post.drop('lgIDLoser', axis=1)
    clean_series_post.to_csv(f'{output_dir_cleaned_datasets}/clean_series_post.csv', index=False)


    # Teams Post
    clean_teams_post = teams_post.drop('lgID', axis=1)
    clean_teams_post.drop_duplicates()
    #print(clean_teams_post.max()) # Check for inconsistent data
    #print(clean_teams_post.min()) # Check for inconsistent data
    clean_teams_post.to_csv(f'{output_dir_cleaned_datasets}/clean_teams_post.csv', index=False)


    # Teams
    clean_teams = teams.drop('lgID', axis=1)
    clean_teams.drop_duplicates()
    #print(clean_teams.max()) # Check for inconsistent data
    #print(clean_teams.min()) # Check for inconsistent data
    clean_teams = clean_teams.drop('divID', axis=1) # Remove empty column
    clean_teams['finals'] = clean_teams['finals'].fillna('X')
    clean_teams['semis'] = clean_teams['semis'].fillna('X')
    clean_teams['firstRound'] = clean_teams['firstRound'].fillna('X')
    clean_teams.to_csv(f'{output_dir_cleaned_datasets}/clean_teams.csv', index=False)

def clean_test_year(): 
    coaches = pd.read_csv('./data/year11/yearCoaches_11.csv')
    players_teams = pd.read_csv('./data/year11/yearPlayers_11.csv')
    teams = pd.read_csv('./data/year11/yearTeams_11.csv')

    # Coaches
    coaches.fillna('', inplace=True)
    clean_coaches = coaches.drop_duplicates()
    clean_coaches.to_csv('./data/year/coaches/yearCoaches_11.csv', index=False)
    
    # Players Teams
    players_teams.fillna('', inplace=True)
    clean_players_teams = players_teams.drop_duplicates(subset=['playerID'], keep='first')
    clean_players_teams.to_csv('./data/year/players/yearPlayers_11.csv', index=False)

    # Teams
    teams.fillna('', inplace=True)
    clean_teams= teams.drop_duplicates()
    clean_teams.to_csv('./data/year/teams/yearTeams_11.csv', index=False)