import pandas as pd
import os
from features import calculate_player_years_experience, calculate_team_experience, calculate_player_awards, calculate_regular_season_player_score, calculate_postseason_season_player_score, calculate_shooting_percentages, calculate_player_efficiency, calculate_player_experience, calculate_player_score_year, calculate_mean_team_players_score, calculate_pre_season_player_score,calculate_mean_college_score, calculate_franchise_experience, calculate_team_offensive_score, calculate_team_defensive_score, calculate_team_performance_score, calculate_team_score, calculate_pre_season_team_stats_score, calculate_current_team_players_score


# Method to separate the datasets by years and define some important attributes
def organizeByYear():
    dfTeams = pd.read_csv("./data/clean/clean_teams.csv")
    dfPlayers = pd.read_csv("./data/clean/clean_players_teams.csv")
    dfCoaches = pd.read_csv("./data/clean/clean_coaches.csv")
    dfAwards = pd.read_csv("./data/clean/clean_awards_players.csv")  
    output_dir_years_datasets = './data/year/'

    train_years = len(dfTeams['year'].unique()) + 1
    #print(train_years)

    # Create folder 'year' to save the datasets seperated by years 
    if not os.path.exists(output_dir_years_datasets):
        os.makedirs(output_dir_years_datasets)

    # Initialize a dictionary to store team experience data per year
    team_experience_dict = {}
    
    # Create a dictionary to store player scores averaged over the years
    player_score_dict = {}

    # Seperate player datasets by year
    for year in range(1, train_years):
        year_df = calculate_player_years_experience(dfPlayers, year) # player data + experience in the league

        year_df = calculate_player_awards(year_df, dfAwards, year) # add awards to the df
        year_df = calculate_regular_season_player_score(year_df)
        year_df = calculate_postseason_season_player_score(year_df)
        year_df = calculate_shooting_percentages(year_df)
        year_df = calculate_player_efficiency(year_df)
        year_df = calculate_player_experience(year_df)

        year_df = calculate_player_score_year(year_df)

        year_df = calculate_pre_season_player_score(year_df, player_score_dict)

        # Update player score dictionary
        for idx, row in year_df.iterrows():
            playerID = row['playerID']
            player_score = row['player_score']
            if playerID in player_score_dict:
                all_seasons_score, years_played = player_score_dict[playerID]
                all_seasons_score += player_score
                years_played += 1
                player_score_dict[playerID] = (all_seasons_score, years_played)
            else:
                player_score_dict[playerID] = (player_score, 1)

        filename = f"players/yearPlayers_{year}.csv"
        year_df.to_csv(os.path.join(output_dir_years_datasets, filename), index=False)

        # Calculate average team experience
        team_experience = calculate_team_experience(year_df)
        team_experience_dict[year] = team_experience


    # Create a dictionary to store teams stats scores averaged over the years
    team_stats_score_dict = {}

    # Separate team datasets by year
    for year in range(1, train_years):
        year_df = dfTeams[dfTeams['year'] == year]

        # Check data existence of experience for current year
        team_experience = team_experience_dict[year]
        year_df = pd.merge(year_df, team_experience, on='tmID', how='left')
        year_df['avg_experience'] = year_df['avg_experience'].fillna(0)

        # Calculate and merge franchise experience
        year_df = calculate_franchise_experience(dfTeams, year)
        year_df['franchise_experience'] = year_df['franchise_experience'].fillna(0)
        year_df = calculate_team_offensive_score(year_df)
        year_df = calculate_team_defensive_score(year_df)
        year_df = calculate_team_performance_score(year_df)
        year_df = calculate_team_score(year_df)
        year_df = round(calculate_pre_season_team_stats_score(year_df, team_stats_score_dict),1)
        
        for idx, row in year_df.iterrows():
            teamID = row['tmID']
            team_stats_score = row['team_stats_score']
            if teamID in team_stats_score_dict:
                all_seasons_score, years_played = team_stats_score_dict[teamID]
                all_seasons_score += team_stats_score
                years_played += 1
                team_stats_score_dict[teamID] = (all_seasons_score, years_played)
            else:
                team_stats_score_dict[teamID] = (team_stats_score, 1)

        filename = f"teams/yearTeams_{year}.csv"
        year_df.to_csv(os.path.join(output_dir_years_datasets, filename), index=False)
        
    # Separate coach datasets by year
    for year in range(1, train_years):
        year_df = dfCoaches[dfCoaches['year'] == year]
        filename = f"coaches/yearCoaches_{year}.csv"
        year_df.to_csv(os.path.join(output_dir_years_datasets, filename), index=False)


    teams_dataset = dfPlayers['tmID'].unique() # All existent teams on the dataset 

    # Create a dictionary where keys represent teams ID and values the future mean team players score
    team_players_score_dict = {team: 0.0 for team in teams_dataset}
    all_time_team_players_score = {team: 0.0 for team in teams_dataset}

    years = len(dfTeams['year'].unique()) + 1
    for year in range(1, train_years):
        dfPlayers = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        dfTeams = pd.read_csv(f"./data/year/teams/yearTeams_{year}.csv")

        weight = year / years

        for teamID in team_players_score_dict.keys():
            all_time_team_players_score[teamID] = round(team_players_score_dict[teamID], 1)
            team_players_score_dict[teamID] += weight * calculate_mean_team_players_score(dfPlayers, teamID, year)

        teams_current_year = dfTeams['tmID'].unique()
        current_team_players_score = {team: 0.0 for team in teams_current_year} # Determine the team players score of the year according to their pre_season_player_score (mean of Player Score on the last years)

        for teamID in current_team_players_score.keys():
            current_team_players_score[teamID] = calculate_current_team_players_score(dfPlayers, teamID, year)

        dfTeams['all_time_team_players_score'] = dfTeams['franchID'].map(all_time_team_players_score)
        dfTeams['current_team_players_score'] = dfTeams['franchID'].map(current_team_players_score)
        dfTeams['team_players_score'] = dfTeams['franchID'].map(team_players_score_dict)
        
        if (year - 1 == 0):
            dfTeams['all_time_team_players_score'] = round(dfTeams['all_time_team_players_score'], 1)
            dfTeams['current_team_players_score'] = round(dfTeams['current_team_players_score'], 1)
        else:
            dfTeams['all_time_team_players_score'] = round(dfTeams['all_time_team_players_score'] / (year - 1), 1) # Means of Player Score of each team until the last year
            dfTeams['current_team_players_score'] = round(dfTeams['current_team_players_score'] / (year - 1), 1)  # Player Score of the current team according to thei pre_season_player_score
        
        dfTeams['team_players_score'] = round(dfTeams['team_players_score'] / year, 1)  # Sum of means of Player Score of each team over the years. Counts the current year too

        dfTeams['team_score'] = (0.5 * dfTeams['team_players_score']) + (0.5 * dfTeams['team_stats_score'])
        dfTeams['team_score'] = dfTeams['team_score'].round(1)

        dfTeams['pre_season_team_score'] = (
            0.5 * dfTeams['all_time_team_players_score'] +
            0.5 * dfTeams['pre_season_team_stats_score']
        ).round(1)

        filename = f"teams/yearTeams_{year}.csv"
        dfTeams.to_csv(os.path.join(output_dir_years_datasets, filename), index=False)

        # print(team_players_score_dict)
        # print()


def rookieScore():
    dfTeams = pd.read_csv("./data/clean/clean_teams.csv")
    dfPlayers = pd.read_csv("./data/clean/clean_players.csv")
    allRookies = pd.DataFrame()
    output_dir_years_rookies_datasets = './data/year/rookie/'

    train_years = len(dfTeams['year'].unique()) + 1

    if not os.path.exists(output_dir_years_rookies_datasets):
        os.makedirs(output_dir_years_rookies_datasets)

    years = len(dfTeams['year'].unique()) + 1

    colleges = dfPlayers['college'].unique()
    college_dic = {college : 0.0 for college in colleges}
    
    for year in range(2, train_years):
        dfPlayersYear = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        
        dfRookies = dfPlayersYear[dfPlayersYear['experience'] == 1]
        dfRookies = dfRookies.merge(dfPlayers, left_on='playerID', right_on='bioID', how='left')
        allRookies = pd.concat([allRookies, dfRookies], ignore_index=True)

    for college in college_dic.keys():
        college_dic[college] += calculate_mean_college_score(allRookies, college, year)

    filename = f"rookies.csv"
    allRookies.to_csv(os.path.join(output_dir_years_rookies_datasets, filename), index=False)

    mean_score = round(allRookies['player_score'].mean(), 1)
    with open("./data/year/rookie/mean.txt", "w") as file:
        file.write(str(mean_score))

    dfCollegeScores = pd.DataFrame(list(college_dic.items()), columns=['College', 'Score'])
    dfCollegeScores['Score'] = dfCollegeScores['Score'].round(1)
    dfCollegeScores = dfCollegeScores.sort_values(by='Score', ascending=False)
    dfWithoutTop15 = dfCollegeScores.iloc[15:]
    meanScoreWithoutTop15 = dfWithoutTop15['Score'].mean()

    dfCollegeScores['Score'] = dfCollegeScores['Score'].replace(0.0, meanScoreWithoutTop15).round(1)
    dfCollegeScores.to_csv(os.path.join(output_dir_years_rookies_datasets, "college_scores.csv"), index=False)
    #print(college_dic)

def finals_winners_playoff_analysis():
    dfTeams = pd.read_csv("./data/clean/clean_teams.csv")
    output_dir_years_datasets = './data/year/'
    train_years = len(dfTeams['year'].unique()) + 1

    results = []
    dfTeams = dfTeams.sort_values(by='year')

    for year in range(1, train_years):
        # Get  team that won
        winning_team = dfTeams[(dfTeams['year'] == year) & (dfTeams['finals'] == 'W')]

        if not winning_team.empty:
            team_id = winning_team['tmID'].values[0]

            # Check playoff next year
            if year < 10:
                next_year_playoff = dfTeams[(dfTeams['year'] == year + 1) & (dfTeams['tmID'] == team_id) & (dfTeams['playoff'] == 'Y')]
                if not next_year_playoff.empty:
                    # Determine  farthest playoff round reached
                    if next_year_playoff['finals'].values[0] in ['W', 'L']:
                        playoff_round = 'Finals'
                    elif next_year_playoff['semis'].values[0] in ['W', 'L']:
                        playoff_round = 'Semifinals'
                    elif next_year_playoff['firstRound'].values[0] in ['W', 'L']:
                        playoff_round = 'First Round'
                    else:
                        playoff_round = 'Unknown'
                    
                    qualified_next_season = 'Yes'
                else:
                    qualified_next_season = 'No'
                    playoff_round = 'N/A'
            else:
                qualified_next_season = 'TBD'
                playoff_round = 'TBD'

            # Append result
            results.append({
                'Year': year,
                'TeamID': team_id,
                'Finals Win': 'Yes',
                'Qualified Next Season Playoffs': qualified_next_season,
                'Playoff Round Reached': playoff_round
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir_years_datasets, 'finals_winners_playoff_qualification.csv'), index=False)

# Method for cleaning test dataset
def dataYear11():
    dfTeams11 = pd.read_csv("./data/year/teams/yearTeams_11.csv")
    dfTeams10 = pd.read_csv("./data/year/teams/yearTeams_10.csv")
    dfPlayers10 = pd.read_csv("./data/year/players/yearPlayers_10.csv")
    dfPlayers11 = pd.read_csv("./data/year/players/yearPlayers_11.csv")

    mergedPlayers_df = dfPlayers11.merge(dfPlayers10, on='playerID', how='left', suffixes=('_11', '_10'))
    columns_to_keep = [col for col in mergedPlayers_df.columns if col.endswith('11') or col == 'playerID' or col == 'player_score' or col == 'pre_season_player_score']
    mergedPlayers_df = mergedPlayers_df[columns_to_keep]
    mergedPlayers_df.rename(columns={col: col[:-3] for col in mergedPlayers_df.columns if col.endswith('11')}, inplace=True)

    # Update player scores for year 11
    mergedPlayers_df['pre_season_player_score'] = round(
        (mergedPlayers_df['pre_season_player_score'] + mergedPlayers_df['player_score'] * 0.1), 1
    )

    # Fill missing values for player scores using mean value from file
    with open('./data/year/rookie/mean.txt', 'r') as file:
        content = file.read()
    number = float(content.strip())
    mergedPlayers_df['player_score'] = mergedPlayers_df['player_score'].fillna(number)
    mergedPlayers_df['pre_season_player_score'] = mergedPlayers_df['pre_season_player_score'].fillna(number)

    # Save the updated player data for 2021
    mergedPlayers_df.to_csv("./data/year/players/yearPlayers_11.csv", index=False)

    # Merge team data for year 11
    merged_df = dfTeams11.merge(dfTeams10, on='franchID', how='left', suffixes=('_11', '_10'))
    columns_to_keep = [col for col in merged_df.columns if col.endswith('11') or col == 'franchID' or col == 'all_time_team_players_score' or col == 'team_players_score' or col == 'pre_season_team_score']
    merged_df = merged_df[columns_to_keep]
    merged_df.rename(columns={col: col[:-3] for col in merged_df.columns if col.endswith('11')}, inplace=True)
    
    # Calculate the team players' score for each team
    tmID_list = merged_df["tmID"].tolist()
    for team in tmID_list:
        mean = calculate_mean_team_players_score(mergedPlayers_df, team, 1)
        current_score = calculate_current_team_players_score(mergedPlayers_df, team, 11)
        merged_df.loc[merged_df["tmID"] == team, "team_players_score1"] = round(mean, 1)
        merged_df.loc[merged_df["tmID"] == team, "current_team_players_score"] = round(current_score, 1)

    # Update the cumulative team scores
    merged_df['team_players_score'] = round(
        (merged_df['team_players_score'] +
        merged_df['team_players_score1']), 1
    )

    merged_df['all_time_team_players_score'] = round(
        (merged_df['all_time_team_players_score'] +
        merged_df['team_players_score'] * 0.2), 1
    )

    merged_df.to_csv("./data/year/teams/yearTeams_11.csv", index=False)



