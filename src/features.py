import pandas as pd

# Method to calculate players experience according to years played
def calculate_player_years_experience(dfPlayers, current_year):
    # Include only records of the previous years of the current season
    dfPlayers_before_year = dfPlayers[dfPlayers['year'] <= current_year]

    # Calculate distinct years played (unique entrees in clean_players_teams)
    player_experience = dfPlayers_before_year.groupby('playerID')['year'].nunique().reset_index()
    player_experience.rename(columns={'year': 'experience'}, inplace=True)
    
    dfPlayers_current_year = dfPlayers[dfPlayers['year'] == current_year].copy()
    dfPlayers_current_year = pd.merge(dfPlayers_current_year, player_experience, on='playerID', how='left')
    dfPlayers_current_year['experience'] = dfPlayers_current_year['experience'].fillna(0).astype(int)

    return dfPlayers_current_year

# Method to calculate franchise experience according to years played
def calculate_franchise_experience(dfTeams, current_year):
    # Include records of the previous years of the curret season
    dfTeams_before_year = dfTeams[dfTeams['year'] < current_year]

    # Calculate distinct years played by each franchise
    franchise_experience = dfTeams_before_year.groupby('franchID')['year'].nunique().reset_index()
    franchise_experience.rename(columns={'year': 'franchise_experience'}, inplace=True)

    # Get the team data for the current year and merge it
    dfTeams_current_year = dfTeams[dfTeams['year'] == current_year].copy()
    dfTeams_current_year = pd.merge(dfTeams_current_year, franchise_experience, on='franchID', how='left')

    dfTeams_current_year['franchise_experience'] = dfTeams_current_year['franchise_experience'].fillna(0).astype(int)

    return dfTeams_current_year
def calculate_team_offensive_score(dfTeams):
    #(o_pts / GP) + [0.5 * (o_asts / GP)] + [0.3 * (o_reb / GP)] + [0.2 * (o_stl / GP)] - [0.3 * (o_to / GP)]
    games_played = dfTeams['GP']
    points =  dfTeams['o_pts']
    points =  dfTeams['o_asts']
    rebounds =  dfTeams['o_reb']
    steals =  dfTeams['o_stl']
    turn_overs =  dfTeams['o_to']

    offsensive_score = (points / games_played) + (0.5 * (points / games_played)) + (0.3 * (rebounds / games_played)) + (0.2 * (steals / games_played)) - (0.3 * (turn_overs / games_played))
    dfTeams['o_score'] = round(offsensive_score, 1)

    return dfTeams

def calculate_team_defensive_score(dfTeams):
    # (d_pts / GP) + [0.5 * (d_asts / GP)] + [0.3 * (d_reb / GP)] + [0.2 * (d_stl / GP)] - [0.3 * (d_to / GP)]
    games_played = dfTeams['GP']
    points =  dfTeams['d_pts']
    points =  dfTeams['d_asts']
    rebounds =  dfTeams['d_reb']
    steals =  dfTeams['d_stl']
    turn_overs =  dfTeams['d_to']

    defensive_score = (points / games_played) + (0.5 * (points / games_played)) + (0.3 * (rebounds / games_played)) + (0.2 * (steals / games_played)) - (0.3 * (turn_overs / games_played))
    dfTeams['d_score'] = round(defensive_score, 1)

    return dfTeams

def calculate_team_performance_score(dfTeams):
    #(won / GP) + [0.3 * (homeW / (homeW + homeL))] + [0.35 * (awayW / (awayW + awayL))] + [0.5 * (confW / (confW + confL))]  + [play_off_phase * playoff_performance_factor]
    games_played = dfTeams['GP']
    wins = dfTeams['won']
    home_wins = dfTeams['homeW']
    home_losses = dfTeams['homeL']
    away_wins = dfTeams['awayW']
    away_losses = dfTeams['awayL']
    conf_wins = dfTeams['confW']
    conf_losses = dfTeams['confL']
    playoff_performance_factor = 100

    dfTeams['play_off_phase'] = 0.0

    for index, row in dfTeams.iterrows():
        if row['finals'] == 'W':
            dfTeams.at[index, 'play_off_phase'] = 0.5
        elif row['semis'] == 'W':
            dfTeams.at[index, 'play_off_phase'] = 0.3
        elif row['firstRound'] == 'W':
            dfTeams.at[index, 'play_off_phase'] = 0.2
        elif row['firstRound'] == 'L':
            dfTeams.at[index, 'play_off_phase'] = 0.1
        else:
            dfTeams.at[index, 'play_off_phase'] = 0.0

    play_off_phase = dfTeams['play_off_phase']
    
    performance_score  = (wins / games_played) + (0.3 * (home_wins / (home_wins + home_losses))) + (0.35 * (away_wins / (away_wins + away_losses))) + (0.5 * (conf_wins / (conf_wins + conf_losses)))  + (play_off_phase * playoff_performance_factor)
    dfTeams['pf_score'] = round(performance_score, 1)

    columns_to_drop = ['play_off_phase']
    dfTeams.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return dfTeams
    
def calculate_team_score(dfTeams):
    #(0.3 * OS) + (0.3 * DS) + (0.4 * PS)
    offensive_score = dfTeams['o_score']
    defensive_score = dfTeams['d_score']
    performance_score = dfTeams['pf_score']

    team_score = (0.3 * offensive_score) + (0.3 * defensive_score) + (0.4 * performance_score)
    dfTeams['team_stats_score'] = round(team_score, 1)

    return dfTeams

# Method to calculate a teams game experience 
def calculate_team_experience(dfPlayers_current_year):
    # Group by team and calculate average experience
    team_experience = dfPlayers_current_year.groupby('tmID')['experience'].mean().reset_index()
    team_experience['experience'] = team_experience['experience'].round(2)
    team_experience.rename(columns={'experience': 'avg_experience'}, inplace=True)
    
    return team_experience

def calculate_pre_season_team_stats_score(dfTeams_current_year, team_stats_score_dict):
    pre_season_team_stats_scores = []
    for idx, row in dfTeams_current_year.iterrows():
        teamID = row['tmID']
        if teamID in team_stats_score_dict:
            all_seasons_score, years_played = team_stats_score_dict[teamID]
            # Calculate pre-season average befre current year
            pre_season_team_stats_score = all_seasons_score / years_played
        else:
            pre_season_team_stats_score = 0  # Rookie season
        pre_season_team_stats_scores.append(round(pre_season_team_stats_score, 3))
    dfTeams_current_year['pre_season_team_stats_score'] = pre_season_team_stats_scores
    return dfTeams_current_year

# Method to calculate the score that players won in awards
def calculate_player_awards(dfPlayers, dfAwards, current_year):
    dfAwards_before_year = dfAwards[(dfAwards['year'] <= current_year) & (dfAwards['award'] != 'Coach of the Year')]
    total_awards = dfAwards_before_year.groupby(['playerID', 'award']).size().reset_index(name='total_awards')
    
    awards_pivot = total_awards.pivot(index='playerID', columns='award', values='total_awards').fillna(0).astype(int)
    
    dfPlayers = pd.merge(dfPlayers, awards_pivot, on='playerID', how='left')
    
    dfPlayers.fillna(0, inplace=True)

    columns_to_drop = [
        'All-Star Game Most Valuable Player',
        'Defensive Player of the Year',
        'Kim Perrot Sportsmanship Award',
        'Most Improved Player',
        'Most Valuable Player',
        'Rookie of the Year',
        'WNBA Finals Most Valuable Player',
        'Sixth Woman of the Year',
        'WNBA All Decade Team Honorable Mention',
        'WNBA All-Decade Team'
    ]

    for column in columns_to_drop:
        if column not in dfPlayers.columns:
            dfPlayers[column] = 0

    all_star_game_mvp = dfPlayers['All-Star Game Most Valuable Player']
    denfensive_player_year = dfPlayers['Defensive Player of the Year']
    kim_perrot = dfPlayers['Kim Perrot Sportsmanship Award']
    mip = dfPlayers['Most Improved Player']
    mvp = dfPlayers['Most Valuable Player']
    roty = dfPlayers['Rookie of the Year']
    finals_mvp = dfPlayers['WNBA Finals Most Valuable Player']
    sixth_year = dfPlayers['Sixth Woman of the Year']
    all_decade_honor = dfPlayers['WNBA All Decade Team Honorable Mention']
    all_decade = dfPlayers['WNBA All-Decade Team']

    awards_score = 1 * ((100 * mvp) + (90 * finals_mvp) + (70 * mip) + (70 * roty) + (60 * denfensive_player_year) + (50 * all_star_game_mvp) + (10 * kim_perrot) + (50 * sixth_year) + (40 * all_decade_honor) + (60 * all_decade))
    
    dfPlayers['awards_score'] = awards_score

    dfPlayers.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return dfPlayers

# Method to calculate a players regular season score
def calculate_regular_season_player_score(dfPlayers):
    # (1 * Points) + (0.7 * Rebounds) + (0.8 * Assists) + (0.5 * Steals) + (0.5 * Blocks) - (0.7 * Turnovers) - (0.3 * Fouls)
    player_points = dfPlayers['points']
    player_rebounds = dfPlayers['rebounds']
    player_assists = dfPlayers['assists']
    player_steals = dfPlayers['steals']
    player_blocks = dfPlayers['blocks']
    player_turnovers = dfPlayers['turnovers']
    player_fouls = dfPlayers['PF']

    regular_season_score = (1.0 * player_points) + (0.7 * player_rebounds) + (0.8 * player_assists) + (0.5 * player_steals) + (0.5 * player_blocks) - (0.7 * player_turnovers) - (0.3 * player_fouls)
    dfPlayers['regular_season_score'] = round(regular_season_score, 1)

    return dfPlayers

# Method to calculate a players post season score
def calculate_postseason_season_player_score(dfPlayers):
    # (1.2 * Points) + (0.9 * Rebounds) + (1 * Assists) + (0.7 * Steals) + (0.7 * Blocks) - (0.8 * Turnovers) - (0.4 * Fouls)
    player_points = dfPlayers['PostPoints']
    player_rebounds = dfPlayers['PostRebounds']
    player_assists = dfPlayers['PostAssists']
    player_steals = dfPlayers['PostSteals']
    player_blocks = dfPlayers['PostBlocks']
    player_turnovers = dfPlayers['PostTurnovers']
    player_fouls = dfPlayers['PostPF']

    post_season_score = (1.2 * player_points) + (0.9 * player_rebounds) + (1.0 * player_assists) + (0.7 * player_steals) + (0.7 * player_blocks) - (0.8 * player_turnovers) - (0.4 * player_fouls)
    dfPlayers['post_season_score'] = round(post_season_score, 1)

    return dfPlayers

# Method to calculate a players goals conversion rate
def calculate_shooting_percentages(dfPlayers):
    # Initialize with 'N/A'  (Not Enough)
    dfPlayers['FG%'] = 'N/E'
    dfPlayers['3P%'] = 'N/E'
    dfPlayers['FT%'] = 'N/E'

    # Mininum of 20 field goald attempts 
    fg_attempt_mask = dfPlayers['fgAttempted'] >= 20
    dfPlayers.loc[fg_attempt_mask, 'FG%'] = (dfPlayers['fgMade'] / dfPlayers['fgAttempted'] * 100).round(2)

    # minimum of 10 attempted
    three_attempt_mask = dfPlayers['threeAttempted'] >= 10
    dfPlayers.loc[three_attempt_mask, '3P%'] = (dfPlayers['threeMade'] / dfPlayers['threeAttempted'] * 100).round(2)

    # Minimum of 15 fgA
    ft_attempt_mask = dfPlayers['ftAttempted'] >= 15
    dfPlayers.loc[ft_attempt_mask, 'FT%'] = (dfPlayers['ftMade'] / dfPlayers['ftAttempted'] * 100).round(2)

    return dfPlayers

# Method to calculate a players shotting efficiency
def calculate_player_efficiency(dfPlayers):
    # Ensure the columns are numeric, coerce invalid values to NaN
    dfPlayers['FG%'] = pd.to_numeric(dfPlayers['FG%'], errors='coerce')
    dfPlayers['3P%'] = pd.to_numeric(dfPlayers['3P%'], errors='coerce')
    dfPlayers['FT%'] = pd.to_numeric(dfPlayers['FT%'], errors='coerce')

    # Fill NaN values with 0 (or choose another strategy)
    dfPlayers.fillna(0, inplace=True)

    # (1 * FieldGoalEfficiency) + (0.7 * free throw) + (0.8 * three-point-efficiency)
    efficiency = (dfPlayers['FG%']) + (0.8 * dfPlayers['3P%']) + (0.7 * dfPlayers['FT%'])

    dfPlayers['efficiency'] = round(efficiency, 1)

    return dfPlayers

# Method to calculate a players total experience
def calculate_player_experience(dfPlayers):
    experience = dfPlayers['experience']
    awards_score = dfPlayers['awards_score']

    # (0.5 * 10 * anos jogados) +  (1.5 * premios adquiridos)
    total_experience = (0.5 * 10 * experience) + (1.5 * awards_score)

    dfPlayers['total_experience'] = round(total_experience,1)

    return dfPlayers


# Method to calculate a players score on a year
def calculate_player_score_year(dfPlayers):
    regular_season_score = dfPlayers['regular_season_score']
    post_season_score = dfPlayers['post_season_score']
    efficiency = dfPlayers['efficiency']
    total_experience = dfPlayers['total_experience']

    player_score = (0.6 * regular_season_score) + (0.9 * post_season_score) + (0.4 * efficiency) + (0.2 * total_experience)

    dfPlayers['player_score'] = round(player_score, 1)

    return dfPlayers

# Method to calculate the mean of the 8 best players score on a year
def calculate_mean_team_players_score(dfPlayers, teamID, year):
    playersTeam = dfPlayers[dfPlayers['tmID'] == teamID]

    top8Players = playersTeam.sort_values(by='player_score', ascending=False).head(8)

    sizeTeam = len(top8Players)

    if sizeTeam > 0:
        players_score = top8Players['player_score'].sum() / sizeTeam
    else:
        players_score = 0

    #print(str(teamID) + '(' + str(year) + ')' + ": " + str(players_score))

    return players_score

# Method to calculate the current players scores of a team
def calculate_mean_team_players_score(dfPlayers, teamID, year):
    playersTeam = dfPlayers[dfPlayers['tmID'] == teamID]

    top8Players = playersTeam.sort_values(by='player_score', ascending=False).head(8)

    sizeTeam = len(top8Players)

    if sizeTeam > 0:
        players_score = top8Players['player_score'].sum() / sizeTeam
    else:
        players_score = 0

    #print(str(teamID) + '(' + str(year) + ')' + ": " + str(players_score))

    return players_score

# Method to calculate the current players scores of a team according to their pre_season_player_score (mean of Player Score on the last years)
def calculate_current_team_players_score(dfPlayers, teamID, year):
    playersTeam = dfPlayers[dfPlayers['tmID'] == teamID]

    top8Players = playersTeam.sort_values(by='pre_season_player_score', ascending=False).head(8)

    sizeTeam = len(top8Players)

    if sizeTeam > 0:
        pre_season_players_score = top8Players['pre_season_player_score'].sum() / sizeTeam
    else:
        pre_season_players_score = 0

    #print(str(teamID) + '(' + str(year) + ')' + ": " + str(players_score))

    return pre_season_players_score

def calculate_pre_season_player_score(dfPlayers_current_year, player_score_dict):
    pre_season_scores = []
    for idx, row in dfPlayers_current_year.iterrows():
        playerID = row['playerID']
        if playerID in player_score_dict:
            all_seasons_score, years_played = player_score_dict[playerID]
            # Calculate pre-season average befre current year
            pre_season_score = all_seasons_score / years_played
        else:
            pre_season_score = 0  # Rookie season
        pre_season_scores.append(round(pre_season_score, 3))
    dfPlayers_current_year['pre_season_player_score'] = pre_season_scores
    return dfPlayers_current_year

def calculate_mean_college_score(dfRookies, college, year):
    rookies = dfRookies[dfRookies['college'] == college]

    sizeCollege = len(rookies)

    if sizeCollege > 0:
        rookies_score = rookies['player_score'].mean()
    else:
        rookies_score = 0

    #print(str(college) + '(' + str(year) + ')' + ": " + str(rookies_score))

    return rookies_score
