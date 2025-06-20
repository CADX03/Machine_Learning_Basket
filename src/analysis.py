import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

output_dir_plots = './data/plots'

# Method used to generate a graph to analyze the 10 players with the most awards
def player_awards(df : pd.DataFrame):
    awards_count = df['playerID'].value_counts()

    most_awarded_players = awards_count.head(10) # 10 players with the most awards
    #print(most_awarded_players)

    plt.figure(figsize=(8,8))
    plt.pie(most_awarded_players, labels=most_awarded_players.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Awards by Player')
    plt.axis('equal')  
    plt.savefig(f"{output_dir_plots}/awards_distribution.png")
    plt.close()

# Method used to generate graphs to better analyze the coach dataset
def coach_analysis(df: pd.DataFrame):
    # Aggregate wins and losses by coachID
    total_wins = df.groupby('coachID')['won'].sum()
    total_losses = df.groupby('coachID')['lost'].sum()
    post_wins = df.groupby('coachID')['post_wins'].sum()
    post_losses = df.groupby('coachID')['post_losses'].sum()

    # Make an unique dataframe to study the relevant attributes 
    coach_stats = pd.DataFrame({
        'Total Wins': total_wins,
        'Total Losses': total_losses,
        'Postseason Wins': post_wins,
        'Postseason Losses': post_losses
    })

    coach_stats.reset_index(inplace=True) # Make coachID a column

    plt.figure(figsize=(20, 6))
    plt.bar(coach_stats['coachID'], coach_stats['Total Wins'], label='Total Wins', alpha=0.6)
    plt.bar(coach_stats['coachID'], coach_stats['Total Losses'], label='Total Losses', alpha=0.6, bottom=coach_stats['Total Wins'])
    plt.bar(coach_stats['coachID'], coach_stats['Postseason Wins'], label='Postseason Wins', alpha=0.6, color='orange')
    plt.bar(coach_stats['coachID'], coach_stats['Postseason Losses'], label='Postseason Losses', alpha=0.6, color='red', bottom=coach_stats['Postseason Wins'])

    plt.title('Coaches Performance: Wins and Losses')
    plt.xlabel('Coach ID')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{output_dir_plots}/coaches_performance.png')
    plt.close()

# Method used to generate graphs to better analyze a players performance
def player_performance_analysis(df: pd.DataFrame):
    # Aggregate data in regular and postseason for total points, assists, and rebounds
    df['total_rebounds'] = df['rebounds']
    df['PostTotalRebounds'] = df['PostRebounds']
    
    total_points = df.groupby('playerID')['points'].sum()
    total_assists = df.groupby('playerID')['assists'].sum()
    total_rebounds = df.groupby('playerID')['rebounds'].sum()
    total_PostPoints = df.groupby('playerID')['PostPoints'].sum()
    total_PostAssists = df.groupby('playerID')['PostAssists'].sum()
    total_PostTotalRebounds = df.groupby('playerID')['PostTotalRebounds'].sum()

    # Make an unique dataframe to study the relevant attributes 
    player_stats = pd.DataFrame({
        'playerID': total_points.index,
        'points': total_points,
        'assists': total_assists,
        'total_rebounds': total_rebounds,
        'PostPoints': total_PostPoints,
        'PostAssists': total_PostAssists,
        'PostTotalRebounds': total_PostTotalRebounds
    })

    player_stats.set_index('playerID', inplace=True) 

    # Create a plot for analyzing performance of players by points
    player_stats = player_stats.sort_values(by='points', ascending=False).head(50) # Sort by total points in regular season and limit by 50 players
    _, ax = plt.subplots(figsize=(12, 8))
    player_stats[['points', 'PostPoints']].plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_title('Player Performance by Points: Regular Season vs Postseason', fontsize=16)
    ax.set_ylabel('Total Points', fontsize=14)
    ax.set_xlabel('Player ID', fontsize=14)
    ax.legend(['Regular Season', 'Postseason'], loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/player_performance_points.png')
    plt.close()

    # Create a plot for analyzing performance of players by assists
    player_stats = player_stats.sort_values(by='assists', ascending=False).head(50) # Sort by total points in regular season and limit by 50 players
    _, ax = plt.subplots(figsize=(12, 8))
    player_stats[['assists', 'PostAssists']].plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_title('Player Performance by Assists: Regular Season vs Postseason', fontsize=16)
    ax.set_ylabel('Total Assists', fontsize=14)
    ax.set_xlabel('Player ID', fontsize=14)
    ax.legend(['Regular Season', 'Postseason'], loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/player_performance_assists.png')
    plt.close()

    # Create a plot for analyzing performance of players by rebounds
    player_stats = player_stats.sort_values(by='total_rebounds', ascending=False).head(50) # Sort by total points in regular season and limit by 50 players
    _, ax = plt.subplots(figsize=(12, 8))
    player_stats[['total_rebounds', 'PostTotalRebounds']].plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_title('Player Performance by Rebounds: Regular Season vs Postseason', fontsize=16)
    ax.set_ylabel('Total Rebounds', fontsize=14)
    ax.set_xlabel('Player ID', fontsize=14)
    ax.legend(['Regular Season', 'Postseason'], loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/player_performance_rebounds.png')
    plt.close()


# Method used to generate a graph to analyze teams player score over the years
def teams_player_score(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/teams/yearTeams_{year}.csv")
        
        df_year['year'] = year
        
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    df_grouped = df_all_years.groupby(['franchID', 'year'])['team_players_score'].sum().unstack(fill_value=0)

    ax = df_grouped.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='tab20')
    plt.title('Teams Performance: Player Scores Over the Years')
    plt.xlabel('Team ID')
    plt.ylabel('Player Score')
    plt.legend(title='Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/teams_player_score.png')
    plt.close()

# Method used to generate a graph to illustrate the wins and losses of each team over the years
def teams_wins_losses(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/teams/yearTeams_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    df_grouped_wins = df_all_years.groupby(['franchID', 'year'])['won'].sum().unstack(fill_value=0)
    df_grouped_losses = df_all_years.groupby(['franchID', 'year'])['lost'].sum().unstack(fill_value=0)

    ax = df_grouped_wins.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='Blues', alpha=0.7, position=1, width=0.4)
    df_grouped_losses.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='Reds', alpha=0.7, position=0, width=0.4, ax=ax)

    plt.title('Teams Wins and Losses Over the Years')
    plt.xlabel('Team ID')
    plt.ylabel('Number of Wins/Losses')
    plt.legend(title='Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/teams_wins_losses.png')
    plt.close()

# Method used to generate a graph to illustrate the average points scored and conceded by each team over the years
def average_points_scored_conceded(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/teams/yearTeams_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    df_grouped_scored = df_all_years.groupby(['franchID', 'year'])['o_pts'].mean().unstack(fill_value=0).sort_index(axis=1)
    df_grouped_conceded = df_all_years.groupby(['franchID', 'year'])['d_pts'].mean().unstack(fill_value=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(20, 10))
    df_grouped_scored.T.plot(ax=ax, marker='o', linewidth=2, alpha=0.8, cmap='tab20')
    df_grouped_conceded.T.plot(ax=ax, marker='x', linewidth=2, linestyle='dashed', alpha=0.8, cmap='tab20c')
    ax.set_title('Average Points Scored vs. Conceded by Teams Over the Years', fontsize=18)
    ax.set_xlabel('Year', fontsize=15)
    ax.set_ylabel('Average Points', fontsize=15)
    ax.legend(title='Team (Scored / Conceded)', fontsize=12, title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)

    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/average_points_scored_conceded.png')
    plt.close()

# Method to generate a graph to determine the efficiency of the 5 best players (according to player_score)
def player_efficiency_over_time(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    top_players = df_all_years.groupby('playerID')['player_score'].sum().nlargest(5).index

    df_all_years['PPG'] = df_all_years['points'] / df_all_years['GP']
    df_all_years['APG'] = df_all_years['assists'] / df_all_years['GP']
    df_all_years['RPG'] = df_all_years['rebounds'] / df_all_years['GP']

    _, ax = plt.subplots(figsize=(14, 10))
    colors = ['b', 'g', 'r', 'c', 'm']

    for i, player in enumerate(top_players):
        player_data = df_all_years[df_all_years['playerID'] == player]
        
        ax.plot(player_data['year'], player_data['PPG'], marker='o', label=f'{player} - PPG', color=colors[i])
        ax.plot(player_data['year'], player_data['APG'], marker='x', label=f'{player} - APG', color=colors[i], linestyle='dashed')
        ax.plot(player_data['year'], player_data['RPG'], marker='s', label=f'{player} - RPG', color=colors[i], linestyle='dotted')

    ax.set_title('Efficiency Analysis Over Time for Top 5 Players', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Average Per Game', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/efficiency_over_time_top_5_players.png')
    plt.close()

# Method to generate a graph to check if a player experience influences their efficiency
def experience_vs_efficiency(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    avg_efficiency = df_all_years.groupby('playerID')['efficiency'].mean()
    experience = df_all_years.groupby('playerID')['experience'].max()

    player_stats = pd.DataFrame({
        'efficiency': avg_efficiency,
        'experience': experience
    })

    _, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(player_stats['experience'], player_stats['efficiency'], color='b', alpha=0.6)
    ax.set_title('Player Experience vs. Efficiency', fontsize=16)
    ax.set_xlabel('Total Experience (Years)', fontsize=14)
    ax.set_ylabel('Average Efficiency', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/experience_vs_efficiency.png')
    plt.close()

# Method to generate a graph to check how a players efficiency changes over the course of their career (used with 10 of the most experienced players)
def player_efficiency_variance(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    # Filter for players with the most experience (more than 8 years of experience)
    career_data = df_all_years.groupby('playerID')['year'].count()
    long_career_players = career_data[career_data > 8].index
    df_filtered = df_all_years[df_all_years['playerID'].isin(long_career_players)]

    _, ax = plt.subplots(figsize=(14, 10))
    for player in long_career_players[:10]:  # Limit to top 10 players
        player_data = df_filtered[df_filtered['playerID'] == player]
        ax.plot(player_data['year'], player_data['efficiency'], marker='o', label=player)

    ax.set_title('Player Efficiency Variance Over Career', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Efficiency', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/player_efficiency_variance.png')
    plt.close()

# Method to generate a graph to analyze how a players individual stats contributes to the teams overall performance, such as points scored versus team total points (used top 10 players that contributed to scoring points)
def player_team_contribution(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    team_points = df_all_years.groupby(['tmID', 'year'])['points'].sum().reset_index()
    player_points = df_all_years.groupby(['playerID', 'tmID', 'year'])['points'].sum().reset_index()

    player_team_data = pd.merge(player_points, team_points, on=['tmID', 'year'], suffixes=('_player', '_team'))
    player_team_data['contribution_pct'] = (player_team_data['points_player'] / player_team_data['points_team']) * 100

    top_players = player_team_data.groupby('playerID')['contribution_pct'].mean().nlargest(10).index
    contribution_data = player_team_data[player_team_data['playerID'].isin(top_players)]

    _, ax = plt.subplots(figsize=(14, 10))
    for player in top_players:
        player_data = contribution_data[contribution_data['playerID'] == player]
        ax.plot(player_data['year'], player_data['contribution_pct'], marker='o', label=player)

    ax.set_title('Player Contribution to Team Points Over Time', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Contribution Percentage', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/player_team_contribution.png')
    plt.close()

# Method to generate a graph to show the efficiency of the top 10 players in shooting by comparing the number of field goals made versus attempted over the years
def fg_made_vs_attempted(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    # Top 10 players by player_score
    top_players = df_all_years.groupby('playerID')['player_score'].sum().nlargest(5).index
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(top_players)))

    _, ax = plt.subplots(figsize=(14, 10))
    for i, player in enumerate(top_players):
        player_data = df_all_years[df_all_years['playerID'] == player]
        ax.plot(player_data['year'], player_data['fgMade'], marker='o', label=f'{player} - FG Made', linestyle='solid', color=colors[i])
        ax.plot(player_data['year'], player_data['fgAttempted'], marker='x', label=f'{player} - FG Attempted', linestyle='dashed', color=colors[i])

    ax.set_title('Field Goals Made vs. Attempted Over Time for Top 5 Players', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Field Goals', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/fg_made_vs_attempted.png')
    plt.close()


# Method to generate a graph to compare the defensive statistics of top players over the years (steals and blocks)
def steals_vs_blocks(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    top_players = df_all_years.groupby('playerID')['player_score'].sum().nlargest(5).index
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(top_players)))

    _, ax = plt.subplots(figsize=(14, 10))
    for i, player in enumerate(top_players):
        player_data = df_all_years[df_all_years['playerID'] == player]
        ax.plot(player_data['year'], player_data['steals'], marker='o', label=f'{player} - Steals', linestyle='solid', color=colors[i])
        ax.plot(player_data['year'], player_data['blocks'], marker='x', label=f'{player} - Blocks', linestyle='dashed', color=colors[i])

    ax.set_title('Steals vs. Blocks Over Time for Top 5 Players', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/steals_vs_blocks.png')
    plt.close()



# Method to generate a graph to determine the top 20 players registered
def top20_all_time_players(years):
    df_all_years = pd.DataFrame()

    for year in range(1, years):
        df_year = pd.read_csv(f"./data/year/players/yearPlayers_{year}.csv")
        df_year['year'] = year
        df_all_years = pd.concat([df_all_years, df_year], ignore_index=True)

    top_players = df_all_years.groupby('playerID')['player_score'].sum().nlargest(20)
    top_players = top_players.sort_values(ascending=True)

    _, ax = plt.subplots(figsize=(14, 10))
    top_players.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title('Top 20 All-Time Players by Player Score', fontsize=16)
    ax.set_xlabel('Total Player Score', fontsize=14)
    ax.set_ylabel('Player ID', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/top20_all_time_players.png')
    plt.close()

# Method used to generate a graph to illustrate the away and home wins of each team over the years
def home_away_wins(teams):
    df_grouped_home_wins = teams.groupby(['franchID', 'year'])['homeW'].sum().unstack(fill_value=0)
    df_grouped_away_wins = teams.groupby(['franchID', 'year'])['awayW'].sum().unstack(fill_value=0)

    custom_colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(df_grouped_home_wins.columns))) # Darker shades of purple
    custom_colors_away = plt.cm.Oranges(np.linspace(0.4, 0.9, len(df_grouped_away_wins.columns))) # Darker shades of orange

    ax = df_grouped_home_wins.plot(kind='bar', figsize=(20, 8), color=custom_colors, alpha=0.8, position=1, width=0.4)
    df_grouped_away_wins.plot(kind='bar', ax=ax, color=custom_colors_away, alpha=0.8, position=0, width=0.4)

    plt.title('Home vs. Away Wins Analysis Over the Years', fontsize=18)
    plt.xlabel('Team ID', fontsize=15)
    plt.ylabel('Number of Wins', fontsize=15)
    plt.legend(title='Year', fontsize=12, title_fontsize='13', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/home_away_wins.png')
    plt.close()

# Method used to generate a graph to illustrate the top 20 best rookies
def top20_rookies(rookies):
    top_20_rookies = rookies.sort_values(by='player_score', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    plt.barh(top_20_rookies['playerID'], top_20_rookies['player_score'], color='skyblue')
    plt.xlabel('Player Score')
    plt.ylabel('Player ID')
    plt.title('Top 20 Rookies Based on Player Score')
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/top_20_rookies.png')
    plt.close()

# Method used to generate a heatmap to illustrate the correlation between the team stats
def top20_colleges(colleges):
    top_20_colleges = colleges.sort_values(by='Score', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    plt.barh(top_20_colleges['College'], top_20_colleges['Score'], color='skyblue')
    plt.xlabel('Player Score')
    plt.ylabel('Player ID')
    plt.title('Top 20 Colleges Based on Rookies Player Score')
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/top_20_colleges.png')
    plt.close()

# Method used to generate graph to illustrate the top 20 best rookies
def heatmap_team_stats(teams):
    numerical_columns = [
    'rank', 'o_fgm','o_ftm', 'o_3pm', 
    'o_reb', 'o_asts', 'd_asts', 'o_pf', 'o_stl', 'o_to', 
    'o_blk', 'o_pts', 'd_fgm', 'd_ftm', 'd_3pm', 
    'd_reb', 'd_pf', 'd_stl', 
    'd_to', 'd_blk', 'd_pts', 'won', 'lost', 'GP', 'homeW', 'homeL', 
    'awayW', 'awayL', 'confW', 'confL', 'min', 'attend'
    ]

    df_numerical = teams[numerical_columns]
    correlation_matrix = df_numerical.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Team Performance Statistics', fontsize=18)
    plt.xlabel('Statistics', fontsize=14)
    plt.ylabel('Statistics', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/heatmap_team_stats.png')
    plt.close()

def detect_outliers_iqr(teams):
    numerical_columns = [
        'rank', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 
        'o_oreb', 'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 
        'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 
        'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 
        'd_to', 'd_blk', 'd_pts', 'won', 'lost', 'GP', 'homeW', 'homeL', 
        'awayW', 'awayL', 'confW', 'confL', 'min',
    ]

    df_numerical = teams[numerical_columns]

    outliers = {}
    
    plt.figure(figsize=(16, 12))
    
    for idx, col in enumerate(df_numerical.columns):
        if df_numerical[col].dtype in ['int64', 'float64']:
            Q1 = df_numerical[col].quantile(0.25)
            Q3 = df_numerical[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers[col] = df_numerical[(df_numerical[col] < lower_bound) | (df_numerical[col] > upper_bound)]
            
            plt.subplot(7, 6, idx + 1)
            sns.boxplot(data=df_numerical, x=col)
            plt.title(col, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir_plots}/outliers_team_stats.png')
    plt.close()

    return outliers


def calculate_statistics(df):
    # List of statistics to analyze
    stats_columns = [
        'GP', 'GS', 'minutes', 'points', 'oRebounds', 'dRebounds', 'rebounds',
        'assists', 'steals', 'blocks', 'turnovers', 'PF', 'fgAttempted', 'fgMade',
        'ftAttempted', 'ftMade', 'threeAttempted', 'threeMade', 'dq', 'PostGP',
        'PostGS', 'PostMinutes', 'PostPoints', 'PostoRebounds', 'PostdRebounds',
        'PostRebounds', 'PostAssists', 'PostSteals', 'PostBlocks', 'PostTurnovers',
        'PostPF', 'PostfgAttempted', 'PostfgMade', 'PostftAttempted', 'PostftMade',
        'PostthreeAttempted', 'PostthreeMade', 'PostDQ'
    ]
    summary = {}

    for stat in stats_columns:
        if stat in df.columns:
            summary[stat] = {
                'mean': df[stat].mean().round(1),
                'max': df[stat].max(),
                'min': df[stat].min(),
                'std_dev': df[stat].std().round(1),
                'median': df[stat].median()
            }
        else:
            print(f"Warning: Column '{stat}' not found in the DataFrame.")

    statistics_df =  pd.DataFrame(summary).T

    fig, ax = plt.subplots(figsize=(20, len(statistics_df) * 0.8))
    ax.axis('off')
    table = ax.table(
        cellText=statistics_df.values,
        colLabels=statistics_df.columns,
        rowLabels=statistics_df.index,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(23)
    table.auto_set_column_width(col=list(range(len(statistics_df.columns))))

    # Adjust padding for rows and columns
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')  # Optional: Set cell borders
        cell.set_linewidth(1)  # Optional: Adjust the line width of borders
        cell.set_fontsize(23)  # Set font size for each cell
        cell.set_facecolor('white')  # Optional: Set cell background color
        
        # Increase padding by adjusting the width and height of the cell
        if i == 0:  # For column header row, apply more padding for readability
            cell.set_height(0.02)  # Adjust row height
            cell.set_width(0.02)  # Adjust column width
        else:
            cell.set_height(0.02)  # Adjust row height for non-header rows
            cell.set_width(0.02)  # Adjust column width

    plt.savefig(f'{output_dir_plots}/statistics_summary.png')
    plt.close()

def create_3d_surface_plot(data, year_col='year', o_pts_col='o_pts', d_pts_col='d_pts', 
                           tmTRB_col='tmTRB', opptmTRB_col='opptmTRB', x_interval=0.5, y_interval=1, z_interval=1):
    # Extract the data for plotting
    years = data[year_col]
    o_pts = data[o_pts_col]
    d_pts = data[d_pts_col]
    tmTRB = data[tmTRB_col]
    opptmTRB = data[opptmTRB_col]

    # Calculate y (o_pts - d_pts) and z (tmTRB - opptmTRB)
    y = o_pts - d_pts
    z = tmTRB - opptmTRB

    # Create a mesh grid for the plot
    x = np.arange(min(years), max(years) + x_interval, x_interval)  # Adjust x interval
    y_grid = np.arange(min(y), max(y) + y_interval, y_interval)  # Adjust y interval
    x, y_grid = np.meshgrid(x, y_grid)

    # Calculate z as a function of x and y (using the extracted data to interpolate values)
    z_grid = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Interpolate z values based on the current x, y_grid positions
            # Here, we're simply calculating the mean value of z for points near (x[i,j], y_grid[i,j])
            z_grid[i, j] = np.mean(z[(years >= x[i, j] - x_interval / 2) & 
                                     (years <= x[i, j] + x_interval / 2) & 
                                     (y >= y_grid[i, j] - y_interval / 2) & 
                                     (y <= y_grid[i, j] + y_interval / 2)])
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y_grid, z_grid, cmap='viridis')

    # Add labels and title
    ax.set_xlabel(year_col)
    ax.set_ylabel(f'{o_pts_col} - {d_pts_col}')
    ax.set_zlabel(f'{tmTRB_col} - {opptmTRB_col}')
    ax.set_title('3D Surface Plot of Year, Point Difference, and Rebound Difference')

    # Show the plot
    plt.savefig(f'{output_dir_plots}/3d_plot.png')
    plt.close()

# Method used for data analyzis
def data_analysis():
    awards_players = pd.read_csv('./data/clean/clean_awards_players.csv')
    coaches = pd.read_csv('./data/clean/clean_coaches.csv')
    players_teams = pd.read_csv('./data/clean/clean_players_teams.csv')
    players = pd.read_csv('./data/clean/clean_players.csv')
    series_post = pd.read_csv('./data/clean/clean_series_post.csv')
    teams_post = pd.read_csv('./data/clean/clean_teams_post.csv')
    teams = pd.read_csv('./data/clean/clean_teams.csv')
    rookies = pd.read_csv(f"./data/year/rookie/rookies.csv")
    colleges = pd.read_csv(f"./data/year/rookie/college_scores.csv")

    # Create folder 'plots' to save the plots
    if not os.path.exists(output_dir_plots):
        os.makedirs(output_dir_plots)
    
    outliers = detect_outliers_iqr(teams)
    #print(outliers)
    player_awards(awards_players)
    coach_analysis(coaches)
    player_performance_analysis(players_teams)

    numberYears = len(teams['year'].unique()) + 1
    teams_player_score(numberYears)
    teams_wins_losses(numberYears)
    average_points_scored_conceded(numberYears)

    player_efficiency_over_time(numberYears)
    top20_all_time_players(numberYears)
    experience_vs_efficiency(numberYears)
    player_efficiency_variance(numberYears)
    player_team_contribution(numberYears)
    fg_made_vs_attempted(numberYears)
    steals_vs_blocks(numberYears)

    home_away_wins(teams)
    top20_rookies(rookies)
    top20_colleges(colleges)
    heatmap_team_stats(teams)

    calculate_statistics(players_teams)
    create_3d_surface_plot(teams)

if __name__ == "__main__":
    data_analysis()
