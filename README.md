# MachineLearningBasket

Player Features:
- awards_score: Score obtido quanto a prémios recebidos
- regular_season_score: Score obtido quanto a estatísticas obtidas durante a regular season
- post_season_score: Score obtido quanto a estatísticas obtidas durante a post season
- FG%: Percentagem de field goals
- 3P%: Percentagem de conversão de 3 pointers
- FT%: Percentagem de conversão de free throws
- efficiency: Score da conversão de lançamentos feitos
- total_experience: Score da experiência de um jogador (anos jogados e prémios obtidos)
- player_score: Score de qualidade do jogador (associado ao regular season score obtido, ao post season score obtido, à efficiency e à total_experiency)
- pre_season_player_score: Média de player scores obtidos pelo jogador nos anos anteriores

TeamFeatures:
- o_score: Offensive Score do ano atual
- d_score: Defensive Score do ano atual
- pf_score: Performance Score do ano atual (wins/losses ratio,...)
- team_stats_score: Score das stats de uma equipa no ano atual
- pre_season_team_stats_score: Média do score das stats da equipa ao longo dos anos. N inclui o ano atual
- all_time_team_players_score: Média do player score de cada equipa ao longo dos anos. N inclui o ano atual
- current_team_players_score: Player Score do plantel atual, antes do início da época. Para determinação do player score de cada jogador, é utilizado o atributo pre_season_player_score (média do player score obtido nos anos anteriores)
- team_players_score: Média do player score de cada equipa ao longo dos anos, incluindo o ano atual
- team_score: (0.5 * team_players_score) + (0.5 * team_stats_score)


## Result of the models
### Random Forest Classifier
#### pre_season_team_stats_score
- Year 8: 0.588
- Year 9: 0.729
- Year 10: 0.575   

#### current_team_players_score
- Year 8: 0.375
- Year 9: 0.531
- Year 10: 0.625   

#### all_time_team_players_score
- Year 8: 0.4
- Year 9: 0.521
- Year 10: 0.662

#### pre_season_team_score
- Year 8: 0.5
- Year 9: 0.562
- Year 10: 0.325

### XGB Classifier
#### pre_season_team_stats_score
- Year 8: 0.55
- Year 9: 0.76
- Year 10: 0.538   

#### current_team_players_score
- Year 8: 0.425
- Year 9: 0.688
- Year 10: 0.588   

#### all_time_team_players_score
- Year 8: 0.438
- Year 9: 0.615
- Year 10: 0.588

#### pre_season_team_score
- Year 8: 0.512
- Year 9: 0.708
- Year 10: 0.525

### Logistic Regression
#### pre_season_team_stats_score
- Year 8: 0.538
- Year 9: 0.854
- Year 10: 0.512   

#### current_team_players_score
- Year 8: 0.4
- Year 9: 0.667
- Year 10: 0.9 

#### all_time_team_players_score
- Year 8: 0.55
- Year 9: 0.792
- Year 10: 0.65

#### pre_season_team_score
- Year 8: 0.55
- Year 9: 0.833
- Year 10: 0.587

### SVM
#### pre_season_team_stats_score
- Year 8: 0.538
- Year 9: 0.854
- Year 10: 0.512   

#### current_team_players_score
- Year 8: 0.6
- Year 9: 0.667
- Year 10: 0.1   

#### all_time_team_players_score
- Year 8: 0.55
- Year 9: 0.792
- Year 10: 0.65

#### pre_season_team_score
- Year 8: 0.55
- Year 9: 0.833
- Year 10: 0.587