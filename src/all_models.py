import os
import pandas as pd

def aggregate_predictions(folder_path):
    """
    Aggregates 'Predicted Playoff' (sum) and 'Predicted Probability' (mean) for each team
    across all files in a folder and saves the result as a CSV in the same folder.
    
    Args:
        folder_path (str): Path to the folder containing the CSV files.
        
    Returns:
        None
    """
    # Initialize dictionaries to store aggregated results
    playoff_sums = {}
    probability_totals = {}
    team_counts = {}

    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Check if the file is a CSV file
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            
            # Process each row in the DataFrame
            for index, row in df.iterrows():
                team = row["franchID"]
                predicted_probability = row["Predicted Probability"]
                
                playoff_sums[team] = playoff_sums.get(team, 0)
                
                # Aggregate total 'Predicted Probability' and count occurrences for mean calculation
                probability_totals[team] = probability_totals.get(team, 0) + predicted_probability
                team_counts[team] = team_counts.get(team, 0) + 1

    # Calculate mean 'Predicted Probability' for each team
    mean_probabilities = {team: probability_totals[team] / team_counts[team] for team in probability_totals}

    # Load conference of each tean
    teams_data = pd.read_csv("./data/year/teams/yearTeams_11.csv")
    team_conferences = teams_data.set_index("franchID")["confID"].to_dict()

    # Create a DataFrame with the aggregated results
    results = pd.DataFrame({
        "Team": playoff_sums.keys(),
        "Conference": [team_conferences[team] for team in playoff_sums.keys()],
        "Mean of Predicted Probability": [round(mean_probabilities[team], 3) for team in playoff_sums.keys()]
    })

    # Save the results to a new CSV file in the same folder
    output_file = os.path.join("./data/year/combined", "aggregated_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"File saved as {output_file}")

    # Get the top 4 teams by conference
    top_teams_by_conference = (
        results
        .sort_values(by=["Conference", "Mean of Predicted Probability"], ascending=[True, False])
        .groupby("Conference")
        .head(4)
    )

    # Print the top teams by conference
    print("Top 4 Teams by Conference:")
    print(top_teams_by_conference.to_string(index=False))