import cleaning
import organize_year
import data_preparation
import modeling
import all_models
import os

if __name__ == "__main__":
    cleaning.get_clean_data()
    organize_year.organizeByYear()
    organize_year.rookieScore()
    organize_year.finals_winners_playoff_analysis()
    cleaning.clean_test_year()
    organize_year.dataYear11()
    combined_team_data = data_preparation.combine_yearly_team_data()
    models = ["logistic_regression", "svm", "catboost"]
    models_regresion = ["linear_regression", "random_forest_regressor"]

    folder_path = "./data/year/combined/prediction"
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    #modeling.predict_for_year_without_metrics("random_forest", 11)
    #for model_regression in models_regresion:
    #    modeling.year_based_time_series_cv_regression(model_regression)
    for model in models:
        modeling.predict_for_year_without_metrics(model, 11)

    all_models.aggregate_predictions(folder_path)

