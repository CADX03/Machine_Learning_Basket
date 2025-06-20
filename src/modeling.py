import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Dictionary of available models
MODEL_REGISTRY = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "xgboost": lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "lightgbm": lambda: LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "svm": lambda: SVC(probability=True, kernel='linear', random_state=42),
    "catboost": lambda: CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42)
}

# Load and prepare data
def prepare_data():
    df = pd.read_csv('./data/year/combined/combined_team_data.csv')
    target = 'playoff'
    #features = ['franchise_experience', 'pre_season_team_stats_score', 'current_team_players_score', 'all_time_team_players_score', 'pre_season_team_score', 'team_players_score1']
    features = ['current_team_players_score']
    df[target] = df[target].apply(lambda x: 1 if x == 'Y' else 0)

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean', fill_value=0)  # Replace 'mean' with 'median' or 'constant' as needed
    df[features] = imputer.fit_transform(df[features])

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    #print("Class distribution before SMOTE:", y_train.value_counts())
    #print("Class distribution after SMOTE:", pd.Series(y_train_smote).value_counts())

    return df, features, target

# Evaluate model predictions
def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    top_8_indices = y_pred_proba.argsort()[-8:][::-1]
    y_pred = [1 if idx in top_8_indices else 0 for idx in range(len(y_pred_proba))]
    auc = roc_auc_score(y_test, y_pred_proba).round(3)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    classification_metrics = {
        "AUC-ROC": auc,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "Classification Report": classification_report(y_test, y_pred)
    }
    return y_pred_proba, y_pred, classification_metrics

# Plot ROC curve
def plot_roc_curve(y_test, y_pred_proba, test_year, auc):
    plots_dir = './data/plots/roc_curves/'

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Year {test_year} ROC (AUC = {auc:.2f})', color='b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Year {test_year}')
    plt.legend(loc='lower right')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f'{plots_dir}roc_curve_year{test_year}.png')
    plt.close()

# Train the model using Random Forest
def train_model(model_name, X_train, y_train):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)
    return model

# Cross-validate by year
def year_based_time_series_cv(model_name):
    df, features, target = prepare_data()
    years = sorted(df['year'].unique())
    fold_results = []
    min_training_years = 9

    for fold, i in enumerate(range(min_training_years, len(years)), start=1):
        train_years, test_year = years[:i], years[i]
        train_df, test_df = df[df['year'].isin(train_years)], df[df['year'] == test_year]
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # Train model
        model = train_model(model_name, X_train, y_train)
        
        # Evaluate model and gather metrics
        y_pred_proba, y_pred, metrics = evaluate_model(model, X_test, y_test)
        
        # Print evaluation results
        print(f"Fold {fold} - Year {test_year} - Model: {model_name} - AUC-ROC Score: {metrics['AUC-ROC']}")
        print("Classification Report:\n", metrics["Classification Report"])
        print(f"Confusion Matrix: TN={metrics['True Negatives']}, FP={metrics['False Positives']}, FN={metrics['False Negatives']}, TP={metrics['True Positives']}\n")
        
        # Display predictions vs actual results
        retrieve_predictions(test_df, y_test, y_pred_proba, y_pred, fold, test_year, model_name)
        
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, test_year, metrics["AUC-ROC"])
        
        fold_results.append({
            "Fold": fold,
            "Year": test_year,
            "AUC-ROC": metrics["AUC-ROC"],
            "True Negatives": metrics["True Negatives"],
            "False Positives": metrics["False Positives"],
            "False Negatives": metrics["False Negatives"],
            "True Positives": metrics["True Positives"]
        })
    combined_cv_results(fold_results, model_name)

# Display teams prediction vs actual results
def retrieve_predictions(test_df, y_test, y_pred_proba, y_pred, fold, test_year, model_name):
    predictions_df = pd.DataFrame()
    predictions_path = f'./data/year/combined/prediction/model_{model_name}.csv'

    predictions_df = test_df[['tmID', 'name']].copy()
    predictions_df['Actual Playoff'] = y_test.values
    predictions_df['Predicted Probability'] = y_pred_proba
    predictions_df['Predicted Playoff'] = y_pred
    predictions_df.to_csv(predictions_path, index=False)

    results_df = test_df[['tmID', 'name']].copy()
    results_df['Actual Playoff'] = y_test.values
    results_df['Predicted Probability'] = y_pred_proba
    results_df['Predicted Playoff'] = y_pred
    print(f"Fold {fold} - Year {test_year} - Predictions vs. Actual:")
    print(results_df[['name', 'tmID', 'Actual Playoff', 'Predicted Probability', 'Predicted Playoff']].to_string(index=False), "\n")
    

# Cross-validation results
def combined_cv_results(fold_results, model_used):
    results_path = f'./data/year/combined/result/model_result_{model_used}.csv'

    results_df = pd.DataFrame(fold_results)
    avg_auc = results_df['AUC-ROC'].mean()
    print(f"\nAverage AUC-ROC across all folds: {avg_auc:.3f}")
    results_df.to_csv(results_path, index=False)

def predict_for_year_without_metrics(model_name, prediction_year):
    df, features, target = prepare_data()
    years = sorted(df['year'].unique())
    
    # Ensure the prediction year is valid
    if prediction_year not in years:
        raise ValueError(f"Year {prediction_year} not found in data.")
    
    train_years = [year for year in years if year < prediction_year]
    if not train_years:
        raise ValueError(f"Not enough training data before year {prediction_year}.")
    
    train_df = df[df['year'].isin(train_years)]
    test_df = df[df['year'] == prediction_year][['franchID', 'name', 'confID'] + features]
    #print(test_df)

    X_train, y_train = train_df[features], train_df[target]
    X_test = test_df[features]

    # Train the model
    model = train_model(model_name, X_train, y_train)

    # Predict probabilities for test year
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Save predictions
    predictions_df = test_df[['franchID', 'name', 'confID']].copy()
    predictions_df['Predicted Probability'] = y_pred_proba.round(3)

    predictions_path = f'./data/year/combined/prediction/predictions_year{prediction_year}_model_{model_name}.csv'
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"Predicted probabilities for year {prediction_year} saved to {predictions_path}.")

# Handle Linear Regression
def evaluate_model_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R-squared (coefficient of determination)
    
    # Print evaluation results
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R-squared: {r2:.3f}")
    
    return y_pred, mse, r2

def train_model_regression(model_name, X_train, y_train):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)
    return model

def plot_predictions_vs_actual(y_test, y_pred, test_year, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actuals (Year {test_year} - {model_name})')
    plt.savefig(f'./data/plots/regresion/{model_name}_predictions_vs_actual_{test_year}.png')
    plt.close()

def year_based_time_series_cv_regression(model_name):
    df, features, target = prepare_data()
    years = sorted(df['year'].unique())
    fold_results = []
    min_training_years = 9

    for fold, i in enumerate(range(min_training_years, len(years)), start=1):
        train_years, test_year = years[:i], years[i]
        train_df, test_df = df[df['year'].isin(train_years)], df[df['year'] == test_year]
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # Train model
        model = train_model_regression(model_name, X_train, y_train)
        
        # Evaluate model and gather metrics
        y_pred, mse, r2 = evaluate_model_regression(model, X_test, y_test)
        
        # Print evaluation results
        print(f"Fold {fold} - Year {test_year} - Model: {model_name} - MSE: {mse:.3f} - R2: {r2:.3f}")
        
        # Plot predictions vs actual
        plot_predictions_vs_actual(y_test, y_pred, test_year, model_name)
        
        fold_results.append({
            "Fold": fold,
            "Year": test_year,
            "MSE": mse,
            "R2": r2
        })
    combined_cv_results_regression(fold_results, model_name)

# Cross-validation results
def combined_cv_results_regression(fold_results, model_used):
    results_path = f'./data/year/combined/result/model_result_{model_used}.csv'

    # Convert the fold results to a DataFrame
    results_df = pd.DataFrame(fold_results)

    # Calculate average MSE and R2 across all folds
    avg_mse = results_df['MSE'].mean()
    avg_r2 = results_df['R2'].mean()

    # Print average performance metrics
    print(f"\nAverage Mean Squared Error (MSE) across all folds: {avg_mse:.3f}")
    print(f"Average R-squared (R²) across all folds: {avg_r2:.3f}")

    # Save the detailed results to a CSV file
    results_df.to_csv(results_path, index=False)