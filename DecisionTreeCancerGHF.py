import numpy as np
import pandas as pd # Included for general data science context, and for readability of some outputs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn import datasets # To load the Breast Cancer dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # For potential direct evaluation

def load_breast_cancer_data():
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.

    This dataset is used for binary classification (malignant vs. benign tumors).
    It contains 569 samples with 30 features, computed from a digitized image
    of a fine needle aspirate (FNA) of a breast mass.

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data, target,
                             feature_names, target_names, and description.
    """
    print("Loading Breast Cancer Wisconsin (Diagnostic) dataset...")
    data = datasets.load_breast_cancer()

    # Print some information for better understanding
    print("\n--- Breast Cancer Data Overview ---")
    print(f"Number of samples: {data.data.shape[0]}")
    print(f"Number of features: {data.data.shape[1]}")
    print("Feature names (first 5):", data.feature_names[:5])
    print("Target names (classes):", data.target_names) # 'malignant', 'benign'
    print(f"Class distribution: {np.bincount(data.target)}") # Count of samples in each class
    print("\nFirst 5 rows of features:")
    print(data.data[:5])
    print("\nFirst 5 target labels:")
    print(data.target[:5]) # 0 for malignant, 1 for benign
    return data

def prepare_features_target(dataset):
    """
    Prepares the features (X) and target (y) arrays from the loaded dataset.

    Args:
        dataset (sklearn.utils.Bunch): The loaded dataset.

    Returns:
        tuple: A tuple containing (features (X), target (y)).
    """
    features = dataset.data
    target = dataset.target
    return features, target

def split_data(features, target, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The feature data.
        target (numpy.ndarray): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Ensures reproducibility.

    Returns:
        tuple: A tuple containing (feature_train, feature_test, target_train, target_test).
    """
    print(f"\nSplitting data into training ({(1-test_size)*100:.0f}%) and testing ({test_size*100:.0f}%) sets...")
    # `stratify=target` ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset. This is crucial for classification tasks.
    feature_train, feature_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    print(f"Training set size: {len(feature_train)} samples")
    print(f"Testing set size: {len(feature_test)} samples")
    return feature_train, feature_test, target_train, target_test

def tune_decision_tree_hyperparameters(feature_train, target_train, random_state=42):
    """
    Performs GridSearchCV to find the optimal hyperparameters for a Decision Tree Classifier.

    GridSearchCV exhaustively searches over specified parameter values for an estimator,
    using cross-validation to evaluate the performance of each combination.

    Args:
        feature_train (numpy.ndarray): Training features.
        target_train (numpy.ndarray): Training target.
        random_state (int): Seed for reproducibility of the DecisionTreeClassifier.

    Returns:
        sklearn.model_selection.GridSearchCV: The fitted GridSearchCV object,
                                              containing the best estimator.
    """
    print("\n--- Tuning Decision Tree Hyperparameters using GridSearchCV ---")
    # Define the Decision Tree Classifier (estimator)
    # Set random_state for reproducibility of the tree's split decisions
    dt_classifier = DecisionTreeClassifier(random_state=random_state)

    # Define the parameter grid to search over
    # 'max_depth': The maximum depth of the tree. Controls overfitting;
    #              a deeper tree can model more complex relationships but might overfit.
    # 'criterion': The function to measure the quality of a split.
    #              'gini' for Gini impurity, 'entropy' for information gain, 'log_loss' (same as entropy).
    param_grid = {
        'max_depth': np.arange(1, 11), # Test depths from 1 to 10
        'criterion': ['gini', 'entropy'] # 'log_loss' is equivalent to 'entropy' for Decision Trees
    }

    # Initialize GridSearchCV
    # estimator: The model to tune (DecisionTreeClassifier)
    # param_grid: Dictionary of parameter names and values to test
    # refit=True: Refit the estimator with the best found parameters on the whole training dataset.
    #             This means `grid.best_estimator_` will be the final trained model.
    # cv: Number of folds for cross-validation (default is 5).
    # verbose: Controls the verbosity of the output. Higher values mean more messages.
    grid_search = GridSearchCV(
        estimator=dt_classifier,
        param_grid=param_grid,
        refit=True, # After finding best params, retrain on all training data
        cv=10,      # Use 10-fold cross-validation for parameter tuning
        verbose=1,  # Print progress
        scoring='accuracy' # Metric to optimize for
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(feature_train, target_train)

    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (accuracy): {grid_search.best_score_:.4f}")
    print(f"Best estimator (trained model): {grid_search.best_estimator_}")

    return grid_search

def evaluate_model_with_cross_validation(model_to_evaluate, features, target, cv_folds=10):
    """
    Evaluates the model's performance using cross-validation on the entire dataset.

    This function uses `cross_validate` which allows for multiple metrics and
    returns a dictionary of scores.

    Args:
        model_to_evaluate (sklearn.base.BaseEstimator): The estimator (e.g., the best_estimator_
                                                         from GridSearchCV or a simple model).
        features (numpy.ndarray): The full feature dataset.
        target (numpy.ndarray): The full target dataset.
        cv_folds (int): The number of folds for cross-validation.

    Returns:
        dict: A dictionary of scores from cross_validate.
    """
    print(f"\n--- Evaluating Model Performance using {cv_folds}-Fold Cross-Validation ---")
    # cross_validate returns a dictionary with scores for each fold.
    # 'test_score' contains the score on the test set for each fold.
    # 'fit_time' and 'score_time' give insights into computation time.
    cv_results = cross_validate(
        model_to_evaluate,
        features, # Use the full dataset here to get a generalizable estimate of performance
        target,
        cv=cv_folds,
        scoring='accuracy', # We are interested in accuracy
        return_train_score=False # No need for train scores in this case
    )

    print(f"Cross-validation test scores for each fold: {cv_results['test_score']}")
    mean_accuracy = np.mean(cv_results['test_score'])
    std_accuracy = np.std(cv_results['test_score'])

    print(f"\nMean Cross-validation Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy*2:.4f})") # Mean +/- 2 std_dev
    return cv_results

def make_predictions_on_test_set(model, feature_test, target_test, target_names):
    """
    Makes predictions on the held-out test set and prints detailed metrics.

    Args:
        model (sklearn.base.BaseEstimator): The trained model (e.g., best_estimator_ from GridSearchCV).
        feature_test (numpy.ndarray): The features of the test set.
        target_test (numpy.ndarray): The true labels of the test set.
        target_names (list): List of class names for readability.
    """
    print("\n--- Model Performance on Held-Out Test Set ---")
    test_predictions = model.predict(feature_test)

    # Accuracy Score
    accuracy = accuracy_score(target_test, test_predictions)
    print(f"Accuracy Score: {accuracy:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(target_test, test_predictions)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(target_test, test_predictions, target_names=target_names))

if __name__ == "__main__":
    # Define parameters for reproducibility and split size
    TEST_DATA_SPLIT_RATIO = 0.3
    RANDOM_SEED = 42 # Used for data splitting and Decision Tree
    CV_FOLDS_FOR_EVALUATION = 10

    # 1. Load the Breast Cancer Dataset
    cancer_data = load_breast_cancer_data()
    if cancer_data is None:
        exit() # Should not happen for built-in datasets

    # 2. Prepare Features (X) and Target (y)
    X, y = prepare_features_target(cancer_data)

    # 3. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=TEST_DATA_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )

    # 4. Tune Decision Tree Hyperparameters using GridSearchCV
    # This step trains the GridSearchCV object itself, performing internal CV on X_train.
    grid_search_result = tune_decision_tree_hyperparameters(X_train, y_train, random_state=RANDOM_SEED)

    # The best_estimator_ attribute of GridSearchCV is the best model, already fitted on X_train.
    best_dt_model = grid_search_result.best_estimator_

    # 5. Evaluate the Best Model's Generalization using Cross-Validation on the full dataset
    # This evaluates how well the *type* of model (with the best hyperparameters) generalizes
    # across different folds of the entire dataset.
    cv_evaluation_results = evaluate_model_with_cross_validation(best_dt_model, X, y, cv_folds=CV_FOLDS_FOR_EVALUATION)

    # 6. Evaluate the Best Model's Performance on the held-out Test Set
    # This gives a final, unbiased estimate of performance on completely unseen data.
    make_predictions_on_test_set(best_dt_model, X_test, y_test, cancer_data.target_names)

    print("\nScript execution complete.")
