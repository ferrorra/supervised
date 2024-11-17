from joblib import dump, load
import os
def evaluate_model_with_sampling(model_func, model_name, X, y):
    """
    Evaluates a model with different sampling techniques (oversampling, undersampling, combination).
    It adjusts the sampling parameters dynamically and calls the provided model function.

    Parameters:
    - model_func: Callable, the function of the model to be evaluated.
    - model_name: str, the name of the model.
    - X: Features (DataFrame or array-like).
    - y: Target variable (Series or array-like).
    
    Returns:
    - results: dict, containing the best parameters for each sampling method.
    - best_method: str, the sampling method with the best performance.
    """
    from sklearn.metrics import accuracy_score

    results = {}
    sampling_methods = {
        "Oversampling": {"sampling_type": "oversampling", "method": "SMOTE"},
        "Undersampling": {"sampling_type": "undersampling", "method": "NearMiss"},
        "Combination": {"sampling_type": "combination", "method": "SMOTETomek"}
    }

    print(f"\nEvaluating {model_name}...")

    best_accuracy = -1
    best_method = None

    for sampling_type, params in sampling_methods.items():
        # Adjust model function to include sampling parameters
        print(f"Applying {params['method']} for {sampling_type.lower()}...")
        model, best_params = model_func(
            X, y, sampling_type=params["sampling_type"], method=params["method"]
        )
        print(f"{model_name} ({sampling_type}): Best Params: {best_params}")

        # Evaluate model on the same dataset (X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy for {model_name} ({sampling_type}): {accuracy:.4f}")

        # Store the results
        results[sampling_type] = {"Best Params": best_params, "Accuracy": accuracy}

        # Check if this is the best method
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = sampling_type

    print(f"\nBest Sampling Method for {model_name}: {best_method} with Accuracy: {best_accuracy:.4f}")
    return results, best_method
