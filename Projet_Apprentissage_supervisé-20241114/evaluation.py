from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import accuracy_score  # Ensure this is imported

import numpy as np

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
    results = {}
    sampling_methods = {
        "Oversampling": {"sampling_type": "oversampling", "method": "SMOTE"},
        "Undersampling": {"sampling_type": "undersampling", "method": "NearMiss"},
        "Combination": {"sampling_type": "combination", "method": "SMOTETomek"}
    }

    print(f"\nEvaluating {model_name}...")

    best_accuracy = -1
    best_method = None

    # Initialize OneHotEncoder for categorical encoding
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    
    if len(categorical_cols) > 0:
        X_encoded = pd.DataFrame(
            encoder.fit_transform(X[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index
        )
        # Combine numeric features with encoded features
        X_encoded = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)
    else:
        X_encoded = X.copy()

    for sampling_type, params in sampling_methods.items():
        print(f"Applying {params['method']} for {sampling_type.lower()}...")

        # Call the model function with balanced data
        model, best_params = model_func(
            X_encoded, y, sampling_type=params["sampling_type"], method=params["method"]
        )
        print(f"{model_name} ({sampling_type}): Best Params: {best_params}")

        # Evaluate model on the same dataset (X, y)
        y_pred = model.predict(X_encoded)
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


from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def full_analysis_with_visuals(
    model_name,
    model_func,
    results,
    best_method,
    X_test,
    y_test,
    encoder,
    title="Model Performance Analysis",
):
    """
    Performs a full analysis of the model with detailed visualizations:
    - ROC curve
    - AUC score
    - Precision-Recall Curve
    - Confusion Matrix
    - Classification Report
    
    Parameters:
    - model_name: str, name of the model.
    - model_func: Callable, the function of the model.
    - results: dict, containing the best parameters for each sampling method.
    - best_method: str, the sampling method with the best performance.
    - X_test: Features for testing (DataFrame or array-like).
    - y_test: Target variable for testing (Series or array-like).
    - encoder: OneHotEncoder used during training.
    - title: str, Title of the overall analysis.
    
    Returns:
    - None, generates visualizations.
    """
    # Transform test data with the same encoder used during training
    if encoder:
        categorical_cols = X_test.select_dtypes(include=["object", "category"]).columns
        X_encoded = pd.DataFrame(
            encoder.transform(X_test[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X_test.index,
        )
        X_test = pd.concat([X_test.drop(categorical_cols, axis=1), X_encoded], axis=1)

    # Map y_test values to 0 and 1 if they are in {1, 2}
    print("Mapping target labels {1, 2} to {0, 1} for compatibility...")
    y_test = np.where(y_test == 1, 0, 1)

    # Retrieve the best parameters and train the model
    best_params = results[best_method]["Best Params"]
    print(f"\nTraining the best model ({best_method}) with parameters: {best_params}")
    model = model_func(X_test, y_test, sampling_type=None, method=None)[0]

    # Predict probabilities and labels
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    # ROC and AUC
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = (
        precision_recall_curve(y_test, y_pred_proba)
        if y_pred_proba is not None
        else (None, None)
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    # Classification Report
    clf_report = classification_report(y_test, y_pred)

    # Subplot 1: ROC Curve
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot(
            [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier"
        )
        plt.title(f"{title} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    # Subplot 2: Precision-Recall Curve
    if precision is not None and recall is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="purple", lw=2, label="Precision-Recall Curve")
        plt.title(f"{title} - Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.grid()
        plt.show()

    # Subplot 3: Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Print Classification Report
    print(f"\n{title} - Classification Report\n")
    print(clf_report)
