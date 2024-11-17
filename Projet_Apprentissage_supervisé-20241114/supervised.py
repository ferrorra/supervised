
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from preparation import encode_categorical, check_multicollinearity, scale_data, check_class_imbalance
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from preparation import encode_categorical, check_multicollinearity, scale_data, check_class_imbalance
from preparation import balance_data  # Importing the balance_data function

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import dump, load
import os
from preparation import *
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from preparation import encode_categorical, check_multicollinearity, scale_data, balance_data, check_class_imbalance

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)



# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


def logistic_regression_model(X, y, sampling_type=None, method=None):
    """
    Logistic Regression with Grid Search and Sampling.
    Preprocessing: OneHotEncoding, multicollinearity check (VIF).
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="onehot")  # Ensure categorical variables are encoded
    vif_data = check_multicollinearity(X)

    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)

    #print(f'Data has {X.isna().sum().sum()} missing values after preprocessing.')

    # Logistic Regression parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100,1000],
        'penalty': ['l2','l1'],  # Regularization
        'solver': ['lbfgs']
    }
    
    model = LogisticRegression()

    # Perform Grid Search
    print("Running Logistic Regression grid search...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def random_forest_model(X, y, sampling_type=None, method=None):
    """
    Random Forest with Grid Search and Sampling.
    Preprocessing: LabelEncoding for categorical variables.
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="label")
    
    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)
    
    # Random Forest parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    model = RandomForestClassifier()

    # Perform Grid Search
    print("Running Random Forest grid search...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def gradient_boosting_model(X, y, sampling_type=None, method=None):
    """
    Gradient Boosting with Grid Search and Sampling.
    Preprocessing: OneHotEncoding, class imbalance check and adjustment with class_weight.
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="onehot")
    class_distribution = check_class_imbalance(y)
    
    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)
    
    # Gradient Boosting parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    model = GradientBoostingEarlyStopping()

    # Perform Grid Search
    print("Running Gradient Boosting grid search...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def decision_tree_model(X, y, sampling_type=None, method=None):
    """
    Decision Tree with Grid Search and Sampling.
    Preprocessing: LabelEncoding for categorical variables.
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="label")
    
    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)
    
    # Decision Tree parameter grid
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    model = DecisionTreeClassifier()

    # Perform Grid Search
    print("Running Decision Tree grid search...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def knn_model(X, y, sampling_type=None, method=None):
    """
    K-Nearest Neighbors with Grid Search and Sampling.
    Preprocessing: OneHotEncoding, StandardScaler for normalization.
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="onehot")
    X = scale_data(X, method="standard")
    
    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)
    
    # KNN parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    model = KNeighborsClassifier()

    # Perform Grid Search
    print("Running KNN grid search...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def svm_model(X, y, sampling_type=None, method=None):
    """
    Support Vector Machine (Linear) with Grid Search and Sampling.
    Preprocessing: OneHotEncoding, StandardScaler for normalization.
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="onehot")
    X = scale_data(X, method="standard")
    
    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)
    
    # SVM parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear']
    }
    
    model = SVC()

    # Perform Grid Search
    print("Running SVM grid search...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def naive_bayes_model(X, y, sampling_type=None, method=None):
    """
    Naive Bayes with Sampling.
    Preprocessing: OneHotEncoding.
    """
    # Apply necessary preprocessing
    X = encode_categorical(X, method="onehot")
    
    # Handle class imbalance if needed
    if sampling_type and method:
        X, y = balance_data(X, y, method=method, sampling_type=sampling_type)
    
    # No hyperparameters for tuning
    model = GaussianNB()

    print("Fitting Naive Bayes...")
    model.fit(X, y)

    return model, None  # No grid search for Naive Bayes
