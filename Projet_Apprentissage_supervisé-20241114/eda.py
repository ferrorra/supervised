'''
Ce module est spécialement fait pour explorer les données,  les visualiser 
pour comprendre le maximum avant de faire quoi que ce soit

'''

# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder


def summarize_data(data, target_column):
    """
    Résumé du dataset : dimensions, informations générales, statistiques descriptives, et distribution des classes cibles.
    """
    print("Shape of dataset:", data.shape)
    print("\nInfo:")
    print(data.info())
    print("\nDescriptive Statistics:")
    print(data.describe())
    print("\nClass Distribution:")
    if target_column in data.columns:
        print(data.groupby(target_column).size())
    else:
        print(f"Target column '{target_column}' not found in the dataset.")



def study_relationships(data, target_column):
    """
    Étudie les relations entre les variables et la variable cible.
    - Chi2 pour catégorielles.
    - ANOVA pour continues vs catégorielles.
    """
    results = {}
    
    if target_column not in data.columns:
        print(f"Target column '{target_column}' not found in the dataset.")
        return
    
    for col in data.columns:
        if col == target_column:
            continue
        
        if data[col].dtype == 'object' or data[col].nunique() < 15:
            # Catégorielle vs catégorielle : Chi2
            contingency_table = pd.crosstab(data[col], data[target_column])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            results[col] = {"test": "Chi2", "p-value": p}
        else:
            # Continue vs catégorielle : ANOVA
            groups = [data[data[target_column] == val][col] for val in data[target_column].unique()]
            f_stat, p = f_oneway(*groups)
            results[col] = {"test": "ANOVA", "p-value": p}
    
    return pd.DataFrame(results).T


def variable_importance(data, target_column):
    """
    Évalue l'importance des variables par rapport à la cible.
    - Information mutuelle pour catégorielles.
    - Corrélation de Pearson pour continues.
    """
    if target_column not in data.columns:
        print(f"Target column '{target_column}' not found in the dataset.")
        return
    
    importance = {}
    target = data[target_column]
    
    for col in data.columns:
        if col == target_column:
            continue
        
        if data[col].dtype == 'object' or data[col].nunique() < 15:
            # Catégorielle : Information mutuelle
            le = LabelEncoder()
            x_encoded = le.fit_transform(data[col].astype(str))
            y_encoded = le.fit_transform(target.astype(str))
            mi = mutual_info_classif(x_encoded.reshape(-1, 1), y_encoded, discrete_features=True)
            importance[col] = mi[0]
        else:
            # Continue : Corrélation de Pearson
            corr = data[col].corr(target)
            importance[col] = corr
    
    importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=["Importance"])
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    return importance_df

def detect_missing_values(data):
    """
    Identifie et visualise les valeurs manquantes dans le dataset.
    """
    missing = data.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        print("Aucune valeur manquante détectée.")
        return
    
    print("Missing Values:\n", missing)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.index, y=missing.values, palette="viridis")
    plt.title("Nombre de valeurs manquantes par colonne")
    plt.xticks(rotation=45)
    plt.show()

def detect_outliers(data, columns):
    """
    Détecte les valeurs aberrantes dans les colonnes continues en utilisant l'IQR.
    """
    outlier_indices = {}
    
    for col in columns:
        if col not in data.columns:
            print(f"Colonne {col} non trouvée.")
            continue
        if data[col].dtype != 'float' and data[col].dtype != 'int':
            continue
        
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
        outlier_indices[col] = outliers
    
    return outlier_indices


from scipy.stats import shapiro

def univariate_analysis(data, column):
    """
    Analyse univariée : histogramme, boxplot et test de normalité pour une colonne continue.
    """
    if column not in data.columns:
        print(f"Colonne '{column}' non trouvée.")
        return
    
    print(f"Statistiques descriptives pour {column}:\n")
    print(data[column].describe())
    
    # Visualisations
    plt.figure(figsize=(12, 5))
    
    # Histogramme
    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True, color="blue")
    plt.title(f"Histogramme de {column}")
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data[column], color="orange")
    plt.title(f"Boxplot de {column}")
    plt.show()
    
    # Test de normalité
    stat, p = shapiro(data[column].dropna())
    print(f"Test de Shapiro-Wilk pour {column} :")
    print(f"Statistique = {stat:.3f}, p-valeur = {p:.3f}")
    if p > 0.05:
        print(f"La distribution de {column} semble normale (p > 0.05).")
    else:
        print(f"La distribution de {column} n'est pas normale (p <= 0.05).")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bivariate_analysis(data, column1, column2):
    """
    Analyse bivariée : scatter plot pour deux variables continues ou boxplot pour une variable continue vs catégorielle.
    Prend en charge les types : int64, category, object.
    """
    if column1 not in data.columns or column2 not in data.columns:
        print(f"Les colonnes {column1} ou {column2} n'ont pas été trouvées.")
        return

    # Numérique vs Numérique : Scatter plot
    if pd.api.types.is_numeric_dtype(data[column1]) and pd.api.types.is_numeric_dtype(data[column2]):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=column1, y=column2, palette="viridis")
        plt.title(f"Scatter plot : {column1} vs {column2}")
        plt.show()

    # Numérique vs Catégorielle : Boxplot
    elif pd.api.types.is_numeric_dtype(data[column1]) and (
        pd.api.types.is_categorical_dtype(data[column2]) or data[column2].dtype == 'object'):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x=column2, y=column1, palette="viridis")
        plt.title(f"Boxplot : {column1} par {column2}")
        plt.show()

    # Catégorielle vs Numérique : Inverse (Boxplot)
    elif (pd.api.types.is_categorical_dtype(data[column1]) or data[column1].dtype == 'object') and pd.api.types.is_numeric_dtype(data[column2]):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x=column1, y=column2, palette="viridis")
        plt.title(f"Boxplot : {column2} par {column1}")
        plt.show()

    # Catégorielle vs Catégorielle : Heatmap de fréquences
    elif (pd.api.types.is_categorical_dtype(data[column1]) or data[column1].dtype == 'object') and (
        pd.api.types.is_categorical_dtype(data[column2]) or data[column2].dtype == 'object'):
        crosstab = pd.crosstab(data[column1], data[column2])
        plt.figure(figsize=(10, 8))
        sns.heatmap(crosstab, annot=True, fmt="d", cmap="viridis")
        plt.title(f"Heatmap : {column1} vs {column2}")
        plt.show()

    else:
        print("Type de variables non pris en charge pour une analyse bivariée.")


def compare_distributions(data, column, target_column):
    """
    Compare les distributions d'une variable continue entre différentes classes de la cible.
    Prend en charge les types : int64, category, object pour la cible.
    """
    if column not in data.columns or target_column not in data.columns:
        print(f"Les colonnes {column} ou {target_column} n'ont pas été trouvées.")
        return

    if not pd.api.types.is_numeric_dtype(data[column]):
        print(f"La colonne {column} doit être numérique pour comparer les distributions.")
        return

    if not (pd.api.types.is_categorical_dtype(data[target_column]) or data[target_column].dtype == 'object'):
        print(f"La colonne cible {target_column} doit être catégorielle pour comparer les distributions.")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x=column, hue=target_column, fill=True, palette="viridis", alpha=0.6)
    plt.title(f"Distribution de {column} par classes dans {target_column}")
    plt.show()



def analyze_class_imbalance(data, target_column):
    """
    Analyse et visualise le déséquilibre des classes cibles.
    """
    if target_column not in data.columns:
        print(f"Target column '{target_column}' not found in the dataset.")
        return
    
    class_counts = data[target_column].value_counts()
    print("Distribution des classes :\n", class_counts)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title("Distribution des classes cibles")
    plt.show()
