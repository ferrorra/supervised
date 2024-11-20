def convert_to_categorical(data):
    """
    Convertit les colonnes de type int64 en données catégorielles si elles ont moins de 10 valeurs distinctes.
    
    Paramètres :
        data (pd.DataFrame) : Le DataFrame contenant les données.
    
    Retour :
        pd.DataFrame : Le DataFrame avec les colonnes converties si nécessaire.
    """
    for column in data.select_dtypes(include=['int64']).columns:
        unique_values = data[column].nunique()
        if unique_values <= 10:
            print(f"Conversion de la colonne '{column}' en type catégoriel (valeurs distinctes : {unique_values}).")
            data[column] = data[column].astype('category')
    return data


from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils import resample
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from node2vec import Node2Vec
import networkx as nx

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

def balance_data(X, y, method="SMOTE", sampling_type="oversampling"):
    """
    Gère le déséquilibre des classes en utilisant les méthodes de Imbalanced-learn.
    
    Paramètres :
    - X : pd.DataFrame ou array-like, caractéristiques.
    - y : pd.Series ou array-like, étiquettes de classe.
    - method : str, méthode à utiliser pour équilibrer les données :
        - Pour oversampling : "SMOTE", "ADASYN".
        - Pour undersampling : "RUS" (Random Under Sampling), "TomekLinks", "NearMiss".
        - Pour combinaison : "SMOTETomek", "SMOTEENN".
    - sampling_type : str, "oversampling", "undersampling", ou "combination".
    
    Retour :
    - X_resampled, y_resampled : ensembles rééquilibrés.
    """
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
    from imblearn.combine import SMOTETomek, SMOTEENN

    X_resampled, y_resampled = None, None

    # Convert DataFrame to ensure compatibility with Imbalanced-learn
    if isinstance(X, pd.DataFrame):
        X = X.copy()  # Avoid changing original data

    # Define the sampler based on the method and sampling_type
    if sampling_type == "oversampling":
        if method == "SMOTE":
            sampler = SMOTE()
        elif method == "ADASYN":
            # ADASYN requires numeric data
            if not np.issubdtype(X.values.dtype, np.number):
                raise ValueError("ADASYN requires numeric data. Please encode categorical variables first.")
            sampler = ADASYN()
        else:
            raise ValueError(f"Méthode d'oversampling non reconnue : {method}")
    
    elif sampling_type == "undersampling":
        if method == "RUS":
            sampler = RandomUnderSampler()
        elif method == "TomekLinks":
            sampler = TomekLinks()
        elif method == "NearMiss":
            sampler = NearMiss()
        else:
            raise ValueError(f"Méthode d'undersampling non reconnue : {method}")

    elif sampling_type == "combination":
        if method == "SMOTETomek":
            sampler = SMOTETomek()
        elif method == "SMOTEENN":
            sampler = SMOTEENN()
        else:
            raise ValueError(f"Méthode de combinaison non reconnue : {method}")
    
    else:
        raise ValueError(f"Type de sampling non reconnu : {sampling_type}")

    # Preprocess X based on the sampler requirements
    # For oversampling methods like SMOTE and ADASYN, we need numeric data
    if sampling_type in ["oversampling", "combination"]:
        if not np.issubdtype(X.values.dtype, np.number):
            print("Encodage des variables catégoriques pour oversampling.")
            X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables

    # Standardize numerical data for methods that rely on distances (if not already numeric)
    if sampling_type == "oversampling" and method in ["SMOTE", "ADASYN"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply the selected sampling strategy
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame if X has feature names
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

        print(f"Équilibrage effectué avec {method} ({sampling_type}).")
    except Exception as e:
        print(f"Erreur lors de l'application de la méthode {method} : {e}")
        raise

    return X_resampled, y_resampled


# 1. Encodage des variables catégoriques
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_categorical(data, method="onehot"):
    """
    Encode les variables catégoriques.
    - method : "onehot" ou "label".
    """
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    # If data is a numpy array, return it as-is (no encoding is possible)
    if isinstance(data, np.ndarray):
        print("Data is a numpy array. Skipping categorical encoding.")
        return data

    if method == "onehot":
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown="ignore")  # Handle unknown categories gracefully
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Ensure no NaN values in categorical columns before encoding
        if data[categorical_cols].isna().any().any():
            raise ValueError(f"Categorical columns contain missing values: {data[categorical_cols].isna().sum()}")
        
        encoded = pd.DataFrame(
            encoder.fit_transform(data[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=data.index  # Ensure the index is preserved
        )
        
        # Combine with the non-categorical data
        data = pd.concat([data.drop(categorical_cols, axis=1), encoded], axis=1)
        print("Encodage One-Hot effectué.")
    
    elif method == "label":
        label_encoders = {}
        for col in data.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            
            # Ensure no NaN values in the column before label encoding
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values.")
            
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        print("Encodage Label effectué.")
    
    else:
        raise ValueError("Méthode d'encodage non reconnue : choisissez 'onehot' ou 'label'.")
    
    # Final check: Ensure no NaN values in the resulting dataframe
    if data.isna().any().any():
        raise ValueError("Encoding process introduced NaN values.")
    
    return data


# 2. Vérification de la multicolinéarité
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(data, threshold=5.0):
    """
    Vérifie la multicolinéarité des variables numériques via le VIF.
    Retourne un DataFrame avec les VIF calculés.
    """
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Ensure data is a DataFrame
    if isinstance(data, np.ndarray):
        raise ValueError("Data passed to `check_multicollinearity` must be a pandas DataFrame.")

    numeric_data = data.select_dtypes(include=['float', 'int'])
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_data.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])]
    print("Calcul des VIF terminé.")
    return vif_data[vif_data["VIF"] > threshold]



# 3. Vérification du déséquilibre des classes
def check_class_imbalance(y):
    """
    Vérifie la distribution des classes cibles.
    """
    class_distribution = pd.Series(y).value_counts(normalize=True) * 100
    print("Répartition des classes :")
    print(class_distribution)
    return class_distribution

# 4. Normalisation ou standardisation
def scale_data(data, method="standard"):
    """
    Applique une normalisation ou standardisation aux colonnes numériques.
    - method : "standard" (StandardScaler) ou "minmax".
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    numeric_cols = data.select_dtypes(include=['float', 'int']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    print(f"Données normalisées avec la méthode '{method}'.")
    return data

# 5. Réduction de dimension avec PCA
def reduce_dimension_pca(data, n_components=0.95):
    """
    Applique une PCA pour réduire la dimensionnalité tout en préservant l'inertie.
    - n_components : float pour l'inertie cumulative (0.95 par défaut).
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"Inertie cumulée conservée : {np.sum(pca.explained_variance_ratio_):.2f}")
    return reduced_data, pca.explained_variance_ratio_

# 6. Réduction de dimension avec t-SNE
def reduce_dimension_tsne(data, n_components=2, perplexity=30.0):
    """
    Réduit la dimension avec t-SNE.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    reduced_data = tsne.fit_transform(data)
    print(f"Réduction avec t-SNE effectuée en {n_components} dimensions.")
    return reduced_data

# 7. Traitement des vecteurs de mots avec TF-IDF
def process_text_with_tfidf(text_data):
    """
    Transforme une série de texte en vecteurs TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    print("TF-IDF transformation effectuée.")
    return tfidf_matrix, vectorizer

# 8. Vectorisation des graphes
def vectorize_graph(graph, dimensions=128, walk_length=30, num_walks=200):
    """
    Transforme un graphe en vecteurs en utilisant Node2Vec.
    """
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    vectors = {str(node): model.wv[str(node)] for node in graph.nodes()}
    print("Vectorisation du graphe avec Node2Vec effectuée.")
    return vectors
