import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler,LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


class Preprocess:
    def __init__(self, path):
    
        self.path = path
        # Attributs à remplir après load()
        self.data = None
        self.labels = None
        self.types = None
        self.label_encoder = None
        self.name = None
        self.numerical_index = None
        self.categorical_index = None
        self.binary_index = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.task_type = None
    
    
    @staticmethod
    def load_dataset(path):
        if not os.path.isdir(path):
            print("Erreur: le chemin donné n'est pas un dossier ou c'est un chemin qui n'existe pas.")
            return None
        
        folder = os.path.basename(os.path.normpath(path))
        if not folder.startswith("data_"):
            print("Erreur: veuillez donner un dossier de cette forme (data_<Lettre>).")
            return None
        
        if any(os.path.isdir(os.path.join(path, x)) for x in os.listdir(path)):
            print("Veuillez insérer le chemin d'un dossier qui contient directement les fichiers .data, .solution et .type")
            return None
        
        files = os.listdir(path)
        data_files = [f for f in files if f.endswith(".data")]
        sol_files  = [f for f in files if f.endswith(".solution")]
        type_files = [f for f in files if f.endswith(".type")]
        
        if len(data_files) != 1 or len(sol_files) != 1 or len(type_files) != 1:
            print("Erreur: le dossier doit contenir exactement 1 fichier .data, 1 fichier .solution et 1 fichier .type.")
            return None
        
        name = data_files[0].replace(".data", "")
        data_path = os.path.join(path, name + ".data")
        solution_path = os.path.join(path, name + ".solution")
        type_path = os.path.join(path, name + ".type")

        # Charger y et types d'abord pour utiliser n_features dans le cas de sparse
        labels = pd.read_csv(solution_path, sep=r"\s+", header=None)
        types = pd.read_csv(type_path, header=None)[0].tolist()
        n_features = len(types)

        is_sparse = False
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
           for line in f:
               line = line.strip()
               if line:
                  is_sparse = (":" in line)
                  break

        if is_sparse:
            print("dataset sparse détecté (format index:val).")
            X = Preprocess.load_sparse_data_file(data_path, n_features=n_features, n_rows_expected=len(labels))
            return X, labels, types, name

        data = pd.read_csv(data_path, sep=r"\s+", header=None)
        if data.shape[1] != n_features:
             raise ValueError(f"Mismatch features: X={data.shape[1]} mais .type={n_features}")

        return data, labels, types, name
    
    @staticmethod
    def load_sparse_data_file(data_path, n_features, n_rows_expected=None):
        
        data_vals = []
        row_ind = []
        col_ind = []

        n_rows = 0

        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            for r_idx, line in enumerate(f):
                n_rows = r_idx + 1  
                parts = line.strip().split()
                for token in parts:
                    if ":" not in token:
                        continue
                    c_str, v_str = token.split(":", 1)
                    c = int(c_str)
                    v = float(v_str)
                    row_ind.append(r_idx)
                    col_ind.append(c)
                    data_vals.append(v)

        # Cas fichier vide ou sans index:value
        if n_rows == 0:
            return csr_matrix((0, n_features))

        if len(col_ind) == 0:
            X = csr_matrix((n_rows, n_features))
        else:
            # Conversion éventuelle 1-based -> 0-based
            if min(col_ind) == 1:
                col_ind = [c - 1 for c in col_ind]

            # Pour éviter le cas d'un index hors dimension
            if max(col_ind) >= n_features:
                raise ValueError(
                    f"Index colonne hors limite: max_col={max(col_ind)} mais n_features={n_features}. "
                    "Vérifie la base (0/1) et le fichier .type."
                )

            X = csr_matrix((data_vals, (row_ind, col_ind)), shape=(n_rows, n_features))

        # Vérification cohérence avec y
        if n_rows_expected is not None and n_rows != n_rows_expected:
            raise ValueError(f"Mismatch lignes: X={n_rows} vs y={n_rows_expected}")

        return X
    
    
    
    
    # Charger Notre DataSet en faisons appel à la fonction static load_dataset avec un path valid
    # Dans le cas de path invalid des message d'erreur s'affiche 
    # Aprés le chaqrgement du DataSet le csv des fichier .data et .solution et .type est affecter respectivement aux attribut (data, labels, types)
    def load(self):
        result = Preprocess.load_dataset(self.path)
        if result is None:
            return None
        self.data, self.labels, self.types, self.name = result
        return result
    
    
    # Detection du type de la problématique
    def detect_task_type(self):
        if self.labels is None:
            print("Erreur: appelle load() avant detect_task_type().")
            return None
        
        y = self.labels.values
        n_cols = y.shape[1]
        
        # 1 colonne 
        if n_cols == 1:
            y_1d = self.labels.iloc[:, 0].to_numpy()
            target_type = type_of_target(y_1d)
            
            if target_type == "binary":
                self.task_type = "binary"
                return "binary"
            
            if target_type == "continuous":
                self.task_type = "regression"
                return "regression"
            
            if target_type == "multiclass":
                # Ici on gére le cas ou on Corriger le cas de "régression stockée en int"
                uniq = np.unique(y_1d)
                n = len(y_1d)
                if len(uniq) > max(20, int(0.05 * n)):
                    self.task_type = "regression"

                    return "regression"
                self.task_type = "multiclass"
                return "multiclass"
            
            return f"inconnu_{target_type}"
        
        # plusieurs colonnes 
        target_type = type_of_target(y)
        
        if target_type == "multilabel-indicator":
            row_sums = y.sum(axis=1)
            if np.all(row_sums == 1):
                self.task_type = "multiclass_onehot"
                return "multiclass_onehot"
            if n_cols<=4:
                self.task_type = "multiclass_code"
                return "multiclass_code"
            self.task_type = "multilabel"
            return "multilabel"
        
        if target_type == "continuous-multioutput":
            self.task_type = "regression_multioutput"
            return "regression_multioutput"
        
        return f"inconnu_{target_type}"
    
    def encode_target(self, y):
        if self.labels is None:
            print("Erreur: appelle load() avant encode_target().")
            return None
        task = self.detect_task_type()
        if task in ["multiclass_code"]:
            print("Transformation Multilabel -> Multiclass (Label Powerset)...")
            # 1. On convertit le DataFrame/Array en tableau de chaînes (ex: "1010")
            # On s'assure d'abord que c'est du numpy pour éviter les soucis de DataFrame
            y_arr = np.array(y)
            y_str = ["".join(row.astype(str)) for row in y_arr]
            
            # 2. On encode ces chaînes en entiers (0, 1, 2...)
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y_str)
            
            return y_encoded, encoder
            
        # Si ce n'est pas du multiclass_code, on ne touche à rien
        return y, None
    
    
    def get_dataset_info(self):
        if self.data is None or self.labels is None or self.types is None:
            print("Erreur: appelle load() avant get_dataset_info().")
            return None
        
        # indices (au cas où pas encore calculés)
        if self.numerical_index is None or self.categorical_index is None or self.binary_index is None:
            self.numerical_index, self.categorical_index, self.binary_index = self.get_feature_type_indices()
        
        task = self.detect_task_type()
        if issparse(self.data):
            missing_in_data = int(np.isnan(self.data.data).sum())  
        else:
            missing_in_data = int(self.data.isna().sum().sum())
        
        info = {
            "nom_du_dataset": self.name,
            
            ".data_shape": self.data.shape,          # shape du .data
            ".solution_shape": self.labels.shape,    # shape du .solution
            ".type_shape": len(self.types),      # taille du .type (nb de features décrites)
            
            # Détail des types de features
            "nombre_numerical_feature": len(self.numerical_index),
            "nombre_categorical_feature": len(self.categorical_index),
            "nombre_binary_feature": len(self.binary_index),
            
            # Missing values les valeurs Nan
            "missing_in_data": missing_in_data,
            "missing_in_solution": int(self.labels.isna().sum().sum()),
            
            # Type de tâche
            "task_type": task,
        }
        
        # Distribution des classes / labels
        if task in ["binary", "multiclass", "multiclass_code"]:
            info["class_distribution"] = self.labels.iloc[:, 0].value_counts().to_dict()
        
        elif task == "multiclass_onehot":
            y_class = self.labels.values.argmax(axis=1)
            unique, counts = np.unique(y_class, return_counts=True)
            info["class_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}
        
        elif task == "multilabel":
            d = {}
            for j in range(self.labels.shape[1]):
                d[f"label_{j}"] = int(self.labels.iloc[:, j].sum())
            info["label_distribution"] = d
            info["class_distribution"] = None
    
        elif task in ["regression", "regression_multioutput"]:
            info["class_distribution"] = None
        
        else:
            info["class_distribution"] = None
        
        return info

    #retourner un vecteur 1D pour stratify
    #ça aide pour eviter un peu le désiquilibre de classe 
    def get_stratify_vector(self , y_df):
    
        if y_df is None:
            return None

        task = self.detect_task_type()

        if task in ["binary", "multiclass", "multiclass_code"]:
            return y_df.iloc[:, 0].to_numpy()

        if task == "multiclass_onehot":
            return y_df.values.argmax(axis=1)

        #pas de stratify dans le cas de d'autre type de probléme  
        return None

    def check_imbalance(self, rare_threshold=0.10):
         y_strat = self.get_stratify_vector(self.labels)
         if y_strat is None:
             return False, None, []

         unique, counts = np.unique(y_strat, return_counts=True)
         total = counts.sum()

         distribution_percent = {
             k : round((v / int(total)) * 100, 2)
             for k, v in zip(unique, counts)
         }

         rare_classes = [cls for cls, pct in distribution_percent.items() if pct < rare_threshold * 100]
         is_imbalanced = (len(rare_classes) > 0)
         return is_imbalanced, distribution_percent, rare_classes

    
    # On itére sur la liste types de features pour retourner la liste des index pour chaque type (numerical ou categorical ou binary)
    def get_feature_type_indices(self):
        numerical_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "numerical"]
        categorical_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "categorical"]
        binary_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "binary"]
        return numerical_index, categorical_index, binary_index
    
    # Stratégie de nettoyage + encodage (+ normalisation optionnelle)
    def build_preprocessor(self, scale_numeric=True):
        
        # On gére d'abord le cas de saprse Dataset dans ce cas on fait un preprocessing simple
        if self.data is not None and issparse(self.data):
            if scale_numeric:
                return Pipeline(steps=[("scaler", MaxAbsScaler())])
            return "passthrough"
        
        self.numerical_index, self.categorical_index, self.binary_index = self.get_feature_type_indices()
        
        # Si le type de features et numeric on applique la mediane sur les donner Nan
        # StandardScaler() pour centrer et reduire (Optionnel)
        if scale_numeric:
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            # Optionnel car certains model en ont besoin et D'autres non (RandomForest, GradientBoosting)
            # 
        else:
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ])
        
        # Encodage des features categorical en utilisant la methode de OneHotEncoder 
        # Remplacement des Nan par des valeur en utilisant la methode most_frequent
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        
        # Pas besoin d'encodage car c'est deja des chiffre 0 et 1 
        binary_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        
        transformers = []
        if len(self.numerical_index) > 0:
            transformers.append(("num", numerical_pipeline, self.numerical_index))
        if len(self.categorical_index) > 0:
            transformers.append(("cat", categorical_pipeline, self.categorical_index))
        if len(self.binary_index) > 0:
            transformers.append(("bin", binary_pipeline, self.binary_index))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        x=self.data.values
        if x.shape[1]>50:
            task=self.detect_task_type()
            if task in ["binary", "multiclass", "multiclass_code", "multiclass_onehot", "multilabel"]:
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("selector", SelectKBest(score_func=f_classif, k=50))
                ])
            elif task == "regression":
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("selector", SelectKBest(score_func=f_regression, k=50))
                ])
            return full_pipeline
        return preprocessor
    
    
    # Split (train/valid/test) après load, avant fit du preprocessing pour eviter le data leakage
    def split(self, test_size=0.2, validation_size=0.2, random_state=42):
        
        if self.data is None or self.labels is None:
            print("Erreur: appelle load() avant split().")
            return None
        encoded_labels, encoder = self.encode_target(self.labels)
        
        if encoder is not None:
            print(f"Encodage automatique effectué. (Nouveau format: {encoded_labels.shape})")
            self.label_encoder = encoder
            #On reconvertit en DataFrame pour garder la compatibilité avec le reste du code (.iloc)
            self.labels = pd.DataFrame(encoded_labels, columns=["target"])
        
        temp_size = test_size + validation_size
        strat = self.get_stratify_vector(self.labels) 
       
        # Premier Split
        try:
            self.train_data, X_temp, self.train_labels, y_temp = train_test_split(self.data, self.labels, test_size=temp_size, random_state=random_state, stratify=strat)
        except ValueError:
            # fallback sans stratify (au cas où classe trop rare)
            self.train_data, X_temp, self.train_labels, y_temp = train_test_split(self.data, self.labels, test_size=temp_size, random_state=random_state, stratify=None)

        validation_ratio = validation_size / temp_size
        strat2 = self.get_stratify_vector(y_temp) 
        
        # Deuxième Split
        try:
            self.validation_data, self.test_data, self.validation_labels, self.test_labels = train_test_split(X_temp, y_temp, test_size=(1 - validation_ratio), random_state=random_state, stratify=strat2)
        except ValueError:
            # fallback sans stratify (au cas où classe trop rare)
            self.validation_data, self.test_data, self.validation_labels, self.test_labels = train_test_split(X_temp, y_temp, test_size=(1 - validation_ratio), random_state=random_state, stratify=None)

        return self.train_data, self.validation_data, self.test_data, self.train_labels, self.validation_labels, self.test_labels
