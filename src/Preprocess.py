import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from scipy.sparse import csr_matrix, issparse

from sklearn.preprocessing import PolynomialFeatures

class Preprocess:
    def __init__(self, path):

        self.path = path
        # Attributs à remplir après load()
        self.data = None
        self.labels = None
        self.types = None
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

        # infos utiles pour Models (sparse/dense, encodage cat, etc.)
        self.preprocessor_output_ = None
        self.preprocessor_config_ = None

    @staticmethod
    def _resolve_dataset_files(path):
        # Support: soit dossier data_X/, soit chemin direct vers un fichier .data
        if os.path.isdir(path):
            folder = os.path.basename(os.path.normpath(path))
            if not folder.startswith("data_"):
                print("Erreur: veuillez donner un dossier de cette forme (data_<Lettre>).")
                return None
            if any(os.path.isdir(os.path.join(path, x)) for x in os.listdir(path)):
                print("Veuillez insérer le chemin d'un dossier qui contient directement les fichiers .data, .solution et .type")
                return None
            files = os.listdir(path)
            data_files = [f for f in files if f.endswith(".data")]
            sol_files = [f for f in files if f.endswith(".solution")]
            type_files = [f for f in files if f.endswith(".type")]
            if len(data_files) != 1 or len(type_files) != 1:
                print("Erreur: le dossier doit contenir exactement 1 fichier .data et 1 fichier .type (le .solution peut manquer pour un dataset test).")
                return None
            name = data_files[0].replace(".data", "")
            data_path = os.path.join(path, name + ".data")
            solution_path = os.path.join(path, name + ".solution")
            type_path = os.path.join(path, name + ".type")
            return data_path, solution_path, type_path, name

        if os.path.isfile(path) and path.endswith(".data"):
            base = path[:-5]
            data_path = path
            solution_path = base + ".solution"
            type_path = base + ".type"
            name = os.path.basename(base)
            return data_path, solution_path, type_path, name

        print("Erreur: le chemin donné n'est ni un dossier valide, ni un fichier .data.")
        return None

    @staticmethod
    def load_dataset(path):
        resolved = Preprocess._resolve_dataset_files(path)
        if resolved is None:
            return None
        data_path, solution_path, type_path, name = resolved

        if not os.path.isfile(type_path):
            print(f"Erreur: fichier .type introuvable: {type_path}")
            return None

        # Charger types d'abord pour utiliser n_features dans le cas de sparse
        types = pd.read_csv(type_path, header=None)[0].tolist()
        n_features = len(types)

        labels = None
        if os.path.isfile(solution_path):
            labels = pd.read_csv(solution_path, sep=r"\s+", header=None)

        is_sparse = False
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    is_sparse = (":" in line)
                    break

        if is_sparse:
            print("dataset sparse détecté (format index:val).")
            X = Preprocess.load_sparse_data_file(data_path, n_features=n_features, n_rows_expected=(len(labels) if labels is not None else None))
            return X, labels, types, name

        data = pd.read_csv(data_path, sep=r"\s+", header=None)
        if data.shape[1] != n_features:
            raise ValueError(f"Mismatch features: X={data.shape[1]} mais .type={n_features}")
        if labels is not None and len(labels) != len(data):
            raise ValueError(f"Mismatch lignes: X={len(data)} vs y={len(labels)}")

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
                    row_ind.append(r_idx)
                    col_ind.append(int(c_str))
                    data_vals.append(float(v_str))

        # Cas fichier vide ou sans index:value
        if n_rows == 0:
            return csr_matrix((0, n_features))

        if len(col_ind) == 0:
            X = csr_matrix((n_rows, n_features))
        else:
            has_zero = (0 in col_ind)
            if (not has_zero) and (max(col_ind) >= n_features or min(col_ind) == 1):
                col_ind = [c - 1 for c in col_ind]

            if max(col_ind) >= n_features or min(col_ind) < 0:
                raise ValueError(f"Index colonne hors limite: min_col={min(col_ind)} max_col={max(col_ind)} mais n_features={n_features}. Vérifie la base (0/1) et le fichier .type.")

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
        if self.labels is not None:
            self.detect_task_type()
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
                return self.task_type

            if target_type == "continuous":
                self.task_type = "regression"
                return self.task_type

            if target_type == "multiclass":
                # Ici on gére le cas ou on Corriger le cas de "régression stockée en int"
                uniq = np.unique(y_1d[~pd.isna(y_1d)])
                n = len(y_1d)
                if len(uniq) > max(20, int(0.05 * n)):
                    self.task_type = "regression"
                    return self.task_type
                self.task_type = "multiclass"
                return self.task_type

            self.task_type = f"inconnu_{target_type}"
            return self.task_type

        # plusieurs colonnes
        target_type = type_of_target(y)

        if target_type == "multilabel-indicator":
            row_sums = np.nansum(y, axis=1)
            if np.all(row_sums == 1):
                self.task_type = "multiclass_onehot"
                return self.task_type
            self.task_type = "multilabel"
            return self.task_type

        if target_type == "continuous-multioutput":
            self.task_type = "regression_multioutput"
            return self.task_type

        self.task_type = f"inconnu_{target_type}"
        return self.task_type

    def get_y_for_fit(self, y_df):
        if y_df is None:
            return None

        if self.task_type is None:
            self.detect_task_type()

        if self.task_type in ("binary", "multiclass", "regression"):
            return y_df.iloc[:, 0].to_numpy()

        if self.task_type == "multiclass_onehot":
            return y_df.values.argmax(axis=1)

        if self.task_type in ("multilabel", "regression_multioutput"):
            return y_df.values

        return y_df.values

    # retourner un vecteur 1D pour stratify
    # ça aide pour eviter un peu le désiquilibre de classe
    def get_stratify_vector(self, y_df):

        if y_df is None:
            return None

        if self.task_type is None:
            self.detect_task_type()

        if self.task_type in ["binary", "multiclass"]:
            return y_df.iloc[:, 0].to_numpy()

        if self.task_type == "multiclass_onehot":
            return y_df.values.argmax(axis=1)

        # pas de stratify dans le cas de d'autre type de probléme
        return None

    def check_imbalance(self, y_df=None, rare_threshold=0.30):
        if y_df is None:
            y_df = self.labels
        if y_df is None:
            return False, None, []

        if self.task_type is None:
            self.detect_task_type()

        try:
            rare_threshold = float(rare_threshold)
        except Exception:
            rare_threshold = 0.30
        rare_threshold = min(max(rare_threshold, 0.0), 0.49)

        if self.task_type == "multilabel":
            Y = np.array(y_df.values)
            if Y.size == 0:
                return False, {}, []
            Y = np.nan_to_num(Y, nan=0.0)
            pos_rate = Y.mean(axis=0) * 100.0
            distribution_percent = {f"label_{j}": round(float(pos_rate[j]), 2) for j in range(Y.shape[1])}
            low = rare_threshold * 100.0
            high = (1.0 - rare_threshold) * 100.0
            rare_labels = [f"label_{j}" for j in range(Y.shape[1]) if pos_rate[j] < low or pos_rate[j] > high]
            is_imbalanced = (len(rare_labels) > 0)
            return is_imbalanced, distribution_percent, rare_labels

        y_strat = self.get_stratify_vector(y_df)
        if y_strat is None:
            return False, None, []

        y_strat = np.array(y_strat)
        y_strat = y_strat[~pd.isna(y_strat)]
        if y_strat.size == 0:
            return False, {}, []

        unique, counts = np.unique(y_strat, return_counts=True)
        total = float(counts.sum())
        if total <= 0:
            return False, {}, []

        distribution_percent = {str(k): round((float(v) / total) * 100.0, 2) for k, v in zip(unique, counts)}
        low = rare_threshold * 100.0
        high = (1.0 - rare_threshold) * 100.0
        rare_classes = [str(cls) for cls, pct in distribution_percent.items() if pct < low]
        dominant_classes = [str(cls) for cls, pct in distribution_percent.items() if pct > high]
        is_imbalanced = (len(rare_classes) > 0 or len(dominant_classes) > 0)
        flagged = sorted(list(set(rare_classes + dominant_classes)))
        return is_imbalanced, distribution_percent, flagged

    # On itére sur la liste types de features pour retourner la liste des index pour chaque type (numerical ou categorical ou binary)
    def get_feature_type_indices(self):
        if self.types is None:
            return [], [], []
        numerical_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "numerical"]
        categorical_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "categorical"]
        binary_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "binary"]
        return numerical_index, categorical_index, binary_index

    # Stratégie de nettoyage + encodage (+ normalisation optionnelle)
    def build_preprocessor(self, scale_numeric=True, cat_encoding="onehot", onehot_sparse=True, force_output=None,
                           poly_numeric=False, poly_degree=2, poly_interaction_only=True, poly_include_bias=False):

        # On gére d'abord le cas de saprse Dataset dans ce cas on fait un preprocessing simple
        if self.data is not None and issparse(self.data):
            self.preprocessor_output_ = "sparse"
            self.preprocessor_config_ = {"mode": "sparse_dataset", "scale_numeric": bool(scale_numeric),
                                         "cat_encoding": "none", "poly_numeric": False}
            if scale_numeric:
                return Pipeline(steps=[("scaler", MaxAbsScaler())])
            return "passthrough"

        self.numerical_index, self.categorical_index, self.binary_index = self.get_feature_type_indices()

        # Si le type de features et numeric on applique la mediane sur les donner Nan
        # StandardScaler() pour centrer et reduire (Optionnel)
        if poly_numeric:
            if scale_numeric:
                numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                                     ("poly",PolynomialFeatures(degree=int(poly_degree),interaction_only=bool(poly_interaction_only),include_bias=bool(poly_include_bias))),
                                                     ("scaler", StandardScaler())])
            else:
                numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                                     ("poly",PolynomialFeatures(degree=int(poly_degree),interaction_only=bool(poly_interaction_only),include_bias=bool(poly_include_bias)))])
        else:
            if scale_numeric:
                numerical_pipeline = Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())])
                # Optionnel car certains model en ont besoin et D'autres non (RandomForest, GradientBoosting)
                #
            else:
                numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

        # Encodage des features categorical en utilisant la methode de OneHotEncoder
        # Remplacement des Nan par des valeur en utilisant la methode most_frequent
        categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                        sparse_output=bool(onehot_sparse)))])

        # Pas besoin d'encodage car c'est deja des chiffre 0 et 1
        binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

        transformers = []
        if len(self.numerical_index) > 0:
            transformers.append(("num", numerical_pipeline, self.numerical_index))
        if len(self.categorical_index) > 0:
            transformers.append(("cat", categorical_pipeline, self.categorical_index))
        if len(self.binary_index) > 0:
            transformers.append(("bin", binary_pipeline, self.binary_index))

        # Gestion sortie sparse/dense du ColumnTransformer
        if force_output is None:
            sparse_threshold = 0.3
        else:
            force_output = str(force_output).strip().lower()
            sparse_threshold = 1.0 if force_output == "sparse" else 0.0 if force_output == "dense" else 0.3

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=sparse_threshold)

        # infos “prévisionnelles” (la sortie réelle sera confirmée par analyser_sortie_preprocessor)
        self.preprocessor_output_ = "dense" if force_output == "dense" else "sparse" if (
                    bool(onehot_sparse) or force_output == "sparse") else "dense"

        self.preprocessor_config_ = {"scale_numeric": bool(scale_numeric), "cat_encoding": "onehot",
                                     "onehot_sparse": bool(onehot_sparse), "force_output": force_output,
                                     "sparse_threshold": sparse_threshold, "poly_numeric": bool(poly_numeric),
                                     "poly_degree": int(poly_degree) if poly_numeric else None,
                                     "poly_interaction_only": bool(poly_interaction_only) if poly_numeric else None,
                                     "poly_include_bias": bool(poly_include_bias) if poly_numeric else None}
        return preprocessor

    # Analyse rapide de la sortie réelle (sparse/dense) après application du preprocessor sur un échantillon
    def analyser_sortie_preprocessor(self, preprocessor, X=None, n_lignes=200):
        if preprocessor is None:
            return None
        if X is None:
            X = self.train_data if self.train_data is not None else self.data
        if X is None:
            return None
        try:
            if (not issparse(X)) and hasattr(X, "__len__") and len(X) > int(n_lignes):
                X_fit = X.iloc[:int(n_lignes)] if isinstance(X, pd.DataFrame) else X[:int(n_lignes)]
            else:
                X_fit = X
            Z = preprocessor.fit_transform(X_fit)
            self.preprocessor_output_ = "sparse" if issparse(Z) else "dense"
            if self.preprocessor_config_ is None:
                self.preprocessor_config_ = {}
            self.preprocessor_config_["sortie_reelle"] = self.preprocessor_output_
            self.preprocessor_config_["n_features_sortie"] = int(Z.shape[1]) if hasattr(Z, "shape") else None
            return self.preprocessor_output_
        except Exception:
            return self.preprocessor_output_

    # Split (train/valid/test) après load, avant fit du preprocessing pour eviter le data leakage
    def split(self, test_size=0.2, validation_size=0.2, random_state=42):

        if self.data is None or self.labels is None:
            print("Erreur: appelle load() avant split().")
            return None

        test_size = float(test_size)
        validation_size = float(validation_size)
        if test_size < 0 or validation_size < 0 or (test_size + validation_size) >= 1.0:
            raise ValueError("Erreur: test_size et validation_size doivent être >=0 et test_size+validation_size < 1.")

        temp_size = test_size + validation_size
        strat = self.get_stratify_vector(self.labels)

        # Premier Split
        if temp_size == 0.0:
            self.train_data, self.train_labels = self.data, self.labels
            self.validation_data, self.validation_labels = None, None
            self.test_data, self.test_labels = None, None
            return self.train_data, self.validation_data, self.test_data, self.train_labels, self.validation_labels, self.test_labels

        try:
            self.train_data, X_temp, self.train_labels, y_temp = train_test_split(self.data, self.labels, test_size=temp_size, random_state=random_state, stratify=strat)
        except ValueError:
            # fallback sans stratify (au cas où classe trop rare)
            self.train_data, X_temp, self.train_labels, y_temp = train_test_split(self.data, self.labels, test_size=temp_size, random_state=random_state, stratify=None)

        if validation_size == 0.0:
            self.validation_data, self.validation_labels = None, None
            self.test_data, self.test_labels = X_temp, y_temp
            return self.train_data, self.validation_data, self.test_data, self.train_labels, self.validation_labels, self.test_labels

        validation_ratio = validation_size / temp_size
        strat2 = self.get_stratify_vector(y_temp)

        # Deuxième Split
        try:
            self.validation_data, self.test_data, self.validation_labels, self.test_labels = train_test_split(X_temp, y_temp, test_size=(1 - validation_ratio), random_state=random_state, stratify=strat2)
        except ValueError:
            # fallback sans stratify (au cas où classe trop rare)
            self.validation_data, self.test_data, self.validation_labels, self.test_labels = train_test_split(X_temp, y_temp, test_size=(1 - validation_ratio), random_state=random_state, stratify=None)

        return self.train_data, self.validation_data, self.test_data, self.train_labels, self.validation_labels, self.test_labels

    # Une fonction qui represente des infos du dataset
    def get_dataset_info(self):
        if self.data is None or self.types is None:
            print("Erreur: appelle load() avant get_dataset_info().")
            return None

        if self.labels is not None and self.task_type is None:
            self.detect_task_type()

        is_sparse_dataset = bool(issparse(self.data))
        shape_data = tuple(self.data.shape) if hasattr(self.data, "shape") else None
        task = self.task_type if self.labels is not None else None
        has_rare = None
        if self.labels is not None and task is not None and task in ["binary", "multiclass", "multiclass_onehot", "multilabel"]:
            has_rare, _, _ = self.check_imbalance(self.labels, rare_threshold=0.30)

        info = {
            "nom_du_dataset": self.name,
            "task_type": task,
            "data_shape": shape_data,
            "dataset_is_sparse": is_sparse_dataset,
            "preprocessor_output": self.preprocessor_output_,
            "rareté_detectée": bool(has_rare) if has_rare is not None else None,
        }
        return info
