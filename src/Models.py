from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
import os
import csv
import json
from datetime import datetime


class Models:
    def __init__(self, preprocessor_obj, random_state=42):
        self.pre = preprocessor_obj
        self.random_state = random_state
        if self.pre.task_type is None:
            self.pre.detect_task_type()
        self.task_type = self.pre.task_type
        self.best_name = None
        self.best_pipeline = None
        self.cv_summary = None
        self.val_scores = None

    def get_models(self):
        rs = self.random_state
        t = self.task_type

        if t in ("binary", "multiclass", "multiclass_onehot"):
            return [
                ("logreg", LogisticRegression(max_iter=2000, random_state=rs)),
                ("rf", RandomForestClassifier(random_state=rs)),
                ("hgb", HistGradientBoostingClassifier(random_state=rs)),
            ]

        if t == "regression":
            return [
                ("ridge", Ridge()),
                ("rf_reg", RandomForestRegressor(random_state=rs)),
                ("hgb_reg", HistGradientBoostingRegressor(random_state=rs)),
            ]

        if t == "multilabel":
            return [
                ("ovr_logreg", OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=rs))),
                ("ovr_rf", OneVsRestClassifier(RandomForestClassifier(random_state=rs))),
                ("ovr_hgb", OneVsRestClassifier(HistGradientBoostingClassifier(random_state=rs))),
            ]

        return []

    def get_scoring(self, rare_threshold=0.10):

        t = self.task_type
        is_imb, _, _ = self.pre.check_imbalance(y_df=self.pre.train_labels, rare_threshold=rare_threshold)

        if t in ("binary", "multiclass", "multiclass_onehot"):
            main_key = "f1_macro" if is_imb else "accuracy"
            multi_scoring = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro",
            }
            return main_key, multi_scoring

        if t in ("regression", "regression_multioutput"):
            main_key = "mae"
            multi_scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            }
            return main_key, multi_scoring

        if t == "multilabel":
            main_key = "f1_macro" if is_imb else "f1_micro"
            multi_scoring = {
                "f1_micro": "f1_micro",
                "f1_macro": "f1_macro",
            }
            return main_key, multi_scoring

        return None, None

    def select_best_model(self, cv=5, scale_numeric=True, rare_threshold=0.10, n_jobs=-1):
        if self.pre.train_data is None or self.pre.train_labels is None:
            raise ValueError("Appelle pre.split() avant select_best_model().")

        main_key, multi_scoring = self.get_scoring(rare_threshold=rare_threshold)
        if main_key is None:
            raise ValueError(f"Task type non géré: {self.task_type}")

        X_train = self.pre.train_data
        y_train = self.pre.get_y_for_fit(self.pre.train_labels)

        models = self.get_models()
        if not models:
            raise ValueError(f"Aucun modèle retourné pour task_type={self.task_type}")

        results = []
        best_name, best_pipe, best_main = None, None, None

        for name, model in models:
            preproc = self.pre.build_preprocessor(scale_numeric=scale_numeric)
            pipe = Pipeline([("preprocess", preproc), ("model", model)])

            cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=multi_scoring, n_jobs=n_jobs, error_score="raise",)

            row = {"model": name}
            for k in multi_scoring.keys():
                m = float(cv_res[f"test_{k}"].mean())
                s = float(cv_res[f"test_{k}"].std())
                scorer_name = multi_scoring[k]
                if isinstance(scorer_name, str) and scorer_name.startswith("neg_"):
                    m, s = -m, s
                row[f"{k}_mean"] = m
                row[f"{k}_std"] = s

            results.append(row)

            current_main = row[f"{main_key}_mean"]
            if (best_main is None) or (current_main > best_main):
                best_main = current_main
                best_name = name
                best_pipe = pipe


        #Ici on fait le refit (entraînement du model) aprés qu'on a fait des fits juste sur des folds (Cross Validation)
        best_pipe.fit(X_train, y_train)

        self.best_name = best_name
        self.best_pipeline = best_pipe
        self.cv_summary = results


        # On fait un dernier check sur validation
        if self.pre.validation_data is not None and self.pre.validation_labels is not None:
            X_val = self.pre.validation_data
            y_val = self.pre.get_y_for_fit(self.pre.validation_labels)

            val_scores = {}
            for k, scorer_str in multi_scoring.items():
                # get_scorer calcul des micro qu'on cherche dans multi_scoring
                sc = get_scorer(scorer_str)
                v = float(sc(self.best_pipeline, X_val, y_val))
                if isinstance(scorer_str, str) and scorer_str.startswith("neg_"):
                    v = -v
                val_scores[k] = v
            self.val_scores = val_scores

        return self.best_name, self.best_pipeline, self.cv_summary, self.val_scores





    # Cette fonction permet d'enregistrer les resultat du process de selection de model
    # ces resultat son stocker sous forme des fichier :
    #
    # meta.json : infos du run (task, modèle choisi, métrique principale…)
    #
    # cv_summary.csv : tableau des scores (mean/std) pour chaque modèle
    #
    # val_scores.json : scores sur validation


    def save_results(self, base_dir="resultats", stage="selection", save_model=False):
        os.makedirs(base_dir, exist_ok=True)

        dataset_tag = None
        if hasattr(self.pre, "path") and self.pre.path is not None:
            dataset_tag = os.path.basename(os.path.normpath(self.pre.path))
        if not dataset_tag:
            dataset_tag = getattr(self.pre, "name", "dataset")

        folder_name = f"{stage}_{dataset_tag}"
        run_path = os.path.join(base_dir, folder_name)

        if os.path.exists(run_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_path = os.path.join(base_dir, f"{folder_name}_{ts}")

        os.makedirs(run_path, exist_ok=True)

        meta = {
            "dataset": dataset_tag,
            "task_type": getattr(self.pre, "task_type", None),
            "best_model": getattr(self, "best_name", None),
            "stage": stage,
        }

        with open(os.path.join(run_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        cv_summary = getattr(self, "cv_summary", None)
        if isinstance(cv_summary, list) and len(cv_summary) > 0:
            keys = []
            for row in cv_summary:
                for k in row.keys():
                    if k not in keys:
                        keys.append(k)

            with open(os.path.join(run_path, "cv_summary.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in cv_summary:
                    w.writerow(row)

        val_scores = getattr(self, "val_scores", None)
        if isinstance(val_scores, dict):
            with open(os.path.join(run_path, "val_scores.json"), "w", encoding="utf-8") as f:
                json.dump(val_scores, f, indent=2, ensure_ascii=False)


        return run_path