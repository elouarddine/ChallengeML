from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import get_scorer
from scipy.stats import loguniform, randint, uniform


class Optimise:
    def __init__(self, preprocessor_obj, models_obj, random_state=42):
        self.pre = preprocessor_obj
        self.models = models_obj
        self.random_state = random_state

        self.tuned_pipeline = None
        self.tuned_best_params = None
        self.tuned_val_scores = None
        self.tuned_cv_results = None


    # Définir
    def get_param_distributions(self, model_name: str):
        if model_name == "logreg":
            return [
                {
                    "model__solver": ["saga"],
                    "model__penalty": ["l2"],
                    "model__C": loguniform(1e-4, 1e3),
                    "model__class_weight": [None, "balanced"],
                    "model__max_iter": [2000],
                },
                {
                    "model__solver": ["saga"],
                    "model__penalty": ["l1"],
                    "model__C": loguniform(1e-4, 1e3),
                    "model__class_weight": [None, "balanced"],
                    "model__max_iter": [2000],
                },
                {
                    "model__solver": ["saga"],
                    "model__penalty": ["elasticnet"],
                    "model__C": loguniform(1e-4, 1e3),
                    "model__l1_ratio": uniform(0.0, 1.0),
                    "model__class_weight": [None, "balanced"],
                    "model__max_iter": [2000],
                },
            ]

        if model_name == "rf":
            return {
                "model__n_estimators": randint(300, 2001),
                "model__max_depth": [None] + list(range(5, 61, 5)),
                "model__min_samples_split": randint(2, 31),
                "model__min_samples_leaf": randint(1, 16),
                "model__max_features": ["sqrt", "log2", None],
                "model__bootstrap": [True, False],
                "model__class_weight": [None, "balanced"],
            }

        if model_name == "hgb":
            return {
                "model__learning_rate": loguniform(1e-3, 5e-1),
                "model__max_depth": [None, 3, 5, 7, 9],
                "model__max_leaf_nodes": randint(15, 256),
                "model__min_samples_leaf": randint(10, 201),
                "model__l2_regularization": loguniform(1e-4, 10.0),
                "model__max_iter": randint(100, 801),
            }

        if model_name == "ridge":
            return {
                "model__alpha": loguniform(1e-5, 1e5),
            }

        if model_name == "rf_reg":
            return {
                "model__n_estimators": randint(300, 2001),
                "model__max_depth": [None] + list(range(5, 61, 5)),
                "model__min_samples_split": randint(2, 31),
                "model__min_samples_leaf": randint(1, 16),
                "model__max_features": ["sqrt", "log2", None],
                "model__bootstrap": [True, False],
            }

        if model_name == "hgb_reg":
            return {
                "model__learning_rate": loguniform(1e-3, 5e-1),
                "model__max_depth": [None, 3, 5, 7, 9],
                "model__max_leaf_nodes": randint(15, 256),
                "model__min_samples_leaf": randint(10, 201),
                "model__l2_regularization": loguniform(1e-4, 10.0),
                "model__max_iter": randint(100, 801),
            }

        if model_name == "ovr_logreg":
            return [
                {
                    "model__estimator__solver": ["saga"],
                    "model__estimator__penalty": ["l2"],
                    "model__estimator__C": loguniform(1e-4, 1e3),
                    "model__estimator__class_weight": [None, "balanced"],
                    "model__estimator__max_iter": [2000],
                },
                {
                    "model__estimator__solver": ["saga"],
                    "model__estimator__penalty": ["l1"],
                    "model__estimator__C": loguniform(1e-4, 1e3),
                    "model__estimator__class_weight": [None, "balanced"],
                    "model__estimator__max_iter": [2000],
                },
                {
                    "model__estimator__solver": ["saga"],
                    "model__estimator__penalty": ["elasticnet"],
                    "model__estimator__C": loguniform(1e-4, 1e3),
                    "model__estimator__l1_ratio": uniform(0.0, 1.0),
                    "model__estimator__class_weight": [None, "balanced"],
                    "model__estimator__max_iter": [2000],
                },
            ]

        if model_name == "ovr_rf":
            return {
                "model__estimator__n_estimators": randint(300, 2001),
                "model__estimator__max_depth": [None] + list(range(5, 61, 5)),
                "model__estimator__min_samples_split": randint(2, 31),
                "model__estimator__min_samples_leaf": randint(1, 16),
                "model__estimator__max_features": ["sqrt", "log2", None],
                "model__estimator__bootstrap": [True, False],
                "model__estimator__class_weight": [None, "balanced"],
            }

        if model_name == "ovr_hgb":
            return {
                "model__estimator__learning_rate": loguniform(1e-3, 5e-1),
                "model__estimator__max_depth": [None, 3, 5, 7, 9],
                "model__estimator__max_leaf_nodes": randint(15, 256),
                "model__estimator__min_samples_leaf": randint(10, 201),
                "model__estimator__l2_regularization": loguniform(1e-4, 10.0),
                "model__estimator__max_iter": randint(100, 801),
            }

        return {}

    def tune_selected_model(self, selected_model_name: str, n_iter=60, cv=5, rare_threshold=0.10, n_jobs=-1):
        if self.pre.task_type is None:
            self.pre.detect_task_type()

        X_train = self.pre.train_data
        y_train = self.pre.get_y_for_fit(self.pre.train_labels)

        main_key, multi_scoring = self.models.get_scoring(rare_threshold=rare_threshold)

        candidates = dict(self.models.get_models())
        if selected_model_name not in candidates:
            raise ValueError(f"Modèle inconnu: {selected_model_name}")

        estimator = clone(candidates[selected_model_name])

        pipe = Pipeline([
            ("preprocess", self.pre.build_preprocessor()),
            ("model", estimator),
        ])

        param_distributions = self.get_param_distributions(selected_model_name)
        if not param_distributions:
            raise ValueError(f"Aucune distribution définie pour: {selected_model_name}")

        search = RandomizedSearchCV(estimator=pipe, param_distributions=param_distributions, n_iter=n_iter, scoring=multi_scoring, refit=main_key, cv=cv, random_state=self.random_state, n_jobs=n_jobs, verbose=0,)

        search.fit(X_train, y_train)

        tuned_pipe = search.best_estimator_
        best_params = search.best_params_
        cv_results = search.cv_results_

        val_scores = None
        if self.pre.validation_data is not None and self.pre.validation_labels is not None:
            X_val = self.pre.validation_data
            y_val = self.pre.get_y_for_fit(self.pre.validation_labels)

            val_scores = {}
            for k, scorer_str in multi_scoring.items():
                sc = get_scorer(scorer_str)
                v = float(sc(tuned_pipe, X_val, y_val))
                if isinstance(scorer_str, str) and scorer_str.startswith("neg_"):
                    v = -v
                val_scores[k] = v

        self.tuned_pipeline = tuned_pipe
        self.tuned_best_params = best_params
        self.tuned_val_scores = val_scores
        self.tuned_cv_results = cv_results

        return tuned_pipe, best_params, cv_results, val_scores