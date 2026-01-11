import os
import csv
import json
from datetime import datetime
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.ensemble import VotingClassifier, VotingRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical
from .ModelesConditions import ModelesConditions
from .ConstructeurPipeline import ConstructeurPipeline


class Models:
    def __init__(self, preprocessor_obj, random_state=42):
        self.pre = preprocessor_obj
        self.random_state = int(random_state)
        if self.pre.task_type is None:
            self.pre.detect_task_type()
        self.task_type = self.pre.task_type
        self.catalogue = ModelesConditions(random_state=self.random_state)
        self.builder = ConstructeurPipeline(preprocessor_obj=self.pre, random_state=self.random_state)
        self.best_name = None
        self.best_pipeline = None
        self.cv_summary = None
        self.val_scores = None
        self.selection_report = None

    def get_scoring(self, rare_threshold=0.30):
        t = self.task_type
        y_ref = self.pre.train_labels if self.pre.train_labels is not None else self.pre.labels
        is_imb, _, _ = self.pre.check_imbalance(y_df=y_ref, rare_threshold=rare_threshold)

        if t in ("binary", "multiclass", "multiclass_onehot"):
            main_key = "f1_macro" if is_imb else "accuracy"
            multi_scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}
            return main_key, multi_scoring

        if t in ("regression", "regression_multioutput"):
            main_key = "mae"
            multi_scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
            return main_key, multi_scoring

        if t == "multilabel":
            main_key = "f1_macro" if is_imb else "f1_micro"
            multi_scoring = {"f1_micro": "f1_micro", "f1_macro": "f1_macro"}
            return main_key, multi_scoring

        return None, None

    # permer d'afficher le details des score de chaque model
    def get_scoring_details(self, rare_threshold=0.30):
        t = self.task_type
        y_ref = self.pre.train_labels if self.pre.train_labels is not None else self.pre.labels
        is_imb, info, stats = self.pre.check_imbalance(y_df=y_ref, rare_threshold=rare_threshold)
        main_key, multi_scoring = self.get_scoring(rare_threshold=rare_threshold)
        return main_key, multi_scoring, bool(is_imb), info, stats

    def _is_error_metric(self, main_key):
        return str(main_key).strip().lower() in ("mae", "rmse")

    def _fix_signed_metric(self, scorer_str, value):
        return -float(value) if isinstance(scorer_str, str) and scorer_str.startswith("neg_") else float(value)

    def _is_better(self, main_key, current, best):
        if best is None:
            return True
        if self._is_error_metric(main_key):
            return float(current) < float(best)
        return float(current) > float(best)

    def _sort_candidates(self, main_key, candidates):
        return sorted(candidates, key=lambda d: float(d["score"]), reverse=(not self._is_error_metric(main_key)))

    def _make_search_space(self, params_dict):
        if not isinstance(params_dict, dict) or len(params_dict) == 0:
            return {}
        space = {}
        for k, v in params_dict.items():
            if isinstance(v, (list, tuple)) and len(v) > 0:
                space[f"model__{k}"] = Categorical(list(v))
        return space

    def optimisation_model(self, model_name, model_info, X, y, method="light", cv=3, scoring=None, n_jobs=-1, n_iter_light=8, n_iter_full=25, onehot_sparse=True, force_output=None, reducer=None, n_components=50, scale_override=None, poly_override=None, poly_degree=2, poly_interaction_only=True, poly_include_bias=False):

        if model_info is None or "estimator" not in model_info:
            return None

        pipe = self.builder.build_pipeline(model_info=model_info, onehot_sparse=bool(onehot_sparse), force_output=force_output, reducer=reducer, n_components=int(n_components), scale_override=scale_override, poly_override=poly_override, poly_degree=int(poly_degree), poly_interaction_only=bool(poly_interaction_only), poly_include_bias=bool(poly_include_bias), X_for_probe=X)
        if pipe is None:
            return None

        params_dict = model_info.get("params_light", {}) if str(method).strip().lower() == "light" else model_info.get("params_full", {})
        search_space = self._make_search_space(params_dict)

        if not search_space:
            try:
                cv_res = cross_validate(pipe, X, y, cv=int(cv), scoring=scoring, n_jobs=int(n_jobs), error_score="raise")
                m = float(cv_res["test_score"].mean())
                m = self._fix_signed_metric(scoring, m)
                pipe.fit(X, y)
                pipe._automl_best_score_ = m
                return pipe
            except Exception:
                return None

        n_iter = int(n_iter_light) if str(method).strip().lower() == "light" else int(n_iter_full)

        opt = BayesSearchCV(estimator=pipe, search_spaces=search_space, n_iter=int(n_iter), cv=int(cv), scoring=scoring, n_jobs=int(n_jobs), random_state=self.random_state, verbose=0, refit=True)
        try:
            opt.fit(X, y)
            best = opt.best_estimator_
            best._automl_best_score_ = self._fix_signed_metric(scoring, float(opt.best_score_))
            return best
        except Exception:
            try:
                cv_res = cross_validate(pipe, X, y, cv=int(cv), scoring=scoring, n_jobs=int(n_jobs), error_score="raise")
                m = float(cv_res["test_score"].mean())
                m = self._fix_signed_metric(scoring, m)
                pipe.fit(X, y)
                pipe._automl_best_score_ = m
                return pipe
            except Exception:
                return None

    def _eval_cv(self, pipe, X, y, cv, multi_scoring, n_jobs):
        cv_res = cross_validate(pipe, X, y, cv=int(cv), scoring=multi_scoring, n_jobs=int(n_jobs), error_score="raise")
        row = {}
        for k, scorer_str in multi_scoring.items():
            m = float(cv_res[f"test_{k}"].mean())
            s = float(cv_res[f"test_{k}"].std())
            m = self._fix_signed_metric(scorer_str, m)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s
        return row

    def _build_voting(self, candidates, models_dict, n_jobs):
        if not isinstance(candidates, list) or len(candidates) < 2:
            return None, None

        estimators_list = [(c["name"], c["pipe"]) for c in candidates if c.get("pipe") is not None]
        if len(estimators_list) < 2:
            return None, None

        t = str(self.task_type).strip().lower()

        if t in ("binary", "multiclass", "multiclass_onehot"):
            can_soft = True
            for name, _ in estimators_list:
                info = models_dict.get(name, {})
                if not bool(info.get("has_predict_proba", False)):
                    can_soft = False
            voting_type = "soft" if can_soft else "hard"
            return VotingClassifier(estimators=estimators_list, voting=voting_type, n_jobs=int(n_jobs)), f"VotingClassifier_{voting_type}"

        if t in ("regression", "regression_multioutput"):
            return VotingRegressor(estimators=estimators_list, n_jobs=int(n_jobs)), "VotingRegressor"

        return None, None

    def select_best_model(self, cv=5, rare_threshold=0.30, n_jobs=-1, bayes_cv=3, n_iter_light=8, n_iter_full=25, top_k=3, onehot_sparse=True, force_output=None, reducer=None, n_components=50, scale_override=None, poly_override=None, poly_degree=2, poly_interaction_only=True, poly_include_bias=False):

        print("AutoML | [Selection] Starting model selection", flush=True)

        if self.pre.train_data is None or self.pre.train_labels is None:
            raise ValueError("Appelle pre.split() avant select_best_model().")

        main_key, multi_scoring, is_imb, imb_info, imb_stats = self.get_scoring_details(rare_threshold=rare_threshold)
        if main_key is None:
            raise ValueError(f"Task type non géré: {self.task_type}")

        main_scorer_str = multi_scoring[main_key]
        X_train = self.pre.train_data
        y_train = self.pre.get_y_for_fit(self.pre.train_labels)

        models_dict = self.catalogue.get_modeles(self.task_type)
        ordered = self.catalogue.trier_par_priorite(models_dict)

        results = []
        candidates_light = []
        tested_light = []
        tested_full = []

        best_light_name, best_light_pipe, best_light_score = None, None, None

        for name, info in ordered:
            print("AutoML | [Light] Optimising model: " + str(name), flush=True)
            pipe_light = self.optimisation_model(model_name=name, model_info=info, X=X_train, y=y_train, method="light", cv=bayes_cv, scoring=main_scorer_str, n_jobs=n_jobs, n_iter_light=n_iter_light, n_iter_full=n_iter_full, onehot_sparse=onehot_sparse, force_output=force_output, reducer=reducer, n_components=n_components, scale_override=scale_override, poly_override=poly_override, poly_degree=poly_degree, poly_interaction_only=poly_interaction_only, poly_include_bias=poly_include_bias)
            if pipe_light is None:
                continue
            try:
                score_main = float(getattr(pipe_light, "_automl_best_score_", None))
                if score_main != score_main:
                    continue

                candidates_light.append({"name": name, "pipe": pipe_light, "score": score_main})
                tested_light.append({"name": name, "score": score_main})

                if self._is_better(main_key, score_main, best_light_score):
                    best_light_score, best_light_name, best_light_pipe = score_main, name, pipe_light

                print("AutoML | [Light] Done: " + str(name) + " => " + str(main_key) + "=" + str(score_main), flush=True)

            except Exception:
                continue

        if not candidates_light:
            raise RuntimeError("Tous les modèles ont échoué en phase Light.")

        candidates_light = self._sort_candidates(main_key, candidates_light)
        top = candidates_light[:int(top_k)]
        top_light = [{"name": c["name"], "score": float(c["score"])} for c in top]  # <--- AJOUT

        final_candidates = []
        best_full_name, best_full_pipe, best_full_score = None, None, None

        for cand in top:
            name = cand["name"]
            info = models_dict.get(name, None)
            if info is None:
                continue

            print("AutoML | [Full] Optimising model: " + str(name), flush=True)
            pipe_full = self.optimisation_model(model_name=name, model_info=info, X=X_train, y=y_train, method="full", cv=bayes_cv, scoring=main_scorer_str, n_jobs=n_jobs, n_iter_light=n_iter_light, n_iter_full=n_iter_full, onehot_sparse=onehot_sparse, force_output=force_output, reducer=reducer, n_components=n_components, scale_override=scale_override, poly_override=poly_override, poly_degree=poly_degree, poly_interaction_only=poly_interaction_only, poly_include_bias=poly_include_bias)
            if pipe_full is None:
                continue
            try:
                row = {"model": f"{name} (Full)"}
                row.update(self._eval_cv(pipe=pipe_full, X=X_train, y=y_train, cv=cv, multi_scoring=multi_scoring, n_jobs=n_jobs))
                results.append(row)
                score_main = float(row[f"{main_key}_mean"])
                print("AutoML | [Full] Done: " + str(name) + " => " + str(main_key) + "=" + str(score_main), flush=True)

                final_candidates.append({"name": name, "pipe": pipe_full, "score": score_main})
                tested_full.append({"name": name, "score": score_main})  # <--- AJOUT
                if self._is_better(main_key, score_main, best_full_score):
                    best_full_score, best_full_name, best_full_pipe = score_main, name, pipe_full
            except Exception:
                continue

        if not final_candidates:
            best_name, best_pipe = best_light_name, best_light_pipe
            best_stage = "Light"
        else:
            print("AutoML | [Full] Evaluating voting (if possible)", flush=True)

            final_candidates = self._sort_candidates(main_key, final_candidates)
            best_name, best_pipe, best_score = final_candidates[0]["name"], final_candidates[0]["pipe"], final_candidates[0]["score"]
            best_stage = "Full"

            vote_full, vote_full_desc = self._build_voting(candidates=final_candidates, models_dict=models_dict, n_jobs=n_jobs)
            if vote_full is not None:
                try:
                    row_v = {"model": f"Ensemble ({vote_full_desc}) (Full)"}
                    row_v.update(self._eval_cv(pipe=vote_full, X=X_train, y=y_train, cv=cv, multi_scoring=multi_scoring, n_jobs=n_jobs))
                    results.append(row_v)
                    v_score = float(row_v[f"{main_key}_mean"])
                    tested_full.append({"name": f"Ensemble({vote_full_desc})", "score": v_score})  # <--- AJOUT
                    if self._is_better(main_key, v_score, best_score):
                        best_name, best_pipe = f"Ensemble_{vote_full_desc}_Full", vote_full
                except Exception:
                    pass

        print("AutoML | [Train] Fitting selected model on full training data: " + str(best_name), flush=True)
        best_pipe.fit(X_train, y_train)

        self.best_name = best_name
        self.best_pipeline = best_pipe
        self.cv_summary = results

        print("AutoML | [Validation] Scoring selected model on validation set", flush=True)

        if self.pre.validation_data is not None and self.pre.validation_labels is not None:
            X_val = self.pre.validation_data
            y_val = self.pre.get_y_for_fit(self.pre.validation_labels)
            val_scores = {}
            for k, scorer_str in multi_scoring.items():
                sc = get_scorer(scorer_str)
                v = float(sc(self.best_pipeline, X_val, y_val))
                v = self._fix_signed_metric(scorer_str, v)
                val_scores[k] = v
            self.val_scores = val_scores

        #report clair pour automl.fit()
        self.selection_report = {
            "task_type": str(self.task_type),
            "is_imbalanced": bool(is_imb),
            "rare_threshold": float(rare_threshold),
            "main_metric": str(main_key),
            "tested_light": tested_light,
            "top_light": top_light,
            "tested_full": tested_full,
            "best_model": str(self.best_name),
            "best_stage": str(best_stage),
            "val_scores": self.val_scores
        }

        return self.best_name, self.best_pipeline, self.cv_summary, self.val_scores

    def get_selection_report(self):
        return self.selection_report

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

        meta = {"dataset": dataset_tag, "task_type": getattr(self.pre, "task_type", None), "best_model": getattr(self, "best_name", None), "stage": stage}

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

        if isinstance(self.selection_report, dict):
            with open(os.path.join(run_path, "selection_report.json"), "w", encoding="utf-8") as f:
                json.dump(self.selection_report, f, indent=2, ensure_ascii=False)

        return run_path

