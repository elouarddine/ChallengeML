import os
import json
from datetime import datetime
from sklearn.metrics import get_scorer


class Evaluate:
    def __init__(self, preprocessor_obj, pipeline=None, best_name=None, models_obj=None):
        self.pre = preprocessor_obj
        self.models = models_obj
        if self.pre.task_type is None:
            self.pre.detect_task_type()
        self.task_type = self.pre.task_type
        self.pipeline = pipeline if pipeline is not None else (getattr(self.models, "best_pipeline", None) if self.models is not None else None)
        self.best_name = best_name if best_name is not None else (getattr(self.models, "best_name", None) if self.models is not None else None)
        self.main_key = None
        self.multi_scoring = None
        self.test_scores = None

    def _fix_metric(self, scorer_str, value):
        if self.models is not None and hasattr(self.models, "_fix_signed_metric"):
            return float(self.models._fix_signed_metric(scorer_str, value))
        return -float(value) if isinstance(scorer_str, str) and scorer_str.startswith("neg_") else float(value)

    def evaluate_test(self, pipeline=None, rare_threshold=0.30, scoring=None):
        if pipeline is not None:
            self.pipeline = pipeline
        if self.pipeline is None:
            raise ValueError("Pipeline manquant: passe pipeline=... (ex: best_pipeline) ou donne models_obj avec best_pipeline.")

        if self.pre.test_data is None or self.pre.test_labels is None:
            raise ValueError("test_data/test_labels manquants: fais pre.split(...) avec test_size>0.")

        if scoring is None:
            if self.models is None:
                raise ValueError("Donne models_obj (pour get_scoring) ou passe scoring=... manuellement.")
            main_key, multi_scoring = self.models.get_scoring(rare_threshold=rare_threshold)
            if main_key is None or multi_scoring is None:
                raise ValueError(f"Task type non géré: {self.task_type}")
        else:
            main_key, multi_scoring = None, dict(scoring)

        X_test = self.pre.test_data
        y_test = self.pre.get_y_for_fit(self.pre.test_labels)

        scores = {}
        for k, scorer_str in multi_scoring.items():
            sc = get_scorer(scorer_str)
            v = float(sc(self.pipeline, X_test, y_test))
            scores[k] = self._fix_metric(scorer_str, v)

        self.main_key = main_key
        self.multi_scoring = multi_scoring
        self.test_scores = scores
        return {"task_type": self.task_type, "best_model": self.best_name, "main_key": self.main_key, "test_scores": self.test_scores}

    def save_results(self, base_dir="resultats", stage="test"):
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

        meta = {"dataset": dataset_tag, "task_type": self.task_type, "best_model": self.best_name, "main_key": self.main_key, "stage": stage}
        with open(os.path.join(run_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        with open(os.path.join(run_path, "test_scores.json"), "w", encoding="utf-8") as f:
            json.dump({"test_scores": self.test_scores}, f, indent=2, ensure_ascii=False)

        return run_path
