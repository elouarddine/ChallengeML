from sklearn.metrics import get_scorer, confusion_matrix, classification_report


class Evaluate:
    def __init__(self, preprocessor_obj, models_obj, final_pipeline):
        self.pre = preprocessor_obj
        self.models = models_obj
        self.pipe = final_pipeline

        self.test_scores = None
        self.test_confusion = None
        self.test_report = None

    def evaluate_on_test(self, rare_threshold=0.10):
        if self.pre.test_data is None or self.pre.test_labels is None:
            raise ValueError("test_data/test_labels est None. Appelle split() avant.")

        X_test = self.pre.test_data
        y_test = self.pre.get_y_for_fit(self.pre.test_labels)

        _, multi_scoring = self.models.get_scoring(rare_threshold=rare_threshold)
        if multi_scoring is None:
            raise ValueError("Aucun scoring disponible pour ce task_type.")

        scores = {}
        for k, scorer_str in multi_scoring.items():
            sc = get_scorer(scorer_str)
            v = float(sc(self.pipe, X_test, y_test))
            if isinstance(scorer_str, str) and scorer_str.startswith("neg_"):
                v = -v
            scores[k] = v

        self.test_scores = scores

        t = self.pre.task_type
        if t in ("binary", "multiclass", "multiclass_onehot"):
            y_pred = self.pipe.predict(X_test)
            self.test_confusion = confusion_matrix(y_test, y_pred)
            self.test_report = classification_report(y_test, y_pred, zero_division=0)

        return self.test_scores, self.test_confusion, self.test_report
