class ModelesConditions:
    def __init__(self, random_state=42):
        self.random_state = int(random_state)

    def _ajouter_modele(self, modeles, nom, estimator, task_types, accepte_sparse=True, besoin_normalisation=False, pca_ok=True, svd_ok=True, poly_ok=False, has_predict_proba=False, priorite=50, params_light=None, params_full=None):
        modeles[str(nom)] = {"estimator": estimator, "task_types": tuple(task_types), "accepte_sparse": bool(accepte_sparse), "besoin_normalisation": bool(besoin_normalisation), "pca_ok": bool(pca_ok), "svd_ok": bool(svd_ok), "poly_ok": bool(poly_ok), "has_predict_proba": bool(has_predict_proba), "priorite": int(priorite), "params_light": dict(params_light) if isinstance(params_light, dict) else {}, "params_full": dict(params_full) if isinstance(params_full, dict) else {}}

    def get_modeles(self, task_type):
        rs = self.random_state
        t = str(task_type).strip().lower() if task_type is not None else None
        modeles = {}

        # Classification
        if t == "multiclass_onehot":
            from sklearn.linear_model import SGDClassifier, RidgeClassifier
            from sklearn.svm import LinearSVC
            from sklearn.linear_model import PassiveAggressiveClassifier

            self._ajouter_modele(
                modeles, "SGDClassifier",
                SGDClassifier(random_state=rs, max_iter=2000, tol=1e-3),
                task_types=("multiclass_onehot",),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=10,
                params_light={"loss": ["hinge", "log_loss"], "alpha": [1e-5, 1e-4, 1e-3], "penalty": ["l2", "elasticnet"], "class_weight": [None, "balanced"]},
                params_full={"loss": ["hinge", "log_loss"], "alpha": [1e-6, 1e-5, 1e-4, 1e-3], "penalty": ["l2", "l1", "elasticnet"], "l1_ratio": [0.15, 0.5, 0.85], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles, "RidgeClassifier",
                RidgeClassifier(random_state=rs),
                task_types=("multiclass_onehot",),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=15,
                params_light={"alpha": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles, "LinearSVC",
                LinearSVC(random_state=rs, max_iter=5000),
                task_types=("multiclass_onehot",),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=20,
                params_light={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles, "PassiveAggressive",
                PassiveAggressiveClassifier(random_state=rs, max_iter=2000, tol=1e-3),
                task_types=("multiclass_onehot",),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=25,
                params_light={"C": [0.1, 1.0, 10.0], "loss": ["hinge", "squared_hinge"], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "loss": ["hinge", "squared_hinge"], "class_weight": [None, "balanced"]}
            )

            return modeles

        if t in ("binary", "multiclass"):
            from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
            from sklearn.svm import LinearSVC

            self._ajouter_modele(
                modeles, "LogisticRegression",
                LogisticRegression(max_iter=2000, random_state=rs, solver="saga"),
                task_types=("binary", "multiclass"),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=True, priorite=10,
                params_light={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"], "penalty": ["l2"]}
            )

            self._ajouter_modele(
                modeles, "LinearSVC",
                LinearSVC(random_state=rs, max_iter=5000),
                task_types=("binary", "multiclass"),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=15,
                params_light={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles, "SGDClassifier",
                SGDClassifier(random_state=rs, max_iter=2000, tol=1e-3),
                task_types=("binary", "multiclass"),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=20,
                params_light={"loss": ["hinge", "log_loss"], "alpha": [1e-5, 1e-4, 1e-3], "penalty": ["l2", "elasticnet"], "class_weight": [None, "balanced"]},
                params_full={"loss": ["hinge", "log_loss"], "alpha": [1e-6, 1e-5, 1e-4, 1e-3], "penalty": ["l2", "l1", "elasticnet"], "l1_ratio": [0.15, 0.5, 0.85], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles, "RidgeClassifier",
                RidgeClassifier(random_state=rs),
                task_types=("binary", "multiclass"),
                accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True,
                has_predict_proba=False, priorite=25,
                params_light={"alpha": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            # Modèles non linéaires (souvent mieux sans normalisation, et évite sparse)
            try:
                from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
                self._ajouter_modele(modeles, "RandomForest", RandomForestClassifier(random_state=rs), task_types=("binary", "multiclass"), accepte_sparse=False, besoin_normalisation=False, pca_ok=False, svd_ok=False, poly_ok=False, has_predict_proba=True, priorite=60)
                self._ajouter_modele(modeles, "HistGradientBoosting", HistGradientBoostingClassifier(random_state=rs), task_types=("binary", "multiclass"), accepte_sparse=False, besoin_normalisation=False, pca_ok=False, svd_ok=False, poly_ok=False, has_predict_proba=True, priorite=65)
            except Exception:
                pass

            return modeles

        # Régression
        if t in ("regression", "regression_multioutput"):
            from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
            from sklearn.svm import LinearSVR
            from sklearn.multioutput import MultiOutputRegressor

            ridge_est = Ridge()
            sgd_est = SGDRegressor(random_state=rs, max_iter=2000, tol=1e-3)
            enet_est = ElasticNet(random_state=rs, max_iter=2000)
            lsvr_est = LinearSVR()

            if t == "regression_multioutput":
                sgd_est = MultiOutputRegressor(sgd_est)
                enet_est = MultiOutputRegressor(enet_est)
                lsvr_est = MultiOutputRegressor(lsvr_est)

            self._ajouter_modele(modeles, "Ridge", ridge_est, task_types=("regression", "regression_multioutput"), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=False, priorite=10, params_light={"alpha": [0.1, 1.0, 10.0]}, params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})
            self._ajouter_modele(modeles, "SGDRegressor", sgd_est, task_types=("regression", "regression_multioutput"), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=False, priorite=20, params_light={"alpha": [1e-5, 1e-4, 1e-3], "loss": ["squared_error", "huber"], "penalty": ["l2", "elasticnet"]}, params_full={"alpha": [1e-6, 1e-5, 1e-4, 1e-3], "loss": ["squared_error", "huber"], "penalty": ["l2", "l1", "elasticnet"], "l1_ratio": [0.15, 0.5, 0.85]})
            self._ajouter_modele(modeles, "LinearSVR", lsvr_est, task_types=("regression", "regression_multioutput"), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=False, priorite=25, params_light={"C": [0.1, 1.0, 10.0]}, params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0]})
            self._ajouter_modele(modeles, "ElasticNet", enet_est, task_types=("regression", "regression_multioutput"), accepte_sparse=False, besoin_normalisation=True, pca_ok=True, svd_ok=False, poly_ok=True, has_predict_proba=False, priorite=30, params_light={"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}, params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]})

            # Modèles non linéaires (évite sparse)
            try:
                from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
                hgb_est = HistGradientBoostingRegressor(random_state=rs)
                if t == "regression_multioutput":
                    hgb_est = MultiOutputRegressor(hgb_est)
                self._ajouter_modele(modeles, "RandomForestRegressor", RandomForestRegressor(random_state=rs), task_types=("regression", "regression_multioutput"), accepte_sparse=False, besoin_normalisation=False, pca_ok=False, svd_ok=False, poly_ok=False, has_predict_proba=False, priorite=60)
                self._ajouter_modele(modeles, "HistGradientBoostingRegressor", hgb_est, task_types=("regression", "regression_multioutput"), accepte_sparse=False, besoin_normalisation=False, pca_ok=False, svd_ok=False, poly_ok=False, has_predict_proba=False, priorite=65)
            except Exception:
                pass

            return modeles

        # Multilabel
        if t == "multilabel":
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
            from sklearn.svm import LinearSVC
            self._ajouter_modele(modeles, "OVR_LogisticRegression", OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=rs, solver="saga")), task_types=("multilabel",), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=True, priorite=10, params_light={"estimator__C": [0.1, 1.0, 10.0]}, params_full={"estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0]})
            self._ajouter_modele(modeles, "OVR_LinearSVC", OneVsRestClassifier(LinearSVC(random_state=rs, max_iter=5000)), task_types=("multilabel",), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=False, priorite=15, params_light={"estimator__C": [0.1, 1.0, 10.0]}, params_full={"estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0]})
            self._ajouter_modele(modeles, "OVR_SGDClassifier", OneVsRestClassifier(SGDClassifier(random_state=rs, max_iter=2000, tol=1e-3)), task_types=("multilabel",), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=False, priorite=20, params_light={"estimator__loss": ["hinge", "log_loss"], "estimator__alpha": [1e-5, 1e-4, 1e-3]}, params_full={"estimator__loss": ["hinge", "log_loss"], "estimator__alpha": [1e-6, 1e-5, 1e-4, 1e-3], "estimator__penalty": ["l2", "l1", "elasticnet"], "estimator__l1_ratio": [0.15, 0.5, 0.85]})
            self._ajouter_modele(modeles, "OVR_RidgeClassifier", OneVsRestClassifier(RidgeClassifier(random_state=rs)), task_types=("multilabel",), accepte_sparse=True, besoin_normalisation=True, pca_ok=True, svd_ok=True, poly_ok=True, has_predict_proba=False, priorite=25, params_light={"estimator__alpha": [0.1, 1.0, 10.0]}, params_full={"estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]})
            return modeles

        return modeles

    def trier_par_priorite(self, modeles):
        if not isinstance(modeles, dict):
            return []
        return sorted(modeles.items(), key=lambda kv: int(kv[1].get("priorite", 50)))
