from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import issparse


class ConstructeurPipeline:
    def __init__(self, preprocessor_obj, random_state=42):
        self.pre = preprocessor_obj
        self.random_state = int(random_state)

    def _sortie_est_sparse(self, preproc, X_sample=None):
        out = self.pre.analyser_sortie_preprocessor(preproc, X=X_sample, n_lignes=200)
        return (out == "sparse")

    def build_pipeline(self, model_info, onehot_sparse=True, force_output=None, reducer=None, n_components=50, scale_override=None, poly_override=None, poly_degree=2, poly_interaction_only=True, poly_include_bias=False, X_for_probe=None):
        if model_info is None or "estimator" not in model_info:
            return None

        # Décider normalisation
        scale_numeric = bool(model_info.get("besoin_normalisation", False)) if scale_override is None else bool(scale_override)

        # Décider polynomial sur numériques
        poly_numeric = (bool(poly_override) and bool(model_info.get("poly_ok", False))) if poly_override is not None else False

        # Construire le preprocessor via Preprocess
        preproc = self.pre.build_preprocessor(scale_numeric=scale_numeric, cat_encoding="onehot", onehot_sparse=bool(onehot_sparse), force_output=force_output, poly_numeric=bool(poly_numeric), poly_degree=int(poly_degree), poly_interaction_only=bool(poly_interaction_only), poly_include_bias=bool(poly_include_bias))

        # Confirmer sortie sparse/dense
        is_sparse_out = self._sortie_est_sparse(preproc, X_sample=X_for_probe)

        # Si modèle ne supporte pas sparse, on tente une sortie dense
        if is_sparse_out and (not bool(model_info.get("accepte_sparse", True))):
            preproc = self.pre.build_preprocessor(scale_numeric=scale_numeric, cat_encoding="onehot", onehot_sparse=False, force_output="dense", poly_numeric=False, poly_degree=int(poly_degree), poly_interaction_only=bool(poly_interaction_only), poly_include_bias=bool(poly_include_bias))
            is_sparse_out = self._sortie_est_sparse(preproc, X_sample=X_for_probe)
            if is_sparse_out:
                return None

        steps = [("preprocess", preproc)]

        # Réduction dimension
        if reducer is not None:
            reducer = str(reducer).strip().lower()
            if reducer == "auto":
                reducer = "svd" if is_sparse_out else "pca"

            if reducer == "pca":
                if is_sparse_out:
                    return None
                if not bool(model_info.get("pca_ok", True)):
                    return None
                steps.append(("reduce", PCA(n_components=int(n_components), svd_solver="randomized", random_state=self.random_state)))

            elif reducer == "svd":
                if not bool(model_info.get("svd_ok", True)):
                    return None
                steps.append(("reduce", TruncatedSVD(n_components=int(n_components), random_state=self.random_state)))

        # Ajouter le modèle
        steps.append(("model", clone(model_info["estimator"])))
        return Pipeline(steps=steps)
