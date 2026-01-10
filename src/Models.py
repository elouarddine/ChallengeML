
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.base import clone
from skopt import BayesSearchCV

import warnings

from src import models_prams as mp
import os
import csv
import json
from datetime import datetime
warnings.filterwarnings("ignore", module="skopt")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


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
    
        """ if t in ("binary", "multiclass","multiclass_code", "multiclass_onehot"):
            #import des modèles de classification
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
            from catboost import CatBoostClassifier



            return [
                ("logreg", LogisticRegression(max_iter=2000, random_state=rs)),
                ("rf", RandomForestClassifier(random_state=rs)),
                ("hgb", HistGradientBoostingClassifier(random_state=rs)),
                ("xgb", XGBClassifier(random_state=rs)),
                ("lgbm", LGBMClassifier(random_state=rs, verbose=-1)),
                ("cat", CatBoostClassifier(random_state=rs, verbose=0))
            ]

        if t == "regression":
            #import des modèles de régression
            from sklearn.linear_model import  ElasticNet
            from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            from catboost import CatBoostRegressor

            return [
                
                
                ("elastic", ElasticNet(random_state=rs)),  
                ("rf_reg", RandomForestRegressor(random_state=rs)),
                ("hgb_reg", HistGradientBoostingRegressor(random_state=rs)),
                ("xgb_reg", XGBRegressor(random_state=rs)),
                ("lgbm_reg", LGBMRegressor(random_state=rs, verbose=-1)),
                ("cat_reg", CatBoostRegressor(random_state=rs, verbose=0))
            ] """
        
        if t == "multilabel":
            #import des modèles de multilabel
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
            from sklearn.multiclass import OneVsRestClassifier
            from lightgbm import LGBMClassifier

            return {
                "ovr_logreg": {
                    "estimator": OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=rs)),
                    "params_light": {}, # Vide = pas d'optimisation (utilise les défauts)
                    "params_full": {},
                    "has_predict_proba": True 
                },
                "ovr_rf": {
                    "estimator": OneVsRestClassifier(RandomForestClassifier(random_state=rs)),
                    "params_light": {},
                    "params_full": {},
                    "has_predict_proba": True
                },
                "ovr_hgb": {
                    "estimator": OneVsRestClassifier(HistGradientBoostingClassifier(random_state=rs)),
                    "params_light": {},
                    "params_full": {},
                    "has_predict_proba": True 
                },
                "rf_native": {
                    "estimator": RandomForestClassifier(random_state=rs), # Nativement multilabel
                    "params_light": {},
                    "params_full": {},
                    "has_predict_proba": True
                },
                "ovr_lgbm": {
                    "estimator": OneVsRestClassifier(LGBMClassifier(random_state=rs, verbose=-1)),
                    "params_light": {},
                    "params_full": {},
                    "has_predict_proba": True
                }
            }
        
        # or plus plus facilement on fait 
        return mp.get_models(self.task_type)
        

        

    def get_scoring(self, rare_threshold=0.10):

        t = self.task_type
        is_imb, _, _ = self.pre.check_imbalance(rare_threshold=rare_threshold)

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
    
    def optimisation_model(self, model_name, model_info, X, y, method="light", cv=3, scoring="accuracy", n_jobs=-1):
        """
        Construit un pipeline et optimise les hyperparamètres via BayesSearchCV.
        """
        estimator = model_info['estimator']
        
        # Choix des paramètres et itérations selon la méthode
        if method == "light":
            params_dict = model_info.get('params_light', {})
            n_iter = 5 # Rapide pour le tri initial
        elif method == "full":
            params_dict = model_info.get('params_full', {})
            n_iter = 7 # Approfondi pour le Top 3
        else:
            # Fallback
            params_dict = {}
            n_iter = 1

        # Construction du Pipeline
        # On clone l'estimateur pour ne pas modifier l'original stocké dans self.models
        preproc = self.pre.build_preprocessor(scale_numeric=True)
        pipe = Pipeline([("preprocess", preproc), ("model", clone(estimator))])

        # Si pas de paramètres à optimiser, on retourne le pipe brut fité
        if not params_dict:
            pipe.fit(X, y)
            return pipe

        # Préfixer les paramètres avec "model__" pour le Pipeline
        search_space = {f"model__{k}": v for k, v in params_dict.items()}

        print(f"   -> Optimisation {method} ({n_iter} itér.) pour {model_name}...")

        def status_print(optim_result):
            """Affiche le score à chaque étape de l'optimisation bayésienne"""
            # On récupère tous les scores testés jusqu'ici
            all_scores = optim_result.func_vals
            # Le dernier score obtenu
            current_score = all_scores[-1]
            # Le meilleur score jusqu'ici (attention, skopt minimise, donc on inverse souvent le signe)
            best_score = optim_result.fun

            print(f" > Étape terminée. Score actuel : {-current_score:.4f} | Meilleur global : {-best_score:.4f}")
                
        # Configuration BayesSearchCV
        opt = BayesSearchCV(
            estimator=pipe,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring, # Doit être une métrique unique (ex: 'roc_auc')
            n_jobs=n_jobs,
            random_state=42,
            verbose=0
        )

        try:
            opt.fit(X, y, callback=status_print)
            return opt.best_estimator_
        except Exception as e:
            print(f"   [Warning] Echec optimisation {model_name}: {e}. Utilisation défaut.")
            pipe.fit(X, y)
            return pipe
    


    def select_best_model(self, cv=5, scale_numeric=True, rare_threshold=0.10, n_jobs=-1):
        # --- 1. Vérifications initiales ---
        if self.pre.train_data is None or self.pre.train_labels is None:
            raise ValueError("Appelle pre.split() avant select_best_model().")

        main_key, multi_scoring = self.get_scoring(rare_threshold=rare_threshold)
        if main_key is None:
            raise ValueError(f"Task type non géré: {self.task_type}")
        
        main_scorer_str = multi_scoring[main_key]
        X_train = self.pre.train_data
        y_train = self.pre.train_labels

        if self.task_type == "multiclass_onehot":
            # Conversion OneHot -> Class Index (1D)
            if hasattr(y_train, 'values'):
                y_train = y_train.values.argmax(axis=1)
            else:
                y_train = y_train.argmax(axis=1)
        elif self.task_type != "multilabel":
            # Cas standard (binary/multiclass/regression)
            if hasattr(y_train, 'values'):
                y_train = y_train.values.ravel()
            elif hasattr(y_train, 'ravel'):
                y_train = y_train.ravel()

        models_dict = self.get_models()
        if not models_dict:
            raise ValueError(f"Aucun modèle retourné pour task_type={self.task_type}")

        results = []
        candidates_light = []

        print("\n=== ÉTAPE 1 : Optimisation LIGHT et sélection Top 3 ===")
        
        # --- 2. Boucle Light Optimization (Sécurisée) ---
        for name, info in models_dict.items():
            print(f"-> Traitement modèle : {name}")
            try:
                # A. Optimisation Light
                # (Note: optimisation_model a déjà son propre try/except interne pour renvoyer un pipe par défaut, c'est ok)
                optimized_pipe_light = self.optimisation_model(
                    name, info, X_train, y_train, 
                    method="light", cv=3, scoring=main_scorer_str, n_jobs=n_jobs
                )

                # B. Évaluation Cross-Validation (Le point critique)
                # On utilise error_score="raise" pour que ça plante ici si le modèle est mauvais
                # et que ça aille dans le "except" ci-dessous.
                cv_res = cross_validate(
                    optimized_pipe_light, X_train, y_train, 
                    cv=cv, scoring=multi_scoring, n_jobs=n_jobs, 
                    error_score="raise" 
                )

                # C. Stockage stats
                row = {"model": f"{name} (Light)"}
                for k in multi_scoring.keys():
                    m = float(cv_res[f"test_{k}"].mean())
                    s = float(cv_res[f"test_{k}"].std())
                    sc_name = multi_scoring[k]
                    if isinstance(sc_name, str) and sc_name.startswith("neg_"):
                        m, s = -m, s
                    row[f"{k}_mean"] = m
                    row[f"{k}_std"] = s
                
                results.append(row)
                
                # Ajout aux candidats valides
                candidates_light.append({
                    "name": name,
                    "info": info,
                    "score": row[f"{main_key}_mean"],
                    "pipe": optimized_pipe_light
                })

            except Exception as e:
                print(f"   [SKIP] Le modèle {name} a échoué lors de l'évaluation Light.")
                print(f"   Erreur : {e}")
                continue # On passe au modèle suivant sans arrêter le script

        # --- Vérification de survie ---
        if not candidates_light:
            raise RuntimeError("Tous les modèles ont échoué lors de la phase Light. Vérifiez vos données ou votre scoring.")

        # --- 3. Sélection Top 3 ---
        # Tri décroissant selon le score principal
        candidates_light.sort(key=lambda x: x["score"], reverse=True)
        top_3 = candidates_light[:3]
        
        print(f"\nTop 3 sélectionnés : {[c['name'] for c in top_3]}")

        final_candidates = []
        
        print("\n=== ÉTAPE 2 : Optimisation FULL sur le Top 3 ===")

        # --- 4. Boucle Full Optimization (Sécurisée aussi) ---
        for cand in top_3:
            name = cand["name"]
            info = cand["info"]
            
            try:
                # A. Optimisation Full
                optimized_pipe_full = self.optimisation_model(
                    name, info, X_train, y_train, 
                    method="full", cv=cv, scoring=main_scorer_str, n_jobs=n_jobs
                )
                
                # B. Évaluation Cross-Validation
                cv_res = cross_validate(
                    optimized_pipe_full, X_train, y_train, 
                    cv=cv, scoring=multi_scoring, n_jobs=n_jobs, 
                    error_score="raise"
                )
                
                row = {"model": f"{name} (Full)"}
                for k in multi_scoring.keys():
                    m = float(cv_res[f"test_{k}"].mean())
                    sc_name = multi_scoring[k]
                    if isinstance(sc_name, str) and sc_name.startswith("neg_"):
                        m = -m
                    row[f"{k}_mean"] = m
                
                results.append(row)
                
                final_candidates.append({
                    "name": name,
                    "pipe": optimized_pipe_full,
                    "score": row[f"{main_key}_mean"]
                })
            
            except Exception as e:
                print(f"   [SKIP] Le modèle {name} a échoué lors de l'étape Full. On garde sa version Light si possible.")
                print(f"   Erreur : {e}")
                # Optionnel : On pourrait récupérer le cand['pipe'] (version Light) comme fallback
                # Pour l'instant on l'ignore pour ne pas polluer le Voting avec un modèle instable

        if not final_candidates:
            # Si tout le Full échoue (très improbable), on reprend le Top 1 Light
            print("Attention : Tous les modèles Full ont échoué. Repli sur le meilleur Light.")
            best_light = top_3[0]
            self.best_name = best_light["name"]
            self.best_pipeline = best_light["pipe"]
            self.best_pipeline.fit(X_train, y_train)
            return self.best_name, self.best_pipeline, results, None

        # Tri des finalistes pour trouver le Top 1 actuel
        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        best_single_model = final_candidates[0]
        
        print(f"Meilleur modèle individuel (Full) : {best_single_model['name']} - Score: {best_single_model['score']:.4f}")

        # --- 5. Création et Test du Voting Classifier ---
        winner_name = best_single_model["name"]
        winner_pipe = best_single_model["pipe"]
        winner_score = best_single_model["score"]

        # On ne vote que si on a au moins 2 modèles finaux valides et que c'est de la classification
        if len(final_candidates) > 1 and self.task_type != "multilabel":
            print("\n=== ÉTAPE 3 : Ensemble (Voting) ===")
            
            estimators_list = []
            for cand in final_candidates:
                estimators_list.append((cand["name"], cand["pipe"]))

            voting_model = None
            voting_desc = ""

            # CAS 1 : CLASSIFICATION
            if self.task_type.lower() in ("binary", "multiclass", "multiclass_code", "multiclass_onehot"):
                can_use_soft_voting = True
                for name, _ in estimators_list:
                    # On vérifie dans le dictionnaire original si proba est dispo
                    original_info = models_dict.get(name, {})
                    if not original_info.get('has_predict_proba', False):
                        can_use_soft_voting = False
                
                voting_type = 'soft' if can_use_soft_voting else 'hard'
                voting_desc = f"VotingClassifier ({voting_type})"
                voting_model = VotingClassifier(estimators=estimators_list, voting='hard', n_jobs=n_jobs)

            # CAS 2 : RÉGRESSION
            elif self.task_type.lower() == "regression":
                voting_desc = "VotingRegressor"
                voting_model = VotingRegressor(estimators=estimators_list, n_jobs=n_jobs)

            # Évaluation commune
            if voting_model is not None:
                try:
                    # On utilise les mêmes settings de CV
                    cv_res_v = cross_validate(
                        voting_model, X_train, y_train, 
                        cv=cv, scoring=multi_scoring, n_jobs=n_jobs, 
                        error_score="raise"
                    )
                    
                    v_score_raw = cv_res_v[f"test_{main_key}"].mean()
                    
                    # Gestion du signe (ex: neg_mean_absolute_error)
                    scorer_str = multi_scoring[main_key]
                    if isinstance(scorer_str, str) and scorer_str.startswith("neg_"):
                        v_score = -float(v_score_raw)
                    else:
                        v_score = float(v_score_raw)
                    
                    # Sauvegarde résultat
                    row_v = {"model": f"Ensemble ({voting_desc})"}
                    row_v[f"{main_key}_mean"] = v_score
                    results.append(row_v)
                    
                    print(f"Score {voting_desc}: {v_score:.4f} vs Top 1 Single: {winner_score:.4f}")

                    # Le Voting gagne-t-il ?
                    # Note: Assure-toi que "score" est toujours "plus c'est haut mieux c'est"
                    # Si c'est une erreur (MAE/RMSE), ton code précédent convertissait déjà en positif via le check "neg_"
                    # donc v_score > winner_score est correct si on compare des précisions ou des erreurs inversées (neg_mae).
                    # Si tu compares des MAE pures (positives), il faudrait v_score < winner_score.
                    # Mais vu ta logique précédente (neg_), > est correct.
                    
                    if v_score > winner_score:
                        print(">>> L'ENSEMBLE GAGNE !")
                        winner_name = f"Ensemble_{voting_desc}"
                        winner_pipe = voting_model
                    else:
                        print(">>> LE MODÈLE INDIVIDUEL RESTE MEILLEUR.")

                except Exception as e:
                    print(f"   [WARNING] L'Ensemble a échoué. On reste sur le Top 1.")
                    print(f"   Erreur : {e}")

        # --- 6. Refit Final et Retour ---
        print(f"\nEntraînement final (Refit) du vainqueur : {winner_name}")
        try:
            winner_pipe.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"Le refit final du modèle vainqueur ({winner_name}) a échoué: {e}")

        self.best_name = winner_name
        self.best_pipeline = winner_pipe
        self.cv_summary = results

        # Check Validation
        if self.pre.validation_data is not None and self.pre.validation_labels is not None:
            try:
                X_val = self.pre.validation_data
                y_val = self.pre.validation_labels
                val_scores = {}
                for k, scorer_str in multi_scoring.items():
                    sc = get_scorer(scorer_str)
                    v = float(sc(self.best_pipeline, X_val, y_val))
                    if isinstance(scorer_str, str) and scorer_str.startswith("neg_"):
                        v = -v
                    val_scores[k] = v
                self.val_scores = val_scores
            except Exception as e:
                print(f"Attention: Calcul des scores de validation a échoué: {e}")
                self.val_scores = None

        return self.best_name, self.best_pipeline, self.cv_summary, getattr(self, 'val_scores', None)




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