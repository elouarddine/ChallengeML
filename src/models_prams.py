"""
Docstring for models
ce module définit une fonction pour obtenir des modèles de machine learning

les modèle sont retournés en dictionnaire avec:
- le nom du modèle comme clé
- un dictionnaire comme valeur contenant:
    - 'estimator': l'instance du modèle
    - 'has_predict_proba': booléen indiquant si le modèle supporte predict_proba
    - 'params_light': dictionnaire des hyperparamètres pour une recherche légère (première optimisation)
    - 'params_full': dictionnaire des hyperparamètres pour une recherche complète (optimisation approfondie)
Les modèles inclus sont choisis en fonction du type de problème.

"""

def get_models(problem_type):
    models={}
    if problem_type in ("binary", "multiclass","multiclass_code", "multiclass_onehot"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        models.update({
            'LogisticRegression': {
            'estimator': LogisticRegression(max_iter=1000, solver='lbfgs'),
            'has_predict_proba': True,
            'params_light': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            },
            'params_full': {
                'C': [0.01, 0.1, 1, 10, 100],
                'class_weight': [None, 'balanced']
                # Note: solver lbfgs ne supporte que l2 ou None
            }
        }})
        models.update({
            'RandomForest': {
            'estimator': RandomForestClassifier(n_jobs=-1, random_state=42),
            'has_predict_proba': True,
            'params_light': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_leaf': [1, 4]
            },
            'params_full': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced']
            }
        }})

        models.update({
            'HistGradientBoosting': {
            'estimator': HistGradientBoostingClassifier(random_state=42),
            'has_predict_proba': True,
            'params_light': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_iter': [100, 200]
            },
            'params_full': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_iter': [100, 200, 500],
                'max_leaf_nodes': [31, 50, 100],
                'max_depth': [-1, 10, 20],
                'min_samples_leaf': [20, 50, 100]
            }
        }})

        

        models.update({
            'XGBoost': {
            'estimator': XGBClassifier( eval_metric='logloss', random_state=42, n_jobs=-1),
            'has_predict_proba': True,
            'params_light': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6]
            },
            'params_full': {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
        }})


        models.update({
            'LightGBM': {
            'estimator': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            'has_predict_proba': True,
            'params_light': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50]
            },
            'params_full': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70, 100],
                'max_depth': [-1, 10, 20],
                'min_child_samples': [20, 50, 100],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        }})

        models.update({
            'CatBoost': {
            'estimator': CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False),
            'has_predict_proba': True,
            'params_light': {
                'iterations': [100, 200],
                'learning_rate': [0.03, 0.1],
                'depth': [4, 6]
            },
            'params_full': {
                'iterations': [200, 500, 1000],
                'learning_rate': [0.01, 0.03, 0.1],
                'depth': [4, 6, 7],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'bagging_temperature': [0, 1]
            }
        }})

        return models
    
    elif problem_type == 'regression':
        from sklearn.linear_model import ElasticNet,Ridge
        from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
        
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        models.update({
            'ElasticNet': {
            'estimator': ElasticNet(random_state=42, max_iter=1000),
            'params_light': {
                'alpha': [0.1, 1.0, 10.0],  # Force de la régularisation
                'l1_ratio': [0.1, 0.5, 0.9] # 1 = Lasso, 0 = Ridge
            },
            'params_full': {
                'alpha': [0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'selection': ['cyclic', 'random']
            }
        }})
        models.update({
            'RandomForest': {
            'estimator': RandomForestRegressor(n_jobs=-1, random_state=42),
            'params_light': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 10]
            },
            'params_full': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        }})

        models.update({
            'HistGradientBoosting': {
            'estimator': HistGradientBoostingRegressor(random_state=42),
            'params_light': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_iter': [100, 200]
            },
            'params_full': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_iter': [100, 200, 500],
                'max_leaf_nodes': [31, 50, 100],
                'max_depth': [-1, 10, 20],
                'min_samples_leaf': [20, 50, 100]
            }
        }})

        
        models.update({
            'XGBoost': {
            'estimator': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'params_light': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6]
            },
            'params_full': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        }})
        models.update({
            'LightGBM': {
            'estimator': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'params_light': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50]
            },
            'params_full': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70, 100],
                'max_depth': [-1, 10, 20],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        }})
        models.update({
            'CatBoost': {
            'estimator': CatBoostRegressor(verbose=0, random_state=42, allow_writing_files=False, loss_function='RMSE'),
            'params_light': {
                'iterations': [100, 200],
                'learning_rate': [0.03, 0.1],
                'depth': [4, 6]
            },
            'params_full': {
                'iterations': [200, 500, 1000],
                'learning_rate': [0.01, 0.03, 0.1],
                'depth': [4, 6, 7],
                'l2_leaf_reg': [1, 3, 5, 9]
            }
        }})
        return models
    
    