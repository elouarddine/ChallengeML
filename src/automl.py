import time
import pandas as pd
import numpy as np
import os
try:
    from src.Preprocess import Preprocess
    from src.Models import Models
    from src.Evaluate import Evaluate
except ImportError:
    from Preprocess import Preprocess
    from Models import Models
    from Evaluate import Evaluate




class ChallengeML:
    def __init__(self):
        self.preprocessor = None
        self.models_manager = None
        self.evaluator = None
        self.best_pipeline = None

    def fit(self, path_repertory: str, time_budget=None):
        """
        Processus complet d'entraînement :
        1. Chargement et Préprocessing
        2. Détection du type de tâche
        3. Split Train/Valid/Test
        4. Sélection et optimisation du meilleur modèle
        """
        start_time = time.time()
        print(f"--- Début de l'entraînement (Budget : {time_budget}) ---")

        # 1. Initialisation & Chargement
        self.preprocessor = Preprocess(path_repertory)
        self.preprocessor.load()
        
        # 2. Split des données (Train / Val / Test)
        # On utilise les paramétres par défaut de split (0.2, 0.2)
        
        self.preprocessor.split()
        
        print(f"Tâche détectée : {self.preprocessor.detect_task_type()}")

        # 3. Initialisation du gestionnaire de modèles
        # On passe l'objet preprocess qui contient toutes les infos (data, types, indices...)
        self.models_manager = Models(self.preprocessor)
        
        # 4. Sélection du meilleur modèle
        # Cette méthode gère l'optimisation light + full et le voting si applicable
        self.models_manager.select_best_model() 
        
        # Récupération du pipeline gagnant
        self.best_pipeline = self.models_manager.best_pipeline
        
        # Sauvegarde des résultats
        self.models_manager.save_results()
        
        elapsed_time = time.time() - start_time
        if self.models_manager.best_name:
             print(f"--- Fin de l'entraînement ({elapsed_time:.2f}s). Meilleur modèle : {self.models_manager.best_name} ---")


    def eval(self):
        """
        Évaluation du meilleur modèle sur le jeu de test (split interne).
        Affiche les scores et matrices de confusion.
        """
        if self.best_pipeline is None:
            print("Erreur : Aucun modèle entraîné. Veuillez appeler fit() d'abord.")
            return None

        print("\n--- Évaluation sur le Test Set ---")
        
        # Initialisation de l'évaluateur
        self.evaluator = Evaluate(self.preprocessor, self.models_manager, self.best_pipeline)
        
        # Calcul des scores
        scores, conf_matrix, report = self.evaluator.evaluate_on_test()
        
        # Affichage
        print("\nScores obtenus :")
        for metric, val in scores.items():
            print(f"  - {metric}: {val:.4f}")

        if conf_matrix is not None:
            print("\nMatrice de confusion :")
            print(conf_matrix)
        
        if report is not None:
            print("\nRapport de classification :")
            print(report)
            
        return scores






