import os
import sys

# Ajouter le répertoire courant au path pour pouvoir importer src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.automl import ChallengeML

def main():
    # Définition du chemin vers les données (exemple avec data_A)
    # Ajustez le 'data_A' pour tester d'autres datasets (data_B, data_C, etc.)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'data_C')
    data_path = os.path.normpath(data_path)
    
    # Vérification que le dossier existe
    if not os.path.exists(data_path):
        print(f"Erreur: Le dossier de données n'existe pas : {data_path}")
        # Tentative de fallback ou message d'aide
        print(f"Chemin cherché : {os.path.abspath(data_path)}")
        return

    # Instanciation de l'AutoML
    automl = ChallengeML()
    
    # Entraînement
    print(f"Lancement de l'entraînement sur : {data_path}")
    # On met un budget temps arbitraire (ex: 300s) si votre implémentation l'utilise
    automl.fit(data_path, time_budget=300) 
    
    # Evaluation
    print("Lancement de l'évaluation...")
    automl.eval()

if __name__ == "__main__":
    main()
