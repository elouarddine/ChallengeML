# AutoML Challenge – Méthodologie IA & Méthodes Classiques

## Objectif du projet
Ce projet consiste à construire une **pipeline AutoML** capable d’automatiser les étapes principales d’un entraînement Machine Learning :

- Charger un dataset au format **.data / .solution / .type**
- Détecter automatiquement le **type de tâche** (classification / régression, variantes)
- Appliquer un **prétraitement** adapté (numérique + catégoriel, dense/sparse)
- Tester plusieurs **modèles scikit-learn**
- Faire une **optimisation Light → Full** des hyperparamètres (BayesSearchCV)
- Sélectionner le meilleur modèle (avec option **Voting** quand possible)
- Évaluer le modèle sélectionné sur :
  - **DEV (validation)** pendant `fit()`
  - **TEST** pendant `eval()`
- Générer automatiquement un dossier `resultats/` avec les scores et le report

---

## Données
Chaque dataset est composé de 3 fichiers :

- `data.data` : features (sans en-têtes)
- `data.solution` : target (y)
- `data.type` : type de chaque colonne (`Numerical` / `Categorical`)

Le projet attend un chemin “base” (sans extension), par exemple :
- `.../data_C`  → le code charge `data_C.data`, `data_C.solution`, `data_C.type`

---

## Fonctionnement global (pipeline)
1) **Chargement**
- Lecture des fichiers `.data/.solution/.type`

2) **Détection du type de tâche**
- Classification : `binary`, `multiclass`, `multiclass_onehot`, `multilabel`
- Régression : `regression`, `regression_multioutput`

3) **Split des données**
- Train / DEV (validation) / Test (selon votre implémentation dans `Preprocess`)

4) **Construction du pipeline**
- Prétraitement + modèle (via `ConstructeurPipeline`)

5) **Sélection de modèle**
- Phase **Light** : optimisation rapide pour chaque modèle, puis sélection Top-K
- Phase **Full** : optimisation plus poussée seulement sur les Top-K
- Option **Voting** (si possible) sur les candidats finaux

6) **Entraînement final**
- Fit du meilleur modèle sur tout le train

7) **Validation DEV**
- Calcul des métriques sur le set DEV pendant `fit()`

8) **Évaluation TEST**
- `eval()` évalue le même modèle sélectionné sur le set TEST

---

## Architecture du projet
### Arborescence (principaux fichiers)
```

automl_project/
├── README.md
├── automl.py
├── src/
│   ├── Preprocess.py
│   ├── Models.py
│   ├── ModelesConditions.py
│   ├── ConstructeurPipeline.py
│   └── Evaluate.py
└── resultats/    (généré automatiquement : runs + rapports)

````

---

## Installation


### 1) Installer les dépendances

Dépendances principales :

* `numpy`
* `pandas`
* `scikit-learn`
* `scikit-optimize` # pour l'utilisation de skopt pour utiliser BayesSearchCV


Installation :

```bash
pip install numpy pandas scikit-learn scikit-optimize
```

---

## Utilisation

Le fichier Python qui exécute le projet (ex: `run.py`) doit être placé **au même niveau que `automl.py`** (à la racine du projet).

Exemple minimal :

```python
import automl

data_dest = "/info/corpus/ChallengeMachineLearning/data_C"
automl.fit(data_dest)

res = automl.eval()
print(res)

```

---

## Résultats

Après exécution, un dossier `resultats/` est généré automatiquement.

Il contient, pour chaque run :

* un report de sélection (modèles testés, top-k, meilleur modèle, stage)
* les métriques sur **DEV (validation)**
* les métriques sur **TEST**

---

## Auteurs & Encadrants

**Contributeurs :**
Salah Eddine Elouardi
Mohammed Abdelhadi Benmansour
Odilon Ilboudo

**Encadrants :**
Aghilas Sini, Nicolas Dugue
