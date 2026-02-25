Projet RecSys : Système de recommandation hybride et responsable

## Objectif du projet

Ce projet vise à concevoir en Python un **système de recommandation** (films, séries, livres ou contenus éducatifs) qui
combine :

- un moteur **collaboratif,** basé sur l’historique d’interactions,
- un moteur **contenu,** basé sur la description et le synopsis,
- un module **d’explication** (“Pourquoi cette recommandation ?”),
- et un module **responsable** (diversité, nouveauté, réduction du biais de popularité).

---

## Dataset et Volume

- **Données principales :** Nous utilisons le dataset **MovieLens 25M**, qui constitue le standard de l'industrie pour ce type d'exercice.
- **Enrichissement :** Les données seront couplées à l'API de **TMDB (The Movie Database)** afin de récupérer les synopsis complets et les posters des films, ce qui est indispensable pour la recommandation sémantique (par contenu).

---

## Ce qui sera livré

À l'issue de ce projet, les éléments suivants seront produits :

- Un repository Git structuré (`src/`, `notebooks/`, `data/`, `app/`, `tests/`).
- Des notebooks d'analyse exploratoire (AED), d'entraînement et d'évaluation propres.
- Une application de démonstration interactive développée avec **Streamlit**.
- Un rapport de synthèse de 6 à 10 pages présentant la méthode, les résultats, les limites et les pistes d'amélioration.

---

## Comment reproduire l'ingestion des données

Le projet est conçu pour être entièrement reproductible. Pour télécharger, nettoyer et préparer le dataset brut en un format optimisé (Parquet), exécutez la commande suivante à la racine du projet:

`python scripts/make_dataset.py`

Pour lancer le pipeline complet avec la configuration finale (nettoyage itératif, création des tables et split train/val/test), utilisez :

`python scripts/make_dataset.py --config configs/dataset_v1.yaml`

---

## **Installation et environnement de développement**

Si vous souhaitez contribuer au code ou lancer les tests, voici comment configurer votre environnement :

- **Cloner le projet :**
`git clone https://github.com/ZaKoLoT/recsys-ml25m.git`
- **Créer et activer l'environnement virtuel :**
Sur Mac/Linux : `python3 -m venv .venv puis source .venv/bin/activate`
Sur Windows : `python -m venv .venv puis .venv\Scripts\activate`
- **Installer les dépendances de développement :**
`pip install -r requirements-dev.txt`
- **Activer les vérifications automatiques (pre-commit) :**
`pre-commit install`

## Outils de qualité

Le projet utilise des standards professionnels stricts :

- **Tests unitaires :** `pytest -q`
- **Linting (Ruff) :** `ruff check .`
- **Formatage (Ruff) :** `ruff format .`
- **Pre-commit manuel :** `pre-commit run --all-files`
