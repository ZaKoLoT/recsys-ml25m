# Journal des Décisions Techniques (Projet RecSys)

Ce document centralise nos choix techniques fondamentaux afin de garantir une évaluation cohérente tout au long du projet.

## 1. Type de recommandation et règles de gestion
* **Approche choisie :** Recommandation implicite.
* **Seuil de validation (rating_threshold) :** 4.0.
* **Règle stricte :** Afin de simplifier notre dataset (qui contient pour chaque utilisateur une note de 0.5 à 5.0), nous allons transformer ce système de notation en interactions positives. Si la note d'un utilisateur pour un film est $\ge$ 4.0, l'interaction vaut 1 (l'utilisateur a aimé le film). Dans le cas contraire, la ligne est ignorée et retirée de notre jeu de données. Nous n'exploitons pas les avis négatifs.

## 2. Protocole d’évaluation
* **Méthode de split par utilisateur :** Nous trions l'historique de chaque utilisateur par date (`timestamp`).
  * *Pourquoi ce choix :* C'est pour simuler "le futur" et éviter les fuites de données (data leakage). Dans la vraie vie, on devine ce qu'un utilisateur va regarder demain en se basant sur hier.
* **Paramètres de séparation (V0) :** `N_testV0 = 5`. Les 5 dernières interactions constituent le Test, le reste constitue le Train. *(Note : pour les versions ultérieures, nous introduirons un set de validation avec `N_val = 2`).*
* **Mesure de performance (Top-K) :** L'évaluation se fera sur des listes courtes et moyennes avec `K=10` et `K=20`. Nous utiliserons les métriques **Recall@K** et **NDCG@K**.
  * *Pourquoi K=10 :* C'est une liste courte (type Netflix/Spotify). Cela vérifie si le système est pertinent tout de suite, en haut de la liste.
  * *Pourquoi K=20 :* Donne une évaluation plus complète et stable sur une liste un peu plus longue.
  * *Pourquoi ces deux-là :* C'est le bon compromis (moins de 10 est trop punitif, plus de 20 est inutile pour l'utilisateur).
* **Baselines (Semaine 2) :** Modèle de Popularité basique et modèle ALS (Alternating Least Squares pour feedback implicite).

## 3. Livrables de données attendus (Fin Semaine 1)
À l'issue du sprint d'ingestion, nous générerons les fichiers suivants au format Parquet :
* **Fichiers de base nettoyés :** `interactions.parquet`, `movies.parquet`, `tags.parquet`, `links.parquet`, `genome-scores.parquet`, `genome-tags.parquet`.
* **Fichiers du split de contrôle (V0) :** `interactions_train_v0.parquet` et `interactions_test_v0.parquet`.

## 4. Hypothèses et limites
* **Cold-start :** Ce filtrage initial ne gère pas nativement le problème de démarrage à froid (nouveaux items ou utilisateurs sans historique).
* **Conflits temporels :** Si un utilisateur possède plusieurs interactions avec des timestamps strictement identiques, l'ordre de tri pour le split temporel pourra être arbitraire.
