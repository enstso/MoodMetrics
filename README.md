# MoodMetrics

MoodMetrics est une API Flask d'analyse de sentiments conforme au sujet fourni dans `enonce.pdf`. Elle s'appuie sur deux classifieurs de régression logistique pour estimer les probabilités positive et négative d'un tweet, puis produit un score compris entre `-1` et `1`.

## Architecture prévue

- `moodmetrics/` : API, accès MySQL et inférence.
- `scripts/` : initialisation et réentraînement du modèle.
- `tests/` : tests de l'API et du modèle.
- `artifacts/` : modèle entraîné, non versionné.
- `reports/` : matrices de confusion et rapport d'évaluation.

## Démarrage rapide avec Docker

```bash
docker compose up --build
```

Cette commande démarre MySQL 8.4, importe le dataset `data/tweets.csv`, expose l'API sur `http://localhost:8000` et lance un service cron qui réentraîne le modèle chaque dimanche à 03:00.

Au premier démarrage, l'API entraîne automatiquement un modèle si aucun artefact n'existe, puis génère les métriques et le rapport PDF si les fichiers de rapport sont absents. L'import du dataset ajoute uniquement les tweets absents afin de préserver les annotations déjà stockées. Les volumes `mysql_data`, `model_data` et `report_data` conservent les annotations, le modèle et les livrables d'évaluation.

## API

### `POST /api/v1/sentiments`

Analyse une liste de tweets et renvoie, pour chacun, un **score de sentiment entre -1 et 1**
(négatif → proche de -1, positif → proche de 1, neutre → proche de 0).

**Requête** — objet JSON avec une clé `tweets` :

```json
{
  "tweets": ["J'adore ce produit", "Service décevant"]
}
```

**Réponse** — chaque tweet est associé à son score :

```json
{
  "J'adore ce produit": 0.82,
  "Service décevant": -0.74
}
```

**Exemple d'appel `curl`** :

```bash
curl -X POST http://localhost:8000/api/v1/sentiments \
  -H 'Content-Type: application/json' \
  -d '{"tweets":["J adore ce service","Cette expérience est horrible"]}'
```

Réponse attendue (les scores exacts dépendent du modèle entraîné) :

```json
{
  "J adore ce service": 0.6683,
  "Cette expérience est horrible": -0.329
}
```

**Formats et erreurs :**

- L'endpoint accepte aussi directement un tableau JSON : `["Très bien", "Bof"]`.
- Une liste vide, un JSON invalide, un tweet vide ou une valeur non textuelle → `400 Bad Request`
  (corps `{"error": "..."}`).
- Si aucun modèle n'est encore entraîné → `503 Service Unavailable` avec un message explicite.

### `GET /api/v1/model/metrics`

Expose les dernières métriques d'évaluation générées depuis `reports/evaluation.json` :

```bash
curl http://localhost:8000/api/v1/model/metrics
```

Si l'évaluation n'a pas encore été générée, l'API renvoie `503 Service Unavailable`.

## Développement local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Pour créer le schéma puis réentraîner manuellement le modèle :

```bash
python -m scripts.init_db
python -m scripts.import_dataset
python -m scripts.train_model
```

## Optimisation de l'algorithme

Une étape de benchmark compare plusieurs pipelines sur les mêmes données, avec un split stratifié
80/20 (`random_state=42`) et un score principal basé sur la moyenne des F1-scores des deux
classifieurs (`positive` et `negative`).

```bash
python -m scripts.optimize_model
```

Tentatives mesurées dans `reports/optimization_results.json` :

| Rang | Pipeline | F1 moyen | Accuracy moyenne | Temps d'entraînement |
| --- | --- | ---: | ---: | ---: |
| 1 | `hybrid_char_word_tfidf_logreg` | 0.9704 | 0.9767 | 0.3822s |
| 2 | `baseline_word_tfidf_logreg` | 0.9703 | 0.9767 | 0.0726s |
| 3 | `tuned_word_tfidf_logreg` | 0.9672 | 0.9743 | 0.0892s |

**Solution retenue.** La version la plus optimisée est
`hybrid_char_word_tfidf_logreg`, définie dans `moodmetrics/modeling.py`. Elle combine :

- un TF-IDF de mots en 1-2 grammes pour capturer les expressions courtes ;
- un TF-IDF de caractères `char_wb` en 3-5 grammes pour mieux gérer fautes, hashtags, emojis,
  accents et variations typiques des tweets ;
- une `LogisticRegression` pondérée (`class_weight="balanced"`, `C=2.0`, `solver="liblinear"`).

Ce modèle devient le pipeline par défaut utilisé par `scripts.train_model` et
`scripts.evaluate_model`. Le gain est faible mais mesurable sur le F1 moyen, et le coût reste
acceptable pour le volume du projet.

## Évaluation du modèle et rapport

Le module d'évaluation mesure la qualité des deux classifieurs et produit un **rapport PDF en
français**, accompagné des images des matrices de confusion.

```bash
python -m scripts.evaluate_model    # calcule les métriques -> reports/evaluation.json + PNG
python -m scripts.generate_report   # construit le PDF -> reports/rapport_evaluation.pdf
```

> Le PDF est écrit à l'emplacement `REPORT_PATH` (par défaut `reports/rapport_evaluation.pdf`).

**Méthodologie.** Les tweets sont chargés depuis la table `tweets`, **dédupliqués par texte**
(pour éviter qu'un même tweet se retrouve à la fois en entraînement et en test — fuite de données),
puis répartis par un **split stratifié 80/20** (`random_state=42`). Chaque classifieur est
réévalué indépendamment : matrice de confusion, précision, rappel et F1-score par classe.

**Jeu de tweets supplémentaire.** Le corpus `data/tweets.csv` est synthétique et très gabarité :
évalué seul, il donne des métriques *parfaites* (peu représentatives). Pour crédibiliser
l'évaluation, `data/tweets_supplementary.csv` ajoute des tweets réalistes et difficiles (ironie,
négation, argot, fautes, emojis, cas neutres). Ce fichier est automatiquement fusionné au pool
d'évaluation (chemin configurable via `SUPPLEMENTARY_DATASET_PATH`).

**Artefacts produits dans `reports/`** :

- `evaluation.json` — toutes les métriques calculées ;
- `optimization_results.json` — comparaison des pipelines testés ;
- `confusion_matrix_positive.png` / `confusion_matrix_negative.png` — les matrices de confusion ;
- `rapport_evaluation.pdf` — le rapport d'évaluation (introduction, matrices interprétées,
  tableau des mesures, analyse des forces/faiblesses/biais, recommandations).

Dans l'environnement Docker, le service `trainer` régénère chaque dimanche à 03:00 le modèle,
les métriques et le rapport, et journalise le résumé des performances.

## Qualité continue

Le workflow GitHub Actions exécute les tests Python, valide `compose.yaml` et construit l'image Docker de l'API à chaque push sur `main` et à chaque pull request.
