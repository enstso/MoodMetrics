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

Au premier démarrage, l'API entraîne automatiquement un modèle si aucun artefact n'existe. L'import du dataset ajoute uniquement les tweets absents afin de préserver les annotations déjà stockées. Les volumes `mysql_data` et `model_data` conservent les annotations et le modèle.

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
- `confusion_matrix_positive.png` / `confusion_matrix_negative.png` — les matrices de confusion ;
- `rapport_evaluation.pdf` — le rapport d'évaluation (introduction, matrices interprétées,
  tableau des mesures, analyse des forces/faiblesses/biais, recommandations).

Dans l'environnement Docker, le service `trainer` régénère chaque dimanche à 03:00 le modèle,
les métriques et le rapport, et journalise le résumé des performances.

## Qualité continue

Le workflow GitHub Actions exécute les tests Python, valide `compose.yaml` et construit l'image Docker de l'API à chaque push sur `main` et à chaque pull request.
