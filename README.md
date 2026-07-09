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

## API cible

`POST /api/v1/sentiments`

```json
{
  "tweets": ["J'adore ce produit", "Service décevant"]
}
```

La réponse associe chaque tweet à un score de sentiment :

```json
{
  "J'adore ce produit": 0.82,
  "Service décevant": -0.74
}
```

Exemple :

```bash
curl -X POST http://localhost:8000/api/v1/sentiments \
  -H 'Content-Type: application/json' \
  -d '{"tweets":["J adore ce service","Cette expérience est horrible"]}'
```

L'endpoint accepte aussi directement un tableau JSON. Une liste vide, un JSON invalide ou une valeur non textuelle produit une erreur `400`. L'absence de modèle produit une erreur `503` explicite.

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
