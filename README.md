# MoodMetrics

MoodMetrics est une API Flask d'analyse de sentiments conforme au sujet fourni dans `enonce.pdf`. Elle s'appuie sur deux classifieurs de régression logistique pour estimer les probabilités positive et négative d'un tweet, puis produit un score compris entre `-1` et `1`.

## Architecture prévue

- `moodmetrics/` : API, accès MySQL et inférence.
- `scripts/` : initialisation et réentraînement du modèle.
- `tests/` : tests de l'API et du modèle.
- `artifacts/` : modèle entraîné, non versionné.
- `reports/` : matrices de confusion et rapport d'évaluation.

## Démarrage rapide

Le démarrage Docker complet sera disponible avec `docker compose up --build`. Une copie de `.env.example` vers `.env` permet de personnaliser la configuration locale.

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

