# Machine Learning track

## Training / evaluation / inference

```bash
python src/main.py              # train, evaluate, export metrics, sample inference
python src/tune_hyperparams.py  # GridSearchCV on RandomForest (F1)
python src/etl_batch.py         # batch inference load
```

## Metrics

- Accuracy, F1, classification report  
- Confusion matrix CSV per model  
- Feature importance for tree models  
- Sample: `data/outputs/metrics.sample.json`  

## Reusable pipeline

`src/sklearn_pipeline.py` — StandardScaler + RandomForest as a single sklearn `Pipeline`.

## Hyperparameter tuning

Grid over `n_estimators`, `max_depth`, `min_samples_leaf` with stratified 3-fold CV, scoring **F1**.
