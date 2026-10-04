# Machine Learning track

## Training / evaluation / inference

```bash
python src/main.py              # train, evaluate, export metrics, sample inference
python src/tune_hyperparams.py  # GridSearchCV on RandomForest (F1)
python src/etl_batch.py         # batch inference load
```

## Metrics

- Training-fold stratified cross-validation F1 selects the model without using the holdout labels.
- Holdout reporting includes accuracy, balanced accuracy, fake-class precision/recall/F1, specificity, ROC AUC, average precision, and classification report.
- `data/outputs/metrics.json` includes a fixed-label confusion matrix and the holdout row count.
- Confusion matrix CSV per model  
- Feature importance for tree models  
- Sample: `data/outputs/metrics.sample.json`  

The bundled dataset is very small and its source/ground truth are undocumented.
Treat its holdout metrics as an execution example, not evidence of generalized
performance. See [`data/README.md`](../data/README.md).

## Reusable pipeline

`src/sklearn_pipeline.py` — StandardScaler + RandomForest as a single sklearn `Pipeline`.

## Hyperparameter tuning

Grid over `n_estimators`, `max_depth`, `min_samples_leaf` with stratified 3-fold CV, scoring **F1**. The held-out split is reserved for final evaluation.
