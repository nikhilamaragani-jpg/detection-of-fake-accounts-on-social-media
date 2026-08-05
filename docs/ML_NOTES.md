# Machine Learning notes

## Training

- Models: RandomForest, LogisticRegression, GradientBoosting  
- `class_weight=balanced` where supported  

## Evaluation

- Accuracy + F1 + classification report  
- Confusion matrix export  
- Feature importance for tree models  

## Inference

`predict_account(model, feature_vector)` returns label + p_fake.

## Hyperparameter tuning (TODO)

GridSearchCV on RF `n_estimators` / `max_depth` with stratified CV.
