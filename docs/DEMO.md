# Demo walkthrough

```bash
pip install -r requirements.txt
python src/main.py
```

```text
Fake Account Detection  |  B.Tech ML Prototype
Dataset size: N
--- Model Evaluation ---
RandomForest / LogisticRegression / GradientBoosting
  Accuracy · F1
Best model by F1: ...
--- Sample Prediction ---
Predicted Label: Fake / Genuine
```

```mermaid
flowchart LR
  CSV --> PRE[Preprocess]
  PRE --> M1[RF]
  PRE --> M2[LogReg]
  PRE --> M3[GBM]
  M1 --> EVAL[Evaluate F1]
  M2 --> EVAL
  M3 --> EVAL
  EVAL --> PRED[Predict + SQLite log]
```
