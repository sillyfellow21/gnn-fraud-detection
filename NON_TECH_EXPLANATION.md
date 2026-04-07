# What This Project Does (Non-Technical)

## One-line summary
This project helps identify suspicious Bitcoin activity by looking at how transactions are connected, not just at each transaction in isolation.

## The business problem
Fraudsters often work in groups and hide behind many linked accounts or transactions. If we only review each transaction one by one, we can miss these coordinated patterns.

## What the system looks at
The model uses three kinds of information:
- Transaction details (amount, behavior signals, and other attributes)
- Transaction relationships (who is connected to whom)
- Time order (older activity is used for training, newer activity is used for testing)

This mirrors real-world risk operations, where we learn from the past and score new activity.

## What happens step by step
1. The system loads historical transaction data.
2. It learns patterns of known licit vs illicit behavior.
3. It scores newer transactions by risk level.
4. It reports quality metrics focused on fraud use cases (not just overall accuracy).
5. It generates an explanation graph for a flagged case, showing which nearby connections most influenced the alert.

## What output you get
- A risk model that can rank transactions from low to high suspicion.
- Fraud-focused evaluation metrics (Precision-Recall AUC and Macro F1).
- A visual explanation image (`outputs/gnnexplainer_tp_subgraph.png`) that helps analysts understand why a transaction was flagged.
- A benchmark comparison between:
  - Graph-based model (GraphSAGE)
  - Traditional non-graph baseline (XGBoost)

## Why this is useful for product and operations
- Better analyst trust: each alert can be explained visually.
- Better prioritization: the model can rank cases so teams investigate the highest-risk ones first.
- Better policy confidence: decisions can be tied to understandable network patterns, not just a black-box score.

## Important current finding
The project now includes a fair side-by-side benchmark on the same data split. In current runs, the traditional baseline is still outperforming the graph model. That means the project is complete as a comparison framework, but additional graph-model tuning is needed before claiming graph superiority.

## How to use this in a portfolio story
A strong, honest summary is:
- "I built a production-style fraud pipeline with temporal evaluation, explainability, and model benchmarking."
- "I compared graph and non-graph approaches fairly on the same split."
- "I can explain model decisions for investigators and iterate based on measured results."
