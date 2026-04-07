# Explanation Guide for Viewers

## What this chart is showing
This visualization explains why the model flagged one transaction as suspicious.
The gold node is the target transaction being explained.

## How to read the nodes
- Gold: target transaction under investigation
- Red: known illicit transaction
- Blue: known licit transaction
- Gray: unknown label transaction

## How to read the edges
- Thicker/darker edges had stronger influence on the model's illicit score
- Arrows indicate transaction direction

## Case details
- Target txId: 94336306
- Hops used for explanation: 2
- Licit nodes found in explanation neighborhood: 0
- Minimum licit nodes requested: 1
- Predicted illicit probability: 0.5587
- Decision threshold used: 0.5100

## Key conclusions for this case
- Decision at threshold: FLAGGED illicit
- Strongest edge influence: tx:94152708 -> tx:94336308 (importance=0.6307)
- Local explanation mix: illicit=3, licit=0, unknown=0
- Global labeled mix (for context): illicit=4545/46564 (9.76%), licit=42019/46564 (90.24%)
- Caveat: local neighborhood composition does not equal whole-dataset composition.

## Top influential edges
1. tx:94152708 -> tx:94336308 (importance=0.6307)
2. tx:94336308 -> tx:94336306 (importance=0.5488)
