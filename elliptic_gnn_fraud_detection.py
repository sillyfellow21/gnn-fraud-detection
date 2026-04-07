"""Train GraphSAGE + XGBoost baseline on Elliptic and generate explanations.

Expected files in --data-dir:
- elliptic_txs_features.csv
- elliptic_txs_edgelist.csv
- elliptic_txs_classes.csv
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.utils import k_hop_subgraph


LOG_LEVEL = 1  # 0=quiet, 1=default, 2=verbose


@dataclass
class SplitConfig:
    """Train/val temporal split ratios."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class TrainConfig:
    """GraphSAGE training hyperparameters."""

    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.35
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 250
    grad_clip: float = 2.0
    early_stopping_patience: int = 40
    seed: int = 42


@dataclass
class BaselineConfig:
    """XGBoost baseline hyperparameters."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    seed: int = 42


def set_log_level(*, quiet: bool, verbose: bool) -> None:
    """Configure global log verbosity."""

    global LOG_LEVEL
    if quiet:
        LOG_LEVEL = 0
    elif verbose:
        LOG_LEVEL = 2
    else:
        LOG_LEVEL = 1


def log(message: str, *, level: int = 1) -> None:
    """Small logging helper with three levels."""

    if LOG_LEVEL >= level:
        print(message)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_temporal_cutoffs(timesteps: np.ndarray, split_cfg: SplitConfig) -> Tuple[int, int]:
    """Compute temporal cutoffs from unique timesteps."""

    unique_steps = np.sort(np.unique(timesteps))
    if unique_steps.size < 3:
        raise ValueError("Need at least 3 unique timesteps for train/val/test split.")

    train_end_idx = max(0, int(len(unique_steps) * split_cfg.train_ratio) - 1)
    val_end_idx = max(train_end_idx + 1, int(len(unique_steps) * (split_cfg.train_ratio + split_cfg.val_ratio)) - 1)
    val_end_idx = min(val_end_idx, len(unique_steps) - 2)

    train_cutoff = int(unique_steps[train_end_idx])
    val_cutoff = int(unique_steps[val_end_idx])
    return train_cutoff, val_cutoff


def _filter_edges_by_node_mask(edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Keep only edges where both source and target are in node_mask."""

    keep = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return edge_index[:, keep]


def load_elliptic_as_pyg_data(data_dir: str, split_cfg: SplitConfig) -> Data:
    """Load Elliptic CSV files and build a PyG Data object."""

    data_path = Path(data_dir)
    features_path = data_path / "elliptic_txs_features.csv"
    edges_path = data_path / "elliptic_txs_edgelist.csv"
    classes_path = data_path / "elliptic_txs_classes.csv"

    if not features_path.exists() or not edges_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            "Could not find Elliptic files. Expected: "
            "elliptic_txs_features.csv, elliptic_txs_edgelist.csv, elliptic_txs_classes.csv"
        )

    features_df = pd.read_csv(features_path, header=None)
    edges_df = pd.read_csv(edges_path)
    classes_df = pd.read_csv(classes_path)

    # Standardize class labels and map to {illicit:1, licit:0, unknown:-1}.
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes_df["class"] = classes_df["class"].astype(str).map(label_map)

    # Features file columns: [txId, timestep, feature_1, ..., feature_n].
    tx_ids = features_df.iloc[:, 0].astype(np.int64).to_numpy()
    timesteps = features_df.iloc[:, 1].astype(np.int64).to_numpy()
    x_np = features_df.iloc[:, 2:].astype(np.float32).to_numpy()

    # Merge labels by txId. Unseen labels become unknown (-1).
    class_map = dict(zip(classes_df["txId"].astype(np.int64), classes_df["class"].astype(np.int64)))
    y_np = np.array([class_map.get(tx_id, -1) for tx_id in tx_ids], dtype=np.int64)

    # Build contiguous node indices for PyG graph tensors.
    tx_to_idx: Dict[int, int] = {int(tx): idx for idx, tx in enumerate(tx_ids.tolist())}

    src = edges_df["txId1"].astype(np.int64).to_numpy()
    dst = edges_df["txId2"].astype(np.int64).to_numpy()

    edge_src: List[int] = []
    edge_dst: List[int] = []
    for s, d in zip(src, dst):
        si = tx_to_idx.get(int(s))
        di = tx_to_idx.get(int(d))
        if si is None or di is None:
            continue
        edge_src.append(si)
        edge_dst.append(di)

    if not edge_src:
        raise ValueError("No valid edges were created. Check txId alignment across CSV files.")

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    timestep_t = torch.from_numpy(timesteps)

    labeled_mask = y >= 0
    train_cutoff, val_cutoff = _compute_temporal_cutoffs(timesteps, split_cfg)

    # Label supervision masks.
    train_mask = (timestep_t <= train_cutoff) & labeled_mask
    val_mask = (timestep_t > train_cutoff) & (timestep_t <= val_cutoff) & labeled_mask
    test_mask = (timestep_t > val_cutoff) & labeled_mask

    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise ValueError("At least one split is empty. Adjust split ratios or verify data.")

    # Temporal graph contexts:
    # - Train graph sees only train-time nodes.
    # - Val graph sees train+val-time nodes.
    # - Test graph sees full graph.
    graph_train_nodes = timestep_t <= train_cutoff
    graph_val_nodes = timestep_t <= val_cutoff
    edge_index_train = _filter_edges_by_node_mask(edge_index, graph_train_nodes)
    edge_index_val = _filter_edges_by_node_mask(edge_index, graph_val_nodes)
    edge_index_test = edge_index

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    data.timestep = timestep_t
    data.tx_id = torch.from_numpy(tx_ids)
    data.labeled_mask = labeled_mask
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.edge_index_train = edge_index_train
    data.edge_index_val = edge_index_val
    data.edge_index_test = edge_index_test
    data.train_cutoff = torch.tensor(train_cutoff)
    data.val_cutoff = torch.tensor(val_cutoff)

    return data


class FraudGraphSAGE(nn.Module):
    """GraphSAGE model for binary node classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.35):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
        self.norms.append(BatchNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            self.norms.append(BatchNorm(hidden_dim))

        self.out_conv = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.out_norm = BatchNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return raw logits for binary fraud classification."""

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if h_in.shape == h.shape:
                h = h + h_in
            h = F.dropout(h, p=self.dropout, training=self.training)

        h_out = self.out_conv(h, edge_index)
        h_out = self.out_norm(h_out)
        h_out = F.relu(h_out)
        h_out = F.dropout(h_out, p=self.dropout, training=self.training)

        logits = self.classifier(h_out).squeeze(-1)
        return logits


@torch.no_grad()
def _scores_from_logits(logits: torch.Tensor, y_true: torch.Tensor, threshold: float) -> Dict[str, float]:
    """Compute fraud-relevant metrics for a labeled subset."""

    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y_true.cpu().numpy().astype(int)
    return _scores_from_probabilities(probs, y_np, threshold)


def _scores_from_probabilities(probs: np.ndarray, y_np: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute fraud-relevant metrics from predicted positive-class probabilities."""

    pred_np = (probs >= threshold).astype(int)

    pr_auc = average_precision_score(y_np, probs)
    macro_f1 = f1_score(y_np, pred_np, average="macro", zero_division=0)
    precision = precision_score(y_np, pred_np, zero_division=0)
    recall = recall_score(y_np, pred_np, zero_division=0)

    return {
        "pr_auc": float(pr_auc),
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
    }


@torch.no_grad()
def tune_threshold_by_macro_f1(logits: torch.Tensor, y_true: torch.Tensor) -> Tuple[float, float]:
    """Pick threshold that maximizes macro F1."""

    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y_true.cpu().numpy().astype(int)
    return tune_threshold_by_macro_f1_from_probabilities(probs, y_np)


def tune_threshold_by_macro_f1_from_probabilities(probs: np.ndarray, y_np: np.ndarray) -> Tuple[float, float]:
    """Select threshold that maximizes Macro F1 from class probabilities."""

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (probs >= threshold).astype(int)
        f1 = f1_score(y_np, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def print_metrics(tag: str, metrics: Dict[str, float], threshold: float | None = None) -> None:
    """Compact metric printout for terminal runs."""

    ordered = ["pr_auc", "macro_f1", "precision", "recall"]
    metric_text = " ".join(f"{k}={metrics[k]:.4f}" for k in ordered if k in metrics)
    if threshold is None:
        log(f"[{tag}] {metric_text}")
    else:
        log(f"[{tag}] {metric_text} thr={threshold:.2f}")


def train_model(data: Data, cfg: TrainConfig, device: torch.device) -> Tuple[FraudGraphSAGE, Dict[str, float], float]:
    """Train GraphSAGE and return best state + val threshold."""

    model = FraudGraphSAGE(
        in_dim=data.num_node_features,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    edge_index_train = data.edge_index_train.to(device)
    edge_index_val = data.edge_index_val.to(device)

    train_y = y[train_mask]
    pos_count = int((train_y == 1).sum().item())
    neg_count = int((train_y == 0).sum().item())
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Train split must contain both licit and illicit labels.")

    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    best_state = None
    best_val_metrics: Dict[str, float] = {}
    best_val_pr_auc = -1.0
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_logits = model(x, edge_index_train)
        loss = criterion(train_logits[train_mask], y[train_mask].float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        model.eval()
        val_logits = model(x, edge_index_val)
        tuned_threshold, _ = tune_threshold_by_macro_f1(val_logits[val_mask], y[val_mask])
        val_metrics = _scores_from_logits(val_logits[val_mask], y[val_mask], threshold=tuned_threshold)

        scheduler.step(val_metrics["pr_auc"])

        improved = val_metrics["pr_auc"] > best_val_pr_auc
        if improved:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_val_metrics = val_metrics
            best_threshold = tuned_threshold
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        should_log_epoch = (LOG_LEVEL >= 2) or (epoch == 1 or epoch % 10 == 0)
        if should_log_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            log(
                f"[train ep={epoch:03d}] loss={loss.item():.4f} "
                f"val_ap={val_metrics['pr_auc']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
                f"thr={tuned_threshold:.2f} lr={current_lr:.6f}",
                level=1,
            )

        if patience_counter >= cfg.early_stopping_patience:
            log(f"[train] early-stop ep={epoch} (no AP gain)")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    return model, best_val_metrics, best_threshold


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: Data,
    threshold: float,
    split: str,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on val or test split with chosen operating threshold."""

    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)

    if split == "val":
        logits = model(x, data.edge_index_val.to(device))
        mask = data.val_mask.to(device)
    elif split == "test":
        logits = model(x, data.edge_index_test.to(device))
        mask = data.test_mask.to(device)
    else:
        raise ValueError("split must be 'val' or 'test'")

    return _scores_from_logits(logits[mask], y[mask], threshold=threshold)


def train_xgboost_baseline(
    data: Data,
    cfg: BaselineConfig,
) -> Tuple[object, Dict[str, float], float]:
    """Train XGBoost baseline on node features only (IID baseline)."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the IID tabular baseline. "
            "Install it with: pip install xgboost"
        ) from exc

    x_np = data.x.cpu().numpy()
    y_np = data.y.cpu().numpy().astype(int)
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()

    x_train = x_np[train_mask]
    y_train = y_np[train_mask]
    x_val = x_np[val_mask]
    y_val = y_np[val_mask]

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Train split must contain both licit and illicit labels.")

    scale_pos_weight = neg_count / max(pos_count, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=cfg.seed,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(x_train, y_train, verbose=False)

    val_probs = model.predict_proba(x_val)[:, 1]
    threshold, _ = tune_threshold_by_macro_f1_from_probabilities(val_probs, y_val)
    val_metrics = _scores_from_probabilities(val_probs, y_val, threshold=threshold)
    return model, val_metrics, threshold


def evaluate_xgboost_baseline(
    model: object,
    data: Data,
    threshold: float,
    split: str,
) -> Dict[str, float]:
    """Evaluate the IID tabular baseline on val or test nodes."""

    x_np = data.x.cpu().numpy()
    y_np = data.y.cpu().numpy().astype(int)

    if split == "val":
        mask = data.val_mask.cpu().numpy()
    elif split == "test":
        mask = data.test_mask.cpu().numpy()
    else:
        raise ValueError("split must be 'val' or 'test'")

    x_split = x_np[mask]
    y_split = y_np[mask]
    probs = model.predict_proba(x_split)[:, 1]
    return _scores_from_probabilities(probs, y_split, threshold=threshold)


@torch.no_grad()
def _get_gnn_split_outputs(
    model: nn.Module,
    data: Data,
    split: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probabilities, labels, and timesteps for a split."""

    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    timestep = data.timestep.to(device)

    if split == "val":
        edge_index = data.edge_index_val.to(device)
        mask = data.val_mask.to(device)
    elif split == "test":
        edge_index = data.edge_index_test.to(device)
        mask = data.test_mask.to(device)
    else:
        raise ValueError("split must be 'val' or 'test'")

    logits = model(x, edge_index)
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    y_np = y[mask].detach().cpu().numpy().astype(int)
    timestep_np = timestep[mask].detach().cpu().numpy().astype(int)
    return probs, y_np, timestep_np


def _get_xgb_split_outputs(
    model: object,
    data: Data,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probabilities, labels, and timesteps for XGBoost split."""

    x_np = data.x.cpu().numpy()
    y_np = data.y.cpu().numpy().astype(int)
    t_np = data.timestep.cpu().numpy().astype(int)

    if split == "val":
        mask = data.val_mask.cpu().numpy()
    elif split == "test":
        mask = data.test_mask.cpu().numpy()
    else:
        raise ValueError("split must be 'val' or 'test'")

    probs = model.predict_proba(x_np[mask])[:, 1]
    return probs, y_np[mask], t_np[mask]


def _save_class_mix_overview(data: Data, output_dir: str) -> str:
    """Save global class-distribution overview PNG."""

    y_np = data.y.cpu().numpy().astype(int)
    counts = {
        "licit": int((y_np == 0).sum()),
        "illicit": int((y_np == 1).sum()),
        "unknown": int((y_np == -1).sum()),
    }
    labeled_total = max(counts["licit"] + counts["illicit"], 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    labels_all = ["licit", "illicit", "unknown"]
    vals_all = [counts[k] for k in labels_all]
    colors_all = ["steelblue", "tomato", "lightgray"]
    axes[0].bar(labels_all, vals_all, color=colors_all, edgecolor="black", alpha=0.85)
    axes[0].set_title("Global class mix (all nodes)")
    axes[0].set_ylabel("Node count")

    for idx, val in enumerate(vals_all):
        axes[0].text(idx, val, f"{val:,}", ha="center", va="bottom", fontsize=9)

    labels_lab = ["licit", "illicit"]
    vals_lab = [counts["licit"], counts["illicit"]]
    colors_lab = ["steelblue", "tomato"]
    axes[1].bar(labels_lab, vals_lab, color=colors_lab, edgecolor="black", alpha=0.85)
    axes[1].set_title("Labeled-only class mix")
    axes[1].set_ylabel("Node count")
    axes[1].text(
        0.64,
        0.98,
        (
            f"illicit={counts['illicit']}/{labeled_total} "
            f"({100.0 * counts['illicit'] / labeled_total:.2f}%)"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "gray", "alpha": 0.9},
    )

    for idx, val in enumerate(vals_lab):
        axes[1].text(idx, val, f"{val:,}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Elliptic Class Distribution Overview")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "class_mix_overview.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _save_pr_curve_comparison(
    y_true: np.ndarray,
    gnn_probs: np.ndarray,
    xgb_probs: np.ndarray,
    output_dir: str,
) -> str:
    """Save test-set PR-curve comparison PNG."""

    prec_gnn, rec_gnn, _ = precision_recall_curve(y_true, gnn_probs)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, xgb_probs)
    ap_gnn = average_precision_score(y_true, gnn_probs)
    ap_xgb = average_precision_score(y_true, xgb_probs)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.plot(rec_gnn, prec_gnn, color="firebrick", lw=2.2, label=f"GraphSAGE (AP={ap_gnn:.4f})")
    ax.plot(rec_xgb, prec_xgb, color="navy", lw=2.2, label=f"XGBoost IID (AP={ap_xgb:.4f})")
    ax.axhline(baseline, color="gray", ls="--", lw=1.4, label=f"Random baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Test PR Curve: GraphSAGE vs XGBoost")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "test_pr_curve_comparison.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _save_confusion_matrix_comparison(
    y_true: np.ndarray,
    gnn_probs: np.ndarray,
    gnn_threshold: float,
    xgb_probs: np.ndarray,
    xgb_threshold: float,
    output_dir: str,
) -> str:
    """Save side-by-side confusion matrices for test split."""

    gnn_pred = (gnn_probs >= gnn_threshold).astype(int)
    xgb_pred = (xgb_probs >= xgb_threshold).astype(int)

    cm_gnn = confusion_matrix(y_true, gnn_pred, labels=[0, 1])
    cm_xgb = confusion_matrix(y_true, xgb_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for ax, cm, title in [
        (axes[0], cm_gnn, f"GraphSAGE (thr={gnn_threshold:.2f})"),
        (axes[1], cm_xgb, f"XGBoost IID (thr={xgb_threshold:.2f})"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_rate = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=row_rate, where=row_sums != 0)

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({row_rate[i, j] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred licit", "Pred illicit"])
        ax.set_yticklabels(["True licit", "True illicit"])
        ax.set_title(title)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    fig.suptitle("Test Confusion Matrices (counts and row rates)")

    out_path = os.path.join(output_dir, "test_confusion_matrix_comparison.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _save_score_distribution_comparison(
    y_true: np.ndarray,
    gnn_probs: np.ndarray,
    xgb_probs: np.ndarray,
    output_dir: str,
) -> str:
    """Save score-distribution histograms for both models."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 26)

    for ax, probs, title in [
        (axes[0], gnn_probs, "GraphSAGE score distribution"),
        (axes[1], xgb_probs, "XGBoost IID score distribution"),
    ]:
        ax.hist(
            probs[y_true == 0],
            bins=bins,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            label="True licit",
        )
        ax.hist(
            probs[y_true == 1],
            bins=bins,
            alpha=0.7,
            color="tomato",
            edgecolor="black",
            label="True illicit",
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted illicit probability")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Node count")
    axes[0].legend(loc="upper center", fontsize=9)
    fig.suptitle("Test Probability Distributions by True Class")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "test_score_distribution_comparison.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _save_temporal_risk_trend(
    timesteps: np.ndarray,
    y_true: np.ndarray,
    gnn_probs: np.ndarray,
    xgb_probs: np.ndarray,
    output_dir: str,
) -> str:
    """Save timestep-level risk trend PNG for test split."""

    unique_steps = np.sort(np.unique(timesteps))
    gnn_mean = []
    xgb_mean = []
    actual_rate = []

    for t in unique_steps:
        mask = timesteps == t
        gnn_mean.append(float(np.mean(gnn_probs[mask])))
        xgb_mean.append(float(np.mean(xgb_probs[mask])))
        actual_rate.append(float(np.mean(y_true[mask])))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(unique_steps, gnn_mean, marker="o", lw=2.0, color="firebrick", label="GraphSAGE mean score")
    ax.plot(unique_steps, xgb_mean, marker="o", lw=2.0, color="navy", label="XGBoost mean score")
    ax.plot(unique_steps, actual_rate, marker="o", lw=2.0, color="darkgreen", label="Actual illicit rate")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Rate / Mean predicted illicit probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Test Temporal Risk Trend")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "test_temporal_risk_trend.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _save_calibration_comparison(
    y_true: np.ndarray,
    gnn_probs: np.ndarray,
    gnn_threshold: float,
    xgb_probs: np.ndarray,
    xgb_threshold: float,
    output_dir: str,
) -> str:
    """Save reliability/calibration comparison PNG for both models."""

    frac_gnn, mean_gnn = calibration_curve(y_true, gnn_probs, n_bins=10, strategy="quantile")
    frac_xgb, mean_xgb = calibration_curve(y_true, xgb_probs, n_bins=10, strategy="quantile")

    brier_gnn = brier_score_loss(y_true, gnn_probs)
    brier_xgb = brier_score_loss(y_true, xgb_probs)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.4, label="Perfect calibration")
    axes[0].plot(
        mean_gnn,
        frac_gnn,
        marker="o",
        lw=2.0,
        color="firebrick",
        label=f"GraphSAGE (Brier={brier_gnn:.4f})",
    )
    axes[0].plot(
        mean_xgb,
        frac_xgb,
        marker="o",
        lw=2.0,
        color="navy",
        label=f"XGBoost IID (Brier={brier_xgb:.4f})",
    )
    axes[0].axvline(gnn_threshold, color="firebrick", ls=":", lw=1.5, alpha=0.9)
    axes[0].axvline(xgb_threshold, color="navy", ls=":", lw=1.5, alpha=0.9)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Mean predicted illicit probability (bin)")
    axes[0].set_ylabel("Observed illicit frequency")
    axes[0].set_title("Reliability Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)

    bins = np.linspace(0, 1, 21)
    axes[1].hist(
        gnn_probs,
        bins=bins,
        alpha=0.65,
        color="firebrick",
        edgecolor="black",
        label="GraphSAGE scores",
    )
    axes[1].hist(
        xgb_probs,
        bins=bins,
        alpha=0.55,
        color="navy",
        edgecolor="black",
        label="XGBoost scores",
    )
    axes[1].axvline(gnn_threshold, color="firebrick", ls=":", lw=1.5, label=f"GNN thr={gnn_threshold:.2f}")
    axes[1].axvline(xgb_threshold, color="navy", ls=":", lw=1.5, label=f"XGB thr={xgb_threshold:.2f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Predicted illicit probability")
    axes[1].set_ylabel("Node count")
    axes[1].set_title("Score Histogram")
    axes[1].grid(alpha=0.2)
    axes[1].legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=8,
        framealpha=0.9,
    )

    fig.suptitle("Calibration + Score Reliability Overview")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "test_calibration_comparison.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _save_timestep_pr_auc_trend(
    timesteps: np.ndarray,
    y_true: np.ndarray,
    gnn_probs: np.ndarray,
    xgb_probs: np.ndarray,
    output_dir: str,
) -> str:
    """Save per-timestep PR-AUC trend line chart."""

    unique_steps = np.sort(np.unique(timesteps))
    gnn_ap = []
    xgb_ap = []
    illicit_rate = []
    sample_count = []

    for t in unique_steps:
        m = timesteps == t
        y_t = y_true[m]
        g_t = gnn_probs[m]
        x_t = xgb_probs[m]

        if np.unique(y_t).size < 2:
            gnn_ap.append(np.nan)
            xgb_ap.append(np.nan)
        else:
            gnn_ap.append(float(average_precision_score(y_t, g_t)))
            xgb_ap.append(float(average_precision_score(y_t, x_t)))

        illicit_rate.append(float(np.mean(y_t)))
        sample_count.append(int(m.sum()))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(unique_steps, gnn_ap, marker="o", lw=2.0, color="firebrick", label="GraphSAGE PR-AUC")
    ax_top.plot(unique_steps, xgb_ap, marker="o", lw=2.0, color="navy", label="XGBoost PR-AUC")
    ax_top.plot(
        unique_steps,
        illicit_rate,
        marker="o",
        lw=1.8,
        color="darkgreen",
        ls="--",
        label="Actual illicit rate",
    )
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("PR-AUC / illicit rate")
    ax_top.set_title("Per-Timestep Test PR-AUC Trend")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.text(
        0.99,
        0.82,
        "Gaps indicate timesteps where PR-AUC is undefined (single-class slice).",
        transform=ax_top.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "gray", "alpha": 0.9},
    )

    ax_bottom.bar(unique_steps, sample_count, color="slategray", alpha=0.8, edgecolor="black")
    ax_bottom.set_ylabel("Test nodes")
    ax_bottom.set_xlabel("Timestep")
    ax_bottom.grid(alpha=0.2)

    fig.tight_layout()

    out_path = os.path.join(output_dir, "test_timestep_pr_auc_trend.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def generate_result_visualizations(
    model: nn.Module,
    data: Data,
    gnn_threshold: float,
    xgb_model: object,
    xgb_threshold: float,
    output_dir: str,
    device: torch.device,
) -> List[str]:
    """Generate a bundle of model-result PNGs for reporting."""

    os.makedirs(output_dir, exist_ok=True)

    gnn_probs, y_test_gnn, t_test_gnn = _get_gnn_split_outputs(
        model=model,
        data=data,
        split="test",
        device=device,
    )
    xgb_probs, y_test_xgb, t_test_xgb = _get_xgb_split_outputs(
        model=xgb_model,
        data=data,
        split="test",
    )

    if len(y_test_gnn) != len(y_test_xgb) or not np.array_equal(y_test_gnn, y_test_xgb):
        raise RuntimeError("GNN and XGBoost test labels are not aligned for plotting.")
    if not np.array_equal(t_test_gnn, t_test_xgb):
        raise RuntimeError("GNN and XGBoost test timesteps are not aligned for plotting.")

    y_test = y_test_gnn
    t_test = t_test_gnn

    output_paths = [
        _save_class_mix_overview(data=data, output_dir=output_dir),
        _save_pr_curve_comparison(
            y_true=y_test,
            gnn_probs=gnn_probs,
            xgb_probs=xgb_probs,
            output_dir=output_dir,
        ),
        _save_confusion_matrix_comparison(
            y_true=y_test,
            gnn_probs=gnn_probs,
            gnn_threshold=gnn_threshold,
            xgb_probs=xgb_probs,
            xgb_threshold=xgb_threshold,
            output_dir=output_dir,
        ),
        _save_score_distribution_comparison(
            y_true=y_test,
            gnn_probs=gnn_probs,
            xgb_probs=xgb_probs,
            output_dir=output_dir,
        ),
        _save_temporal_risk_trend(
            timesteps=t_test,
            y_true=y_test,
            gnn_probs=gnn_probs,
            xgb_probs=xgb_probs,
            output_dir=output_dir,
        ),
        _save_calibration_comparison(
            y_true=y_test,
            gnn_probs=gnn_probs,
            gnn_threshold=gnn_threshold,
            xgb_probs=xgb_probs,
            xgb_threshold=xgb_threshold,
            output_dir=output_dir,
        ),
        _save_timestep_pr_auc_trend(
            timesteps=t_test,
            y_true=y_test,
            gnn_probs=gnn_probs,
            xgb_probs=xgb_probs,
            output_dir=output_dir,
        ),
    ]
    return output_paths


def print_metric_comparison(gnn_metrics: Dict[str, float], xgb_metrics: Dict[str, float]) -> None:
    """Print side-by-side fraud metrics for GNN and IID baseline."""

    metrics = ["pr_auc", "macro_f1", "precision", "recall"]
    log("\n[compare] test metrics (same split)")
    log(f"{'metric':<12}{'GraphSAGE':>12}{'XGBoost IID':>14}{'delta':>12}")
    for name in metrics:
        gnn_v = gnn_metrics[name]
        xgb_v = xgb_metrics[name]
        delta = gnn_v - xgb_v
        log(f"{name:<12}{gnn_v:>12.4f}{xgb_v:>14.4f}{delta:>12.4f}")


@torch.no_grad()
def pick_true_positive_illicit_node(
    model: nn.Module,
    data: Data,
    threshold: float,
    device: torch.device,
) -> Tuple[int, float]:
    """Pick a true-positive illicit test node (fallback: top illicit score)."""

    model.eval()
    logits = model(data.x.to(device), data.edge_index_test.to(device)).cpu()
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()

    test_mask = data.test_mask.cpu()
    tp_mask = test_mask & (data.y.cpu() == 1) & (preds == 1)
    tp_indices = tp_mask.nonzero(as_tuple=False).view(-1)

    if tp_indices.numel() > 0:
        best_idx = tp_indices[probs[tp_indices].argmax()].item()
        return int(best_idx), float(probs[best_idx].item())

    illicit_test_mask = test_mask & (data.y.cpu() == 1)
    illicit_indices = illicit_test_mask.nonzero(as_tuple=False).view(-1)
    if illicit_indices.numel() == 0:
        raise RuntimeError("No illicit-labeled nodes in the test split.")

    best_idx = illicit_indices[probs[illicit_indices].argmax()].item()
    adjusted_threshold = max(0.01, float(probs[best_idx].item()) - 1e-5)
    log(
        "[warn] no TP at current threshold; "
        f"fallback illicit prob={probs[best_idx].item():.4f} "
        f"adj_thr={adjusted_threshold:.4f}",
        level=0,
    )
    return int(best_idx), float(probs[best_idx].item())


def explain_and_visualize_node(
    model: nn.Module,
    data: Data,
    node_idx: int,
    output_path: str,
    device: torch.device,
    seed: int = 42,
    num_hops: int = 2,
    max_num_hops: int = 8,
    min_licit_nodes: int = 1,
    adaptive_hops: bool = True,
    explainer_epochs: int = 80,
    predicted_prob: float | None = None,
    decision_threshold: float | None = None,
) -> str:
    """Run GNNExplainer and plot a local explanation graph for one node."""

    model.eval()

    y_cpu = data.y.cpu()
    edge_index_cpu = data.edge_index_test.cpu()

    chosen_hops = num_hops
    chosen_subset = None
    chosen_sub_edge_index = None
    chosen_mapping = None
    chosen_licit_count = 0

    if adaptive_hops:
        for hops in range(num_hops, max_num_hops + 1):
            subset_tmp, sub_edge_index_tmp, mapping_tmp, _ = k_hop_subgraph(
                node_idx,
                num_hops=hops,
                edge_index=edge_index_cpu,
                relabel_nodes=True,
            )
            labels_tmp = y_cpu[subset_tmp]
            licit_count_tmp = int((labels_tmp == 0).sum().item())
            if licit_count_tmp >= min_licit_nodes:
                chosen_hops = hops
                chosen_subset = subset_tmp
                chosen_sub_edge_index = sub_edge_index_tmp
                chosen_mapping = mapping_tmp
                chosen_licit_count = licit_count_tmp
                break

        if chosen_subset is None:
            subset_tmp, sub_edge_index_tmp, mapping_tmp, _ = k_hop_subgraph(
                node_idx,
                num_hops=max_num_hops,
                edge_index=edge_index_cpu,
                relabel_nodes=True,
            )
            labels_tmp = y_cpu[subset_tmp]
            chosen_hops = max_num_hops
            chosen_subset = subset_tmp
            chosen_sub_edge_index = sub_edge_index_tmp
            chosen_mapping = mapping_tmp
            chosen_licit_count = int((labels_tmp == 0).sum().item())
            log(
                "[warn] adaptive hops did not find requested licit nodes "
                f"(min={min_licit_nodes}, max_hops={max_num_hops})",
                level=0,
            )
    else:
        subset_tmp, sub_edge_index_tmp, mapping_tmp, _ = k_hop_subgraph(
            node_idx,
            num_hops=num_hops,
            edge_index=edge_index_cpu,
            relabel_nodes=True,
        )
        labels_tmp = y_cpu[subset_tmp]
        chosen_hops = num_hops
        chosen_subset = subset_tmp
        chosen_sub_edge_index = sub_edge_index_tmp
        chosen_mapping = mapping_tmp
        chosen_licit_count = int((labels_tmp == 0).sum().item())

    subset = chosen_subset
    sub_edge_index = chosen_sub_edge_index
    mapping = chosen_mapping

    log(
        f"[explain] hops={chosen_hops} licit_nodes={chosen_licit_count}"
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explainer_epochs, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "binary_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    explanation = explainer(
        x=data.x[subset].to(device),
        edge_index=sub_edge_index.to(device),
        index=int(mapping.item()),
    )

    if explanation.edge_mask is None:
        raise RuntimeError("GNNExplainer did not return an edge mask.")

    local_edge_importance = explanation.edge_mask.detach().cpu().numpy()
    local_edge_importance = np.clip(local_edge_importance, a_min=0.0, a_max=None)

    graph = nx.DiGraph()
    tx_cpu = data.tx_id.cpu()

    for local_i, global_i in enumerate(subset.tolist()):
        label = int(y_cpu[global_i].item())
        txid = int(tx_cpu[global_i].item())
        graph.add_node(local_i, label=label, txid=txid)

    edges_local = sub_edge_index.t().tolist()
    for (src, dst), importance in zip(edges_local, local_edge_importance.tolist()):
        graph.add_edge(int(src), int(dst), importance=float(importance))

    ranked_edges: List[Tuple[int, int, float]] = []
    if graph.number_of_edges() > 0:
        ranked_edges = sorted(
            [(u, v, d["importance"]) for u, v, d in graph.edges(data=True)],
            key=lambda x: x[2],
            reverse=True,
        )

    center_local_idx = int(mapping.item())

    if local_edge_importance.size == 0:
        norm_imp = np.array([])
    else:
        max_imp = float(np.max(local_edge_importance))
        norm_imp = local_edge_importance / (max_imp + 1e-8)

    edge_widths = [1.0 + 5.0 * v for v in norm_imp.tolist()] if norm_imp.size > 0 else []
    edge_colors = [0.2 + 0.8 * v for v in norm_imp.tolist()] if norm_imp.size > 0 else []

    node_colors = []
    for n in graph.nodes():
        lbl = graph.nodes[n]["label"]
        if n == center_local_idx:
            node_colors.append("gold")
        elif lbl == 1:
            node_colors.append("tomato")
        elif lbl == 0:
            node_colors.append("steelblue")
        else:
            node_colors.append("lightgray")

    pos = nx.spring_layout(graph, seed=seed, k=0.9 / math.sqrt(max(graph.number_of_nodes(), 1)))

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=700,
        node_color=node_colors,
        linewidths=1.5,
        edgecolors="black",
    )

    if graph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            graph,
            pos,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Reds,
            arrows=True,
            arrowsize=16,
            alpha=0.9,
        )

    labels = {
        n: (
            f"TARGET\ntx:{graph.nodes[n]['txid']}"
            if n == center_local_idx
            else f"n{n}\ntx:{graph.nodes[n]['txid']}"
        )
        for n in graph.nodes()
    }

    ax = plt.gca()
    leftmost_node = min(graph.nodes(), key=lambda n: pos[n][0])
    for n, label in labels.items():
        x_coord, y_coord = pos[n]
        x_offset = 0
        horizontal_align = "center"
        if n == leftmost_node:
            # Keep the left-most label on the node's left side for readability.
            x_offset = -10
            horizontal_align = "right"

        ax.annotate(
            label,
            xy=(x_coord, y_coord),
            xytext=(x_offset, 12),
            textcoords="offset points",
            ha=horizontal_align,
            va="bottom",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    local_labels = [graph.nodes[n]["label"] for n in graph.nodes()]
    local_illicit = int(sum(lbl == 1 for lbl in local_labels))
    local_licit = int(sum(lbl == 0 for lbl in local_labels))
    local_unknown = int(sum(lbl == -1 for lbl in local_labels))

    y_all = data.y.cpu().numpy()
    global_illicit = int(np.sum(y_all == 1))
    global_licit = int(np.sum(y_all == 0))
    global_labeled = max(global_illicit + global_licit, 1)
    global_illicit_pct = 100.0 * global_illicit / global_labeled

    pred_line = "Predicted illicit score: n/a"
    decision_line = "Decision: n/a"
    if predicted_prob is not None:
        pred_line = f"Predicted illicit score: {predicted_prob:.3f}"
        if decision_threshold is not None:
            verdict = "FLAGGED illicit" if predicted_prob >= decision_threshold else "NOT flagged illicit"
            decision_line = f"Decision ({decision_threshold:.3f}): {verdict}"

    top_edge_line = "Top edge: n/a"
    if ranked_edges:
        u0, v0, score0 = ranked_edges[0]
        tx_u0 = graph.nodes[u0]["txid"]
        tx_v0 = graph.nodes[v0]["txid"]
        top_edge_line = f"Top edge: tx:{tx_u0} -> tx:{tx_v0} (imp={score0:.3f})"

    conclusion_lines = [
        "Key conclusions for this case",
        pred_line,
        decision_line,
        top_edge_line,
        (
            "Local mix: "
            f"illicit={local_illicit}, licit={local_licit}, unknown={local_unknown}"
        ),
        (
            "Global labeled mix: "
            f"illicit={global_illicit}/{global_labeled} ({global_illicit_pct:.1f}%)"
        ),
        "Caveat: local neighborhood mix does not represent whole dataset.",
    ]

    plt.gcf().text(
        0.99,
        0.93,
        "\n".join(conclusion_lines),
        ha="right",
        va="top",
        fontsize=9,
        linespacing=1.15,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "gray", "alpha": 0.9},
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Target node being explained", markerfacecolor="gold", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Illicit transaction node", markerfacecolor="tomato", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Licit transaction node", markerfacecolor="steelblue", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Unknown-label node", markerfacecolor="lightgray", markersize=10),
        Line2D([0], [0], color=plt.cm.Reds(0.85), lw=3, label="Thicker/darker edge = stronger influence"),
    ]
    plt.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )

    plt.title(
        "GNNExplainer Subgraph for Detected Illicit Node\n"
        "TARGET node is gold. Thicker/darker edges contribute more to illicit score"
    )
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()

    if graph.number_of_edges() > 0:
        log("[explain] top edges (strongest first):")
        for rank, (u, v, score) in enumerate(ranked_edges[:10], start=1):
            tx_u = graph.nodes[u]["txid"]
            tx_v = graph.nodes[v]["txid"]
            log(f"  {rank:02d}) {tx_u}->{tx_v} imp={score:.4f}")

    summary_path = os.path.join(os.path.dirname(output_path), "gnnexplainer_tp_summary.md")
    edges_csv_path = os.path.join(os.path.dirname(output_path), "gnnexplainer_top_edges.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Explanation Guide for Viewers\n\n")
        f.write("## What this chart is showing\n")
        f.write("This visualization explains why the model flagged one transaction as suspicious.\n")
        f.write("The gold node is the target transaction being explained.\n\n")

        f.write("## How to read the nodes\n")
        f.write("- Gold: target transaction under investigation\n")
        f.write("- Red: known illicit transaction\n")
        f.write("- Blue: known licit transaction\n")
        f.write("- Gray: unknown label transaction\n\n")

        f.write("## How to read the edges\n")
        f.write("- Thicker/darker edges had stronger influence on the model's illicit score\n")
        f.write("- Arrows indicate transaction direction\n\n")

        f.write("## Case details\n")
        f.write(f"- Target txId: {int(data.tx_id[node_idx].item())}\n")
        f.write(f"- Hops used for explanation: {chosen_hops}\n")
        f.write(f"- Licit nodes found in explanation neighborhood: {chosen_licit_count}\n")
        f.write(f"- Minimum licit nodes requested: {min_licit_nodes}\n")
        if predicted_prob is not None:
            f.write(f"- Predicted illicit probability: {predicted_prob:.4f}\n")
        if decision_threshold is not None:
            f.write(f"- Decision threshold used: {decision_threshold:.4f}\n")
        f.write("\n")

        f.write("## Key conclusions for this case\n")
        if predicted_prob is not None and decision_threshold is not None:
            verdict = "FLAGGED illicit" if predicted_prob >= decision_threshold else "NOT flagged illicit"
            f.write(f"- Decision at threshold: {verdict}\n")
        if ranked_edges:
            u0, v0, score0 = ranked_edges[0]
            tx_u0 = graph.nodes[u0]["txid"]
            tx_v0 = graph.nodes[v0]["txid"]
            f.write(f"- Strongest edge influence: tx:{tx_u0} -> tx:{tx_v0} (importance={score0:.4f})\n")

        local_labels_md = [graph.nodes[n]["label"] for n in graph.nodes()]
        local_illicit_md = int(sum(lbl == 1 for lbl in local_labels_md))
        local_licit_md = int(sum(lbl == 0 for lbl in local_labels_md))
        local_unknown_md = int(sum(lbl == -1 for lbl in local_labels_md))
        f.write(
            "- Local explanation mix: "
            f"illicit={local_illicit_md}, licit={local_licit_md}, unknown={local_unknown_md}\n"
        )

        global_illicit_md = int((data.y.cpu() == 1).sum().item())
        global_licit_md = int((data.y.cpu() == 0).sum().item())
        global_labeled_md = max(global_illicit_md + global_licit_md, 1)
        global_illicit_pct_md = 100.0 * global_illicit_md / global_labeled_md
        f.write(
            "- Global labeled mix (for context): "
            f"illicit={global_illicit_md}/{global_labeled_md} ({global_illicit_pct_md:.2f}%), "
            f"licit={global_licit_md}/{global_labeled_md} ({100.0 - global_illicit_pct_md:.2f}%)\n"
        )
        f.write("- Caveat: local neighborhood composition does not equal whole-dataset composition.\n\n")

        f.write("## Top influential edges\n")
        if ranked_edges:
            for rank, (u, v, score) in enumerate(ranked_edges[:10], start=1):
                tx_u = graph.nodes[u]["txid"]
                tx_v = graph.nodes[v]["txid"]
                f.write(f"{rank}. tx:{tx_u} -> tx:{tx_v} (importance={score:.4f})\n")
        else:
            f.write("No edges were present in the extracted local subgraph.\n")

    if ranked_edges:
        rows = []
        for rank, (u, v, score) in enumerate(ranked_edges[:10], start=1):
            rows.append(
                {
                    "rank": rank,
                    "src_txid": int(graph.nodes[u]["txid"]),
                    "dst_txid": int(graph.nodes[v]["txid"]),
                    "importance": float(score),
                }
            )
        pd.DataFrame(rows).to_csv(edges_csv_path, index=False)

    return summary_path


def business_impact_summary() -> str:
    """Short non-technical summary for README/console output."""

    return (
        "Business Impact (PM-friendly):\n"
        "1. Better fraud capture under class imbalance: PR-AUC optimization means we\n"
        "   rank suspicious entities more effectively than accuracy-focused models.\n"
        "2. Lower analyst waste: Macro F1 + threshold tuning helps balance precision\n"
        "   (fewer false alerts) and recall (more bad actors caught).\n"
        "3. Explainable flags: GNNExplainer surfaces the suspicious transaction\n"
        "   neighborhood behind each high-risk entity, improving investigator trust\n"
        "   and reducing policy appeal friction.\n"
        "4. Coordinated abuse detection: Graph modeling captures ring-like behavior\n"
        "   that IID tabular baselines miss, especially when fraudsters collaborate."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Elliptic fraud detection with GraphSAGE + GNNExplainer")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing Elliptic CSV files")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots/artifacts")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument(
        "--explainer-epochs",
        type=int,
        default=80,
        help="Optimization steps for GNNExplainer (lower is faster).",
    )
    parser.add_argument(
        "--adaptive-explain-hops",
        action="store_true",
        help="Expand explanation hops until enough licit nodes appear or max hops is reached.",
    )
    parser.add_argument(
        "--max-explain-hops",
        type=int,
        default=8,
        help="Maximum hop depth used when adaptive explanation hops are enabled.",
    )
    parser.add_argument(
        "--min-licit-nodes",
        type=int,
        default=1,
        help="Minimum number of licit nodes desired in explanation neighborhood when adaptive hops are enabled.",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip GNNExplainer step (useful for faster benchmark sweeps).",
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--quiet",
        action="store_true",
        help="Print warnings/errors only.",
    )
    verbosity_group.add_argument(
        "--verbose",
        action="store_true",
        help="Print more detailed training logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_log_level(quiet=args.quiet, verbose=args.verbose)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[run] device={device}")

    split_cfg = SplitConfig()
    train_cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
    )
    baseline_cfg = BaselineConfig(
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        seed=args.seed,
    )

    data = load_elliptic_as_pyg_data(args.data_dir, split_cfg)
    log(
        f"[data] nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"features={data.num_node_features}"
    )
    log(
        "[data] cutoffs "
        f"train<=t{int(data.train_cutoff.item())} "
        f"val<=t{int(data.val_cutoff.item())} "
        f"test>t{int(data.val_cutoff.item())}"
    )
    log(
        f"[data] labels train={int(data.train_mask.sum())} "
        f"val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}"
    )

    model, best_val_metrics, threshold = train_model(data, train_cfg, device)
    print_metrics("gnn val", best_val_metrics, threshold)

    test_metrics = evaluate_model(model, data, threshold=threshold, split="test", device=device)
    print_metrics("gnn test", test_metrics)

    xgb_model, xgb_val_metrics, xgb_threshold = train_xgboost_baseline(data, baseline_cfg)
    print_metrics("xgb val", xgb_val_metrics, xgb_threshold)

    xgb_test_metrics = evaluate_xgboost_baseline(
        model=xgb_model,
        data=data,
        threshold=xgb_threshold,
        split="test",
    )
    print_metrics("xgb test", xgb_test_metrics)

    print_metric_comparison(test_metrics, xgb_test_metrics)

    extra_pngs = generate_result_visualizations(
        model=model,
        data=data,
        gnn_threshold=threshold,
        xgb_model=xgb_model,
        xgb_threshold=xgb_threshold,
        output_dir=args.output_dir,
        device=device,
    )
    log("[out] extra visualizations:")
    for path in extra_pngs:
        log(f"  - {path}")

    if args.skip_explainability:
        log("[explain] skipped")
    else:
        tp_node_idx, tp_prob = pick_true_positive_illicit_node(model, data, threshold, device)
        tp_txid = int(data.tx_id[tp_node_idx].item())
        log(f"[explain] node_idx={tp_node_idx} txid={tp_txid} prob={tp_prob:.4f}")

        fig_path = os.path.join(args.output_dir, "gnnexplainer_tp_subgraph.png")
        summary_path = explain_and_visualize_node(
            model=model,
            data=data,
            node_idx=tp_node_idx,
            output_path=fig_path,
            device=device,
            seed=args.seed,
            num_hops=2,
            max_num_hops=args.max_explain_hops,
            min_licit_nodes=args.min_licit_nodes,
            adaptive_hops=args.adaptive_explain_hops,
            explainer_epochs=args.explainer_epochs,
            predicted_prob=tp_prob,
            decision_threshold=threshold,
        )
        log(f"[out] figure={fig_path}")
        log(f"[out] guide={summary_path}")

    log("[done] run complete", level=0)


if __name__ == "__main__":
    main()
