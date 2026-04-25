"""
train_fnn.py — Entrenamiento e inferencia FNN (PyTorch).
Carga parquets de feature_engineering.py, entrena, guarda modelo, predice.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    FEATURE_COLUMNS,
    TIPO_CICLO_COL,
    TIPO_CICLO_CATEGORIES,
    FEATURES_INFERENCE_FILE,
    FEATURES_TRAIN_FILE,
    MODEL_FILE,
    MODEL_META_FILE,
)


def _build_features(df: pd.DataFrame, feat_cols_base: list) -> tuple[np.ndarray, list]:
    """
    Construye la matriz X aplicando one-hot encoding de tipo_ciclo_b.
    Retorna (X_array, lista_de_columnas_finales).
    Usa categorías fijas para que train e inference sean idénticos.
    """
    X_base = df[feat_cols_base].copy()

    if TIPO_CICLO_COL in df.columns:
        dummies = pd.get_dummies(
            pd.Categorical(df[TIPO_CICLO_COL], categories=TIPO_CICLO_CATEGORIES),
            prefix="tipo",
            drop_first=True,
        )
        dummies.index = X_base.index
        X_all = pd.concat([X_base, dummies], axis=1)
    else:
        X_all = X_base

    final_cols = list(X_all.columns)
    X = np.nan_to_num(X_all.values.astype(np.float32), nan=0.0)
    return X, final_cols


# ── Modelo ────────────────────────────────────────────────────────────────────

class PurchaseFNN(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Evaluación Top-K ──────────────────────────────────────────────────────────

def evaluate_topk(model, df_eval: pd.DataFrame, feat_cols: list, top_k: int = 3):
    """
    Evalúa el modelo agrupando por familia (nucleo):
      - Precision@K: de las top K predicciones, cuántas tienen target=1
      - Hit Rate@K: al menos 1 de las top K tiene target=1
      - Recall@K: de todos los target=1, cuántos están en top K
    """
    X = np.nan_to_num(df_eval[feat_cols].values.astype(np.float32), nan=0.0)
    model.eval()
    with torch.no_grad():
        probas = torch.sigmoid(model(torch.tensor(X))).numpy()

    df_eval = df_eval.copy()
    df_eval["proba"] = probas

    precisions, recalls, hit_rates = [], [], []

    for nucleo, grp in df_eval.groupby("nucleo"):
        if len(grp) < top_k:
            continue
        top_items = grp.nlargest(top_k, "proba")["COD_SUBCATEGORIA"].values
        reales = grp.loc[grp["target"] == 1, "COD_SUBCATEGORIA"].values
        if len(reales) == 0:
            continue
        n_hit = len(set(top_items) & set(reales))
        precisions.append(n_hit / top_k)
        recalls.append(n_hit / len(reales))
        hit_rates.append(1.0 if n_hit > 0 else 0.0)

    n_eval = len(precisions)
    if n_eval == 0:
        return {"precision@k": 0, "recall@k": 0, "hit_rate@k": 0, "n_families": 0}

    results = {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "hit_rate@k": float(np.mean(hit_rates)),
        "n_families": n_eval,
    }
    return results


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def run_training(features_path=None, epochs=50, lr=1e-3, test_size=0.2):
    """Carga features_train.parquet → entrena FNN → guarda modelo + meta."""
    if features_path is None:
        features_path = FEATURES_TRAIN_FILE

    df = pd.read_parquet(features_path)
    feat_cols_base = [c for c in FEATURE_COLUMNS if c in df.columns]
    X, feat_cols = _build_features(df, feat_cols_base)
    print(f"[Train] {len(df)} filas | {df['nucleo'].nunique()} familias | {len(feat_cols)} features")
    print(f"[Train] Target: {int(df['target'].sum())} positivos / {len(df)} ({df['target'].mean()*100:.1f}%)")

    y = df["target"].values.astype(np.float32)

    # Split train/val (guardamos índices para evaluación top-k después)
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    s = int(len(X) * (1 - test_size))
    train_idx, val_idx = idx[:s], idx[s:]
    X_tr, X_val = torch.tensor(X[train_idx]), torch.tensor(X[val_idx])
    y_tr, y_val = torch.tensor(y[train_idx]), torch.tensor(y[val_idx])

    # Peso para desbalance
    n_pos = y_tr.sum().item()
    pw = (len(y_tr) - n_pos) / max(n_pos, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PurchaseFNN(input_dim=len(feat_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        logits = model(X_tr.to(device))
        loss = criterion(logits, y_tr.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val.to(device))
            v_loss = criterion(v_logits, y_val.to(device)).item()
            v_acc = ((torch.sigmoid(v_logits) >= 0.5).float() == y_val.to(device)).float().mean().item()

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={loss.item():.4f} | val_loss={v_loss:.4f} | val_acc={v_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"[Train] {time.time()-t0:.1f}s | best_val_loss={best_loss:.4f}")

    # ── Evaluación Top-K sobre validación ────────────────────────────────
    df_val = df.iloc[val_idx].copy()
    # re-construir X_val con one-hot para evaluate_topk
    X_val_full, _ = _build_features(df_val, feat_cols_base)
    df_val_aug = df_val.copy()
    for i, col in enumerate(feat_cols):
        df_val_aug[col] = X_val_full[:, i]
    topk_metrics = evaluate_topk(model, df_val_aug, feat_cols, top_k=3)
    print(f"[Eval]  Familias evaluadas: {topk_metrics['n_families']}")
    print(f"[Eval]  Precision@3: {topk_metrics['precision@k']:.4f} ({topk_metrics['precision@k']*100:.1f}%)")
    print(f"[Eval]  Hit Rate@3:  {topk_metrics['hit_rate@k']:.4f} ({topk_metrics['hit_rate@k']*100:.1f}%)")
    print(f"[Eval]  Recall@3:    {topk_metrics['recall@k']:.4f} ({topk_metrics['recall@k']*100:.1f}%)")

    # Guardar
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "feature_columns": feat_cols,
        "input_dim": len(feat_cols),
        "best_val_loss": best_loss,
        "epochs": epochs,
        "precision@3": topk_metrics["precision@k"],
        "hit_rate@3": topk_metrics["hit_rate@k"],
        "recall@3": topk_metrics["recall@k"],
        "n_families_eval": topk_metrics["n_families"],
    }
    with open(MODEL_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Train] Guardado: {MODEL_FILE}")
    return model, meta


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(features_path=None, output_path=None):
    """Carga modelo + features_inference.parquet → predice probabilidades."""
    if features_path is None:
        features_path = FEATURES_INFERENCE_FILE

    with open(MODEL_META_FILE) as f:
        meta = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PurchaseFNN(input_dim=meta["input_dim"]).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_parquet(features_path)
    feat_cols_base = [c for c in FEATURE_COLUMNS if c in df.columns]
    X, _ = _build_features(df, feat_cols_base)
    feat_cols = meta["feature_columns"]  # columnas guardadas en training (incluye one-hot)

    with torch.no_grad():
        probas = torch.sigmoid(model(torch.tensor(X).to(device))).cpu().numpy()

    df["proba_compra"] = probas
    df = df.sort_values("proba_compra", ascending=False)

    if output_path is None:
        output_path = FEATURES_INFERENCE_FILE.parent / "predictions.parquet"
    df.to_parquet(output_path, index=False)

    print(f"[Inference] {len(df)} filas → {output_path}")
    show = [c for c in ["nucleo", "COD_SUBCATEGORIA", "proba_compra"] if c in df.columns]
    print(df[show].head(10).to_string(index=False))
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "inference"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--features-path", type=str, default=None)
    args = p.parse_args()

    fp = Path(args.features_path) if args.features_path else None
    if args.action == "train":
        run_training(features_path=fp, epochs=args.epochs, lr=args.lr)
    else:
        run_inference(features_path=fp)
