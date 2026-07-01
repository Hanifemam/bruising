from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class CNNTrainer:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(config.log_dir) if config.log_dir else None
        self.model.to(self.device)

    def fit(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = self._loss(train_loader)
        best_val_acc = -1.0
        no_improve = 0
        patience = getattr(self.config, "early_stop_patience", 0)
        history = []
        try:
            for epoch in tqdm(range(1, self.config.epochs + 1), desc="Epochs"):
                train_loss, train_acc = self._run_epoch(train_loader, criterion, optimizer)
                val_loss, val_acc = self._run_epoch(val_loader, criterion)
                history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improve = 0
                    self._save_best(epoch, val_loss, val_acc)
                else:
                    no_improve += 1
                self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
                tqdm.write(f"epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
                if patience and no_improve >= patience:
                    tqdm.write(f"early stopping at epoch {epoch:03d}; best_val_acc={best_val_acc:.3f}")
                    break
        finally:
            if self.writer is not None:
                self.writer.close()
        return pd.DataFrame(history)

    def predict(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Predicting"):
                preds.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(preds), np.concatenate(labels)

    def log_model_graph(self, input_shape):
        if self.writer is None or not self.config.log_graph:
            return
        height, width, channels = input_shape
        example = torch.zeros(1, channels, height, width, device=self.device)
        try:
            self.writer.add_graph(self.model, example)
            self.writer.flush()
        except Exception as exc:
            tqdm.write(f"TensorBoard graph logging skipped: {exc}")

    def _loss(self, train_loader):
        if not self.config.use_class_weights:
            return nn.CrossEntropyLoss()
        counts = torch.bincount(train_loader.dataset.y, minlength=2).float()
        return nn.CrossEntropyLoss(weight=(counts.sum() / counts.clamp_min(1)).to(self.device))

    def _run_epoch(self, loader, criterion, optimizer=None):
        self.model.train(optimizer is not None)
        total_loss = total_correct = total = 0
        for x, y in tqdm(loader, desc="Train" if optimizer else "Eval", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            with torch.set_grad_enabled(optimizer is not None):
                logits = self.model(x)
                loss = criterion(logits, y)
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return total_loss / total, total_correct / total

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        if self.writer is None:
            return
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)
        self.writer.flush()

    def _save_best(self, epoch, val_loss, val_acc):
        path = Path(getattr(self.config, "best_model_path", "best_model.pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_state_dict": self.model.state_dict(),
        }, path)


def summarize_predictions(split, y_true, y_pred, meta):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    summary = {
        "split": split,
        "n": int(len(y_true)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "n_true_positive_samples": int((y_true == 1).sum()),
        "n_predicted_positive_samples": int((y_pred == 1).sum()),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }
    rows = []
    scored = meta.copy()
    scored["y_true"] = y_true
    scored["y_pred"] = y_pred
    for tp, g in scored.groupby("timepoint", sort=False):
        p, r, f1, _ = precision_recall_fscore_support(g["y_true"], g["y_pred"], average="binary", zero_division=0)
        tn = int(((g["y_true"] == 0) & (g["y_pred"] == 0)).sum())
        fp = int(((g["y_true"] == 0) & (g["y_pred"] == 1)).sum())
        fn = int(((g["y_true"] == 1) & (g["y_pred"] == 0)).sum())
        tp_count = int(((g["y_true"] == 1) & (g["y_pred"] == 1)).sum())
        rows.append({
            "split": split,
            "timepoint": tp,
            "n": len(g),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp_count,
            "n_true_positive_samples": int((g["y_true"] == 1).sum()),
            "n_predicted_positive_samples": int((g["y_pred"] == 1).sum()),
            "accuracy": round(accuracy_score(g["y_true"], g["y_pred"]), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })
    return summary, pd.DataFrame(rows)
