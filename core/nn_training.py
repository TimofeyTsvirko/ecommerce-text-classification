import copy
from time import perf_counter
from typing import Any, Union, Sequence, Optional, Callable, MutableMapping

import numpy as np
import torch
from tqdm import tqdm

import core.visualization as p

from sklearn.model_selection import StratifiedKFold
from sklearn import clone

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.baseline_training import evaluate_classification, aggregate_classification_cv_metrics
from core.training_results import ClassificationMetrics
from core.utils import get_device, free_memory


def _safe_index(data: Any, indices: Union[Sequence[int], np.ndarray]) -> Any:
    if hasattr(data, 'iloc'):
        return data.iloc[indices]
    try:
        return data.iloc[indices]
    except Exception:
        return np.asarray(data)[indices]


def _labels_from_score(y_score: np.ndarray, threshold: float) -> np.ndarray:
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return (y_score.ravel() >= threshold).astype(int)
    return np.argmax(y_score, axis=1)


def _positive_class_probabilities(y_score: np.ndarray) -> Optional[np.ndarray]:
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return y_score.ravel()
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    return None


def cross_validate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, *,
                         cv, criterion: nn.Module, optimizer_class: type,
                         optimizer_params: dict,
                         preprocessor: Optional[Any] = None,
                         num_epochs: int = 10, batch_size: int = 32,
                         device: Optional[str] = None,
                         enable_plot: Optional[bool] = False,
                         model_name: Optional[str] = "PyTorch CV") -> ClassificationMetrics:
    device = device or get_device()

    fit_times, accs, f1s, precs, recalls, aucs = [], [], [], [], [], []
    estimators = []

    n_samples = len(y)
    num_classes = model.fc.out_features

    oof_pred = np.empty(n_samples, dtype=int)
    oof_probs = np.empty((n_samples, num_classes), dtype=float)

    X = np.array(X)
    y = np.array(y)

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"\n🌀 Fold {fold_index + 1}/{cv.get_n_splits()}")
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        fitted_preprocessor = clone(preprocessor) if preprocessor is not None else None
        if fitted_preprocessor is not None:
            X_train = fitted_preprocessor.fit_transform(X_train)
            X_test = fitted_preprocessor.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        fold_model = copy.deepcopy(model).to(device)
        optimizer = optimizer_class(fold_model.parameters(), **optimizer_params)

        fold_model.train()
        start_fit = perf_counter()

        for epoch in tqdm(range(num_epochs), desc=f"Training fold {fold_index + 1}", leave=True):
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = fold_model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            tqdm.write(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

        fit_times.append(perf_counter() - start_fit)

        fold_model.eval()
        with torch.no_grad():
            logits = fold_model(X_test_tensor)                      # (n_test, num_classes)
            probs = torch.softmax(logits, dim=1).cpu().numpy()      # вероятности
            y_pred = probs.argmax(axis=1)                           # классы

        # Метрики фолда
        fold_metrics = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred,
            y_probs=probs,
            model_name=f"fold_{fold_index + 1}",
            enable_plot=enable_plot
        )

        accs.append(fold_metrics.accuracy)
        f1s.append(fold_metrics.f1_score)
        precs.append(fold_metrics.precision)
        recalls.append(fold_metrics.recall)
        aucs.append(fold_metrics.roc_auc if fold_metrics.roc_auc is not None else float("nan"))

        estimators.append({
            "preprocessor": fitted_preprocessor,
            "model": fold_model
        })

        oof_pred[test_index] = y_pred
        oof_probs[test_index] = probs

        del X_train_tensor, y_train_tensor, X_test_tensor, train_loader, optimizer, logits
        free_memory()

    final_metrics = aggregate_classification_cv_metrics(
        accuracy=float(np.nanmean(accs)),
        precision=float(np.nanmean(precs)),
        recall=float(np.nanmean(recalls)),
        f1_score_value=float(np.nanmean(f1s)),
        roc_auc=float(np.nanmean(aucs)),
        training_time=float(np.nansum(fit_times)),
        name=model_name,
        y_true=y,
        y_pred=oof_pred,
        y_probs=oof_probs
    )

    final_metrics.estimators = estimators
    p.plot_classification_results(final_metrics, model_name=model_name)

    return final_metrics


def fine_tune_and_validate(model: nn.Module,
                X, y,
                *,
                criterion: nn.Module,
                optimizer_class: type,
                optimizer_params: dict,
                preprocessor=None,
                num_epochs: int = 5,
                batch_size: int = 32,
                device: str | None = None,
                enable_plot: bool = True,
                model_name: str = "PyTorch Model"):

    model = model.to(device)

    X = preprocessor.transform(X)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    fit_times = []
    model.train()

    start_fit = perf_counter()
    for epoch in tqdm(range(num_epochs), desc="Training", leave=True):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")
    fit_times.append(perf_counter() - start_fit)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        y_score = torch.sigmoid(logits).cpu().numpy().flatten()

    y_pred = _labels_from_score(y_score, 0.5)
    y_probs = _positive_class_probabilities(y_score)

    metrics = evaluate_classification(
        y_true=y,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model_name,
        enable_plot=enable_plot
    )

    final_metrics = aggregate_classification_cv_metrics(
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score_value=metrics.f1_score,
        roc_auc=metrics.roc_auc if metrics.roc_auc is not None else float("nan"),
        training_time=float(np.nansum(fit_times)),
        name=model_name,
        y_true=y,
        y_pred=y_pred,
        y_probs=y_probs
    )

    p.plot_classification_results(final_metrics, model_name="Fine-tuned model")

    return final_metrics



@torch.no_grad()
def predict_with_ensemble(ensemble, X, device, threshold=0.5, batch_size=128):
    all_probs = []

    for start in tqdm(range(0, len(X), batch_size), desc="Predicting ensemble"):
        end = start + batch_size
        X_batch_raw = X[start:end]

        fold_probs = []

        for estimator in ensemble:
            preprocessor = estimator["preprocessor"]
            model = estimator["model"]

            X_batch = preprocessor.transform(X_batch_raw)

            X_batch = torch.tensor(X_batch, dtype=torch.long, device=device)

            model.eval()
            probs = torch.sigmoid(model(X_batch)).detach().cpu().numpy().flatten()
            fold_probs.append(probs)

        mean_probs = np.mean(fold_probs, axis=0)
        all_probs.append(mean_probs)

    all_probs = np.concatenate(all_probs)
    predictions = (all_probs >= threshold).astype(int)
    return predictions, all_probs
