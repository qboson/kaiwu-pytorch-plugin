from .logging import get_logger
logger = get_logger("evaluate")

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)

def compute_threshold(
    free_energy_func,  # 改为可调用对象
    X_val_tensor: torch.Tensor, 
    percentile: float = 95, 
    device=None
) -> float:
    """
    Compute a free‐energy threshold from validation normal data.
    """
    X_val_tensor = X_val_tensor.to(device)
    fe_val = free_energy_func(X_val_tensor).cpu().numpy()  # 调用自由能函数
    threshold = np.percentile(fe_val, percentile)
    return threshold, fe_val.mean(), fe_val.std()

def evaluate_rbm(
    free_energy_func,  # 改为可调用对象
    X_test_tensor: torch.Tensor,
    y_test: torch.Tensor,
    threshold: float,
    logger=None,
) -> dict:
    """
    Evaluate the RBM on test data using free energy as anomaly score.
    Returns a dictionary with all metrics.
    """
    if logger is None:
        logger = globals().get("logger")

    fe_test = free_energy_func(X_test_tensor).cpu().numpy()

    y_test_np = y_test.numpy()
    pred = (fe_test > threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test_np, pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # ROC
    fpr, tpr, _ = roc_curve(y_test_np, fe_test)
    roc_auc = auc(fpr, tpr)

    # PR
    pr_auc = average_precision_score(y_test_np, fe_test)

    metrics = {
        "confusion_matrix": cm.tolist(),          # 转为 list
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fe_test": fe_test.tolist(),
        "y_pred": pred.tolist(),
    }

    # logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return metrics

@torch.no_grad()
def compute_anomaly_scores(free_energy_func, data_loader, device):
    """Compute free energy scores for all samples."""
    scores = []
    labels = []
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        fe = free_energy_func(X_batch).cpu().numpy()
        scores.append(fe)
        labels.append(y_batch.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return scores, labels

def optimize_threshold_rbm(
    free_energy_func, val_loader, device, 
    initial_threshold=None, step=0.001, decay=0.5, 
    num_decay=30, occ=10, metric='f1',
    search_mode='grid', n_candidates=200,
    direction='higher_is_anomaly'
):
    """
    在验证集上优化阈值，最大化指定指标（'f1' 或 'accuracy'）
    返回最佳阈值

    direction:
        'higher_is_anomaly' → pred = (score >  th) 判为异常
        'lower_is_anomaly'  → pred = (score <  th) 判为异常
    """
    if direction not in ('higher_is_anomaly', 'lower_is_anomaly'):
        raise ValueError("direction must be 'higher_is_anomaly' or 'lower_is_anomaly'")
    # 预先计算所有验证样本的分数和标签
    scores, labels = compute_anomaly_scores(free_energy_func, val_loader, device)
    
    if search_mode == 'grid':
        # 自动网格搜索（推荐用于调优）
        import time
        t0 = time.time()
        q5, q95 = np.percentile(scores, [5, 95])
        # 如果分数范围太小，适当扩展
        if q95 - q5 < 1e-6:
            q5 = np.min(scores) - 0.1
            q95 = np.max(scores) + 0.1
        candidate_thresholds = np.linspace(q5, q95, n_candidates)
        logger.info("[grid] start: n=%d candidates in [%.4f, %.4f], metric=%s, n_val=%d, direction=%s",
                    n_candidates, q5, q95, metric, len(labels), direction)
        best_metric = 0.0
        best_threshold = q5
        n_updates = 0
        for idx, th in enumerate(candidate_thresholds):
            pred = ((scores > th) if direction == 'higher_is_anomaly'
                    else (scores < th)).astype(int)
            if metric == 'f1':
                cur_metric = f1_score(labels, pred, zero_division=0)
            elif metric == 'accuracy':
                cur_metric = accuracy_score(labels, pred)
            else:
                raise ValueError("metric must be 'f1' or 'accuracy'")

            is_new_best = cur_metric > best_metric
            logger.info("cand #%3d/%d th=%.6f | %s: %.4f%s",
                        idx + 1, n_candidates, th, metric.upper(), cur_metric,
                        "  *new best*" if is_new_best else "")

            if is_new_best:
                best_metric = cur_metric
                best_threshold = th
                n_updates += 1
        dt = time.time() - t0
        logger.info("[grid] evaluated %d candidates in %.2fs (best updated %d times)",
                    n_candidates, dt, n_updates)
        logger.info("Grid search done. Best threshold: %.6f, best %s: %.4f",
                    best_threshold, metric, best_metric)
        return best_threshold

    else:  # 原有逐步搜索 (stepwise)
        if initial_threshold is None:
            # 若未提供初始阈值，自动取95百分位
            initial_threshold = np.percentile(scores, 95)
        best_threshold = initial_threshold
        best_metric = 0.0
        cur_threshold = initial_threshold
        no_improve_count = 0

        for decay_step in range(num_decay):
            for _ in range(1000):
                # 当前阈值下的预测
                pred = ((scores > cur_threshold) if direction == 'higher_is_anomaly'
                        else (scores < cur_threshold)).astype(int)
                if metric == 'f1':
                    cur_metric = f1_score(labels, pred, zero_division=0)
                elif metric == 'accuracy':
                    cur_metric = accuracy_score(labels, pred)
                else:
                    raise ValueError("metric must be 'f1' or 'accuracy'")

                logger.debug("Threshold: %.6f | %s: %.4f", cur_threshold, metric.upper(), cur_metric)

                if cur_metric > best_metric:
                    best_metric = cur_metric
                    best_threshold = cur_threshold
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= occ:
                    # 步长衰减，重置阈值到最佳值
                    step *= decay
                    cur_threshold = best_threshold
                    logger.debug("Step decayed to %.6f, reset to %.6f", step, best_threshold)
                    break

                cur_threshold += step

            if decay_step == num_decay - 1:
                logger.info("Optimization finished. Best threshold: %.6f | Best %s: %.4f", best_threshold, metric.upper(), best_metric)
                return best_threshold

        return best_threshold
