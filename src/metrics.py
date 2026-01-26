import numpy as np
from sklearn.metrics import roc_auc_score

def roc_auc_manual(y_true, y_pred):
    """
    Простая реализация ROC AUC
    """

    desc_order = np.argsort(-y_pred)
    y_true_sorted = np.array(y_true)[desc_order]

    n_pos = np.sum(y_true_sorted)
    n_neg = len(y_true_sorted) - n_pos

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    tpr = tp / n_pos
    fpr = fp / n_neg

    tpr = np.insert(tpr, 0, 0)
    fpr = np.insert(fpr, 0, 0)

    auc = np.trapz(tpr, fpr)
    return auc

def gini_manual(y_true, y_pred):
    return 2 * roc_auc_manual(y_true, y_pred) - 1

def gini_score(y_true, y_pred_proba):
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1

def precision_recall_curve_manual(y_true, y_pred):
    """
    Возвращает precision, recall для разных порогов
    """
    desc_order = np.argsort(-y_pred)
    y_true_sorted = np.array(y_true)[desc_order]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true_sorted)

    precision = np.insert(precision, 0, 1)
    recall = np.insert(recall, 0, 0)

    return precision, recall

def pr_auc_manual(y_true, y_pred):
    precision, recall = precision_recall_curve_manual(y_true, y_pred)
    return np.trapz(precision, recall)

def recall_score(y_true, y_pred):
    """
    Recall = TP / (TP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def precision_score(y_true, y_pred):
    """
    Precision = TP / (TP + FP)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def f1_score(y_true, y_pred):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def precision_recall_curve(y_true, y_score):
    """
    Строит точки Precision-Recall кривой вручную.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # сортируем по убыванию вероятности
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    tp = 0
    fp = 0
    fn = np.sum(y_true == 1)

    precision = []
    recall = []

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precision.append(p)
        recall.append(r)

    return np.array(precision), np.array(recall)

def auc_pr_score(y_true, y_score):
    """
    AUC PR = ∫ Precision d(Recall)
    """
    precision, recall = precision_recall_curve(y_true, y_score)

    # Recall должен быть по возрастанию
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    # Интегрирование трапецией
    auc = np.trapz(precision, recall)
    return auc
