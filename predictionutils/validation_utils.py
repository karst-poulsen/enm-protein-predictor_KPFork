import pandas as pd
import sklearn
from typing import Union

from . import data_utils


class ValidationUtils(object):
    """
    Convenience tool for calculating model metrics.

    Args:
        labels: Union[pd.DataFrame, pd.Series] | dataframe or series holding actual labels
        predictions: Union[pd.DataFrame, pd.Series] | dataframe or series holding model probability predictions
    """
    def __init__(self, labels: Union[pd.DataFrame, pd.Series], predictions: Union[pd.DataFrame, pd.Series]):
        self.LABELS = labels
        self.PREDICTIONS = predictions

    def well_rounded_validation(self, probability_threshold: float) -> dict:
        """
        Calculate AUROC, Recall, Precision, F1 Score, Accuracy, and confusion matrix for model predictions.

        Args:
            probability_threshold: float | threshold for classifying model predictions into labels

        Returns: dict | dictionary holding metrics

        """
        d = data_utils.DataUtils()
        classified_predictions = d.classify(self.PREDICTIONS, probability_threshold)
        conf_matrix = sklearn.metrics.confusion_matrix(self.LABELS, classified_predictions, labels=None)

        return {
                "AUROC": float(sklearn.metrics.roc_auc_score(self.LABELS, self.PREDICTIONS)),
                "Recall": float(sklearn.metrics.recall_score(self.LABELS, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1]),
                "Precision": float(sklearn.metrics.precision_score(self.LABELS, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1]),
                "F1 Score": float(sklearn.metrics.f1_score(self.LABELS, classified_predictions)),
                "Accuracy": float(sklearn.metrics.accuracy_score(self.LABELS, classified_predictions)),
                "Confusion Matrix": [int(conf_matrix[0][0]), int(conf_matrix[0][1]), int(conf_matrix[1][0]), int(conf_matrix[1][1])]
                }
