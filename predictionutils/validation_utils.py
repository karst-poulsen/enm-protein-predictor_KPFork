import math
import numpy as np
from numpy import array_equal
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import column_or_1d

import visualization_utils
import warnings
import sklearn

import data_utils

class validation_metrics(object):
    """Several statistical tests used to validate the models predictive power

    Args:
        :param true_results (array of ints): The true testing values
        :param predicted_results (array of floats): The predicted probabilities
        generated by the model
    Attributes:
        :true_results (array of ints): The true testing values
        :predicted_results (array of floats): The predicted probabilities
        generated by the model
    """
    def __init__(self, target, prediction):
        self.TARGET = target
        self.PREDICTION = prediction

    def youden_index(self):
        """Calculates the youden_index for each threshold and produces a plot
        of youden index vs threshold

        Args:
            None
        Returns:
            :youden_index_values (np array of floats): The values of the youden
            index at each threshold
        """

        fpr, tpr, thresholds = set_threshold_roc_curve(self.TARGET, self.PREDICTION, pos_label=1, drop_intermediate=True)

        youden_index_values = np.zeros([len(thresholds)])
        for i in range(0, len(thresholds)):
            youden_index_values[i] = ((tpr[i]+(1-fpr[i])-1))/(math.sqrt(2))

        visualization_utils.visualize_data.youden_index_plot(thresholds, youden_index_values)

        return youden_index_values

    def roc_curve(self):
        """Plots the Reciever Operating Characteristic Curve (true positive
        rate vs false positive rate) Generate area under the curve and Calculates
        true positive and false positive rate for an array of thresholds.

        Args, Returns: None
        """
        roc = sklearn.metrics.roc_auc_score(self.TARGET, self.PREDICTION)
        fpr, tpr, thresholds = set_threshold_roc_curve(self.TARGET, self.PREDICTION, pos_label=1, drop_intermediate=True)

        visualization_utils.visualize_data.roc_plot(roc, fpr, tpr, thresholds)

    def well_rounded_validation(self):
        """Calculates the AUROC, Recall, Precision, F1 Score, Accuracy, and
        confusion matrix of the model

        Args:
            None
        Returns:
                (dict) {"AUROC" : (float),
                        "Recall" : (float),
                        "Precision" : (float),
                        "Accuracy" : (float),
                        "Confusion Matrix" : (list)}

        Returns a Dict containing the AUROC, Recall, Precision, F1-score, Accuracy, and confusion matrix from the model
        """
        classified_predictions = data_utils.classify(self.PREDICTION, 0.5)
        conf_matrix = sklearn.metrics.confusion_matrix(self.TARGET, classified_predictions, labels=None)

        return {
                "AUROC" : sklearn.metrics.roc_auc_score(self.TARGET, self.PREDICTION),
                "Recall" : sklearn.metrics.recall_score(self.TARGET, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1],
                "Precision" : sklearn.metrics.precision_score(self.TARGET, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1],
                "F1 Score" : sklearn.metrics.f1_score(self.TARGET, classified_predictions),
                "Accuracy" : sklearn.metrics.accuracy_score(self.TARGET, classified_predictions),
                "Confusion Matrix" : [conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]]
                }

# def y_randomization_test(est, db):
#     """Runs a y_randomization test on the model and writes the results
#     to a text file.
#
#     Args:
#         :param est (obj): The machine learning model
#         :param db (obj): The database containing the test and train data
#     Returns:
#         None
#     """
#     with open('y_randomization.csv', 'w') as f:
#         f.write(' , Accuracy, AUROC, F1-Score, Precision, Recall\n')
#         for i in range(0,50):
#             probability_prediction = est.predict_proba(db.X_test)[:, 1]
#             f.write('Randomized {} times, {}, {}, {}, {}, {} \n'.format(i, sklearn.metrics.accuracy_score(db.Y_test, data_utils.classify(probability_prediction, 0.5)),sklearn.metrics.roc_auc_score(db.Y_test, data_utils.classify(probability_prediction, 0.5)),sklearn.metrics.f1_score(db.Y_test, data_utils.classify(probability_prediction, 0.5)),sklearn.metrics.precision_score(db.Y_test, data_utils.classify(probability_prediction, 0.5)),sklearn.metrics.recall_score(db.Y_test, data_utils.classify(probability_prediction, 0.5))))
#             np.random.shuffle(db.Y_test)

#***HELPER FUNCTIONS***#
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """
    --->Taken From sci-kit learn documentation to help set_threshold_roc_curve

    Args:
        arr : array-like
        To be cumulatively summed as flat
        axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
        rtol : float
        Relative tolerance, see ``np.allclose``
        atol : float
        Absolute tolerance, see ``np.allclose``
    Returns
            cumsum
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def set_threshold_roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """
    --->Taken from sci-kit learn documentation
    Altered to give a constant amount of thresholds for the roc curve<---

    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
        .. versionadded:: 0.17
           parameter *drop_intermediate*.
    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]
    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]
    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]
    return fpr, tpr, thresholds

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    --->Taken from sci-kit learn documentation to help set_threshold_roc_curve()
    Altered to return a constant amount of thrsholds for each roc curve
    Calculate true and false positives per binary classification threshold.<---

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (array_equal(classes, [0, 1]) or
             array_equal(classes, [-1, 1]) or
             array_equal(classes, [0]) or
             array_equal(classes, [-1]) or
             array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = stable_cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]