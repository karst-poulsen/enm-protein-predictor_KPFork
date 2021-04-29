import csv
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from typing import List


class RandomForestClassifierWithCoef(RandomForestClassifier):
    """Adds feature weights for each returned variable from the
    sklearn RandomForestClassifier:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py
    """
    def fit(self, *args, **kwargs):
        """Overloaded fit method to include the feature importances
        of each variable. This is used for RFECV
        """
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

    def optimize(self, num_folds: int, num_trees_grid: List[int], max_features_grid: List[str], max_depth_grid: List[int], min_samples_split_grid: List[str], criterion_grid: List[str], train_features: pd.DataFrame, train_target: pd.DataFrame, best_params_path: str) -> None:
        """This function optimizes the machine learning classifier's hyperparameters,
        print the parameters that give the best accuracy based on a
        5 fold cross validation.

        Args:
            :param model (obj): The sklearn model you wish to optimize
            :param X_train (array): The X values of training data
            :param Y_train (array): The Y values of the training data

        Returns:
            None
        """
        # add whatever your heart desires to param grid, keep in mind its an incredibly inefficient algorithm
        param_grid = {
            'n_estimators': num_trees_grid,
            'max_features': max_features_grid,
            'max_depth': max_depth_grid,
            'min_samples_split': min_samples_split_grid,
            'criterion': criterion_grid,
            'n_jobs': [-1],
        }
        # 5 fold validation
        CV_est = GridSearchCV(estimator=self, param_grid=param_grid, cv=num_folds, verbose=2)
        CV_est.fit(train_features, train_target)
        best_params = CV_est.best_params_
        print(f"Best parameters: \n {best_params}")
        j = json.dumps(best_params)
        with open(best_params_path, 'w+') as f:
            f.write(j)
        return None


    def recursive_feature_elimination(self, step: float, folds: int, min_features: int, scoring: str, train_features: pd.DataFrame, train_target: pd.DataFrame, mask_path: str) -> None:
        """Runs RFECV with 5 folds, stores optimum features
        useful for feature engineering in a text file as a binary mask

        Args:
            :param model (obj): The sklean model you wish to optimize
            :param X_train (array): The X values of training data
            :param Y_train (array): The Y values of the training data
            :param mask_file (string): Path to a textfile to write binary mask
        Returns:
            None
        """

        selector = RFECV(estimator=self, step=step, cv=folds, scoring=scoring, min_features_to_select=min_features, verbose=1)
        print(train_features)
        selector = selector.fit(train_features, train_target)
        print(f"selector support: {selector.support_} \n selector ranking: {selector.ranking_}")
        print(f"Optimal number of features: {selector.n_features_} \n Selector grid scores: {selector.grid_scores_}")
        # write optimum binary mask to text file
        selector_support_list = selector.support_.tolist()
        with open(mask_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(selector_support_list)
        return None
