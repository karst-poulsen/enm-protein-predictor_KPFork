import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from typing import List


class RandomForestClassifierWithCoef(RandomForestClassifier):
    """
    Wrapper for RandomForestClassifier with added feature weights.
    """
    def fit(self, *args, **kwargs):
        """
        Overloaded fit method to include the feature importances of each variable.

        Args:
            *args: see RandomForestClassifier documentation
            **kwargs: see RandomForestClassifier documentation

        Returns: RandomForestClassifierWithCoef | fitted model

        """
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
        return self

    def optimize(self, num_folds: int, num_trees_grid: List[int], max_features_grid: List[str], max_depth_grid: List[int], min_samples_split_grid: List[str], criterion_grid: List[str], features: pd.DataFrame, labels: pd.DataFrame) -> GridSearchCV:
        """
        Optimize the random forest classifier based on grid defined in hyperparameters.

        Args:
            num_folds: int | Number of folds for cross-validation
            num_trees_grid: List[int] | grid listing number of trees for optimization
            max_features_grid: List[int] | grid listing max feature strategies to optimize
            max_depth_grid: List[int] | grid listing max tree depths for optimization
            min_samples_split_grid: List[int] | grid listing min sample split size for optimization
            criterion_grid: List[int] | grid listing functions for measuring quality of split for optimization
            features: pd.DataFrame | dataframe of features
            labels: pd.DataFrame | dataframe of labels

        Returns: GridSearchCV | fitted model

        """
        param_grid = {
            'n_estimators': num_trees_grid,
            'max_features': max_features_grid,
            'max_depth': max_depth_grid,
            'min_samples_split': min_samples_split_grid,
            'criterion': criterion_grid,
            'n_jobs': [-1],
        }
        cv_est = GridSearchCV(estimator=self, param_grid=param_grid, cv=num_folds, verbose=2)
        cv_est.fit(features, labels)
        return cv_est

    def recursive_feature_elimination(self, step: float, folds: int, min_features: int, scoring: str, train_features: pd.DataFrame, train_target: pd.DataFrame) -> RFECV:
        """
        Run recursive feature elimination to optimize features used by model.

        Args:
            step: float | number of features (or percent of features of < 1) to remove at each step
            folds: int | number of folds for cross validation
            min_features: int | minimum allowed number of features
            scoring: str | scoring metric to determine feature importance
            train_features: pd.DataFrame | dataframe of features
            train_target: pd.DataFrame | dataframe of labels

        Returns: RFECV | fitted model

        """

        selector = RFECV(estimator=self, step=step, cv=folds, scoring=scoring, min_features_to_select=min_features, verbose=1)
        selector = selector.fit(train_features, train_target)
        return selector
