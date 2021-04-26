import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV


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

def optimize(model, train_features, train_target):
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
        'n_estimators': [2500],
        'max_features': ['auto'],
        'max_depth': [None],
        # 'min_samples_split': [5, 10, 15, 20, 50], #results from first run was 5
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9],
        # 'min_samples_leaf': [1, 5, 15, 20, 50], # results from first run was 1
        'min_samples_leaf': [1],  # min_samples_leaf isn't necessary when using min_samples_split anyways
        'n_jobs': [-1],
    }
    # 5 fold validation
    CV_est = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    CV_est.fit(train_features, train_target)
    print(f"Best parameters: \n {CV_est.best_params_}")

def recursive_feature_elimination(model, train_features, train_target, mask_file):
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
    assert isinstance(mask_file, str), "please pass a string specifying maskfile location"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mask_file = os.path.join(dir_path, mask_file)

    selector = RFECV(estimator=model, step=1, cv=10, scoring='f1', verbose=1)
    selector = selector.fit(train_features, train_target)
    print(f"selector support: {selector.support_} \n selector ranking: {selector.ranking_}")
    print(f"Optimal number of features: {selector.n_features_} \n Selector grid scores: {selector.grid_scores_}")
    # write optimum binary mask to text file
    with open(mask_file, 'w') as f:
        for item in selector.support_:
            f.write('{}, '.format(item))