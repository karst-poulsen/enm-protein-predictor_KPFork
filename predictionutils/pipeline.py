import random

from . import data_utils, predictor_utils, validation_utils


class Pipeline:
    """
    Runs the pipeline. Trains and evaluates the estimator, outputs metrics and
    information about the model performance.

    Args:
        :param db (database obj): The database object, passed from main.
        Information about this class can be found in data_utils
        :param optimize (bool): Set to true to run Grid search
        :param RFECV (bool): Set to true to run RFECV
    Returns:
        :val.well_rounded_validation() (dict): returns a dictionary of validation metrics
        :feature_importances (dict): contains a dictionary of feature importances
        :classification_information (dict): information about the predictions
    """
    def __init__(self, bootstrap: bool, db: data_utils.Database, estimator_count: int, mask_path: str, min_sample_split: int, num_jobs: int, train_percentage: float):
        self.BOOTSTRAP = bootstrap
        self.DB = db
        self.ESTIMATOR_COUNT = estimator_count
        self.MASK_PATH = mask_path
        self.MIN_SAMPLE_SPLIT = min_sample_split
        self.NUM_JOBS = num_jobs
        self.TRAIN_PERCENTAGE = train_percentage

    def get_train_and_target(self):
        train, target = self.DB.clean_raw_data()
        test_accession_numbers, train_features_no_acc, train_target, val_features_no_acc, val_target = self.DB.split_data(
            self.TRAIN_PERCENTAGE, train, target)
        d = data_utils.DataUtils()
        mask = d.get_mask(self.MASK_PATH)
        train_features = d.apply_mask(mask, train_features_no_acc)
        val_features = d.apply_mask(mask, val_features_no_acc)
        return train_features, train_target, val_features, val_target, test_accession_numbers

    def get_estimator(self):
        est = predictor_utils.RandomForestClassifierWithCoef(
            n_estimators=self.ESTIMATOR_COUNT,
            bootstrap=self.BOOTSTRAP,
            min_samples_split=self.MIN_SAMPLE_SPLIT,
            n_jobs=self.NUM_JOBS,
            random_state=random.randint(1, 2 ** 8)
        )
        return est

    def run(self):
        train_features, train_target, val_features, val_target, test_accession_numbers = self.get_train_and_target()
        est = self.get_estimator()

        est.fit(train_features, train_target)
        probability_prediction = est.predict_proba(val_features)[:, 1]

        #validator.y_randomization_test(est, db) #run y_randomization_test
        val = validation_utils.validation_metrics(val_target, probability_prediction)
        classification_information = (probability_prediction, val_target, test_accession_numbers, val_features)
        feature_importances = dict(zip(list(train_features), est.feature_importances_))

        return val.well_rounded_validation(), feature_importances, classification_information

    def optimize(self):
        train_features, train_target, val_features, val_target, test_accession_numbers = self.get_train_and_target()
        est = self.get_estimator()
        return predictor_utils.optimize(est, train_features, train_target)

    def rfecv(self):
        train_features, train_target, val_features, val_target, test_accession_numbers = self.get_train_and_target()
        est = self.get_estimator()
        return predictor_utils.recursive_feature_elimination(est, train_features, train_target, 'tst.txt')
