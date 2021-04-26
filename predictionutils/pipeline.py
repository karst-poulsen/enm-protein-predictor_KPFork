from dataclasses import dataclass
import random
from typing import List
from yamldataclassconfig.config import YamlDataClassConfig

from . import data_utils, predictor_utils, validation_utils


@dataclass
class Config(YamlDataClassConfig):
    BOOTSTRAP: bool = False
    CATEGORICAL_FIELDS: List[str] = None
    DATA_PATH: str = None
    DROP_FIELDS: List[str] = None
    ENRICHMENT_SPLIT_VALUE: int = None
    ESTIMATOR_COUNT: int = None
    ITERATIONS: int = None
    MASK_PATH: str = None
    MIN_SAMPLE_SPLIT: int = None
    MULTI_LABEL_ENCODE_FIELDS: List[str] = None
    NUM_JOBS: int = None
    TRAIN_PERCENTAGE: float = None

    def list_variables(self):
        return [self.BOOTSTRAP, self.CATEGORICAL_FIELDS, self.DATA_PATH, self.DROP_FIELDS,
                self.ENRICHMENT_SPLIT_VALUE, self.ESTIMATOR_COUNT, self.ITERATIONS,
                self.MASK_PATH, self.MIN_SAMPLE_SPLIT, self.MULTI_LABEL_ENCODE_FIELDS,
                self.NUM_JOBS, self.TRAIN_PERCENTAGE]


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
    def __init__(self, default_config_path="/Users/mct19/repos/ENM-Protein-Predictor/config/config.yml"):
        self.CONFIGS = Config()
        self.DEFAULT_CONFIG_PATH = default_config_path
        self.CONFIGS.load(self.DEFAULT_CONFIG_PATH)
        self.BOOTSTRAP, self.CATEGORICAL_FIELDS, self.DATA_PATH, self.DROP_FIELDS, self.ENRICHMENT_SPLIT_VALUE, self.ESTIMATOR_COUNT, self.ITERATIONS, self.MASK_PATH, self.MIN_SAMPLE_SPLIT, self.MULTI_LABEL_ENCODE_FIELDS, self.NUM_JOBS, self.TRAIN_PERCENTAGE = self.CONFIGS.list_variables()
        self.DB = data_utils.Database(self.DATA_PATH)

    def get_train_and_target(self):
        multi_label_encoded_train = self.DB.multi_label_encode(self.DB.RAW_DATA, self.MULTI_LABEL_ENCODE_FIELDS)
        one_hot_encoded_train = self.DB.one_hot_encode(multi_label_encoded_train, self.CATEGORICAL_FIELDS)
        train, target = self.DB.clean_raw_data(one_hot_encoded_train, self.DROP_FIELDS, self.ENRICHMENT_SPLIT_VALUE)
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
