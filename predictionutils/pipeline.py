from dataclasses import dataclass, field
import pandas as pd
import random
from typing import List, Tuple
from yamldataclassconfig.config import YamlDataClassConfig

from . import data_utils, predictor_utils, validation_utils


@dataclass
class Config(YamlDataClassConfig):
    CLEAN_CATEGORICAL_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_DROP_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_ENRICHMENT_SPLIT_VALUE: int = None
    CLEAN_FILL_NAN_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_MASK_PATH: str = ''
    CLEAN_MULTI_LABEL_ENCODE_FIELDS: List[str] = field(default_factory=lambda: [])
    INIT_DATA_PATH: str = None
    MODEL_BOOTSTRAP: bool = False
    MODEL_ESTIMATOR_COUNT: int = None
    MODEL_MIN_SAMPLE_SPLIT: int = None
    MODEL_NUM_JOBS: int = None
    OPT_NUM_FOLDS: int = 2
    OPT_NUM_TREES_GRID: List[int] = field(default_factory=lambda: [1])
    OPT_MAX_FEATURES_GRID: List[str] = field(default_factory=lambda: ['auto'])
    OPT_MAX_DEPTH_GRID: List[int] = field(default_factory=lambda: [2])
    OPT_MIN_SAMPLES_SPLIT_GRID: List[int] = field(default_factory=lambda: [2])
    OPT_CRITERION_GRID: List[str] = field(default_factory=lambda: ['gini'])
    RFECV_MIN_FEATURES: int = 1
    RFECV_NUM_FOLDS: int = 5
    RFECV_SCORING: str = 'f1'
    RFECV_STEP: float = 1
    TRAIN_SPLIT_PERCENTAGE: float = 0.6

    def list_variables(self):
        return [self.CLEAN_CATEGORICAL_FIELDS, self.CLEAN_DROP_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE,
                self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_MASK_PATH, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS,
                self.INIT_DATA_PATH, self.MODEL_BOOTSTRAP, self.MODEL_ESTIMATOR_COUNT,
                self.MODEL_MIN_SAMPLE_SPLIT, self.MODEL_NUM_JOBS, self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID,
                self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID,
                self.OPT_CRITERION_GRID, self.RFECV_MIN_FEATURES, self.RFECV_NUM_FOLDS,
                self.RFECV_SCORING, self.RFECV_STEP, self.TRAIN_SPLIT_PERCENTAGE]


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
    def __init__(self, config_path="/Users/mct19/repos/ENM-Protein-Predictor/config/config-train.yml"):
        self.CONFIGS = Config()
        self.CONFIG_PATH = config_path
        self.CONFIGS.load(self.CONFIG_PATH)
        self.CLEAN_CATEGORICAL_FIELDS, self.CLEAN_DROP_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE, self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_MASK_PATH, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS, self.INIT_DATA_PATH, self.MODEL_BOOTSTRAP, self.MODEL_ESTIMATOR_COUNT, self.MODEL_MIN_SAMPLE_SPLIT, self.MODEL_NUM_JOBS, self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID, self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID, self.OPT_CRITERION_GRID, self.RFECV_MIN_FEATURES, self.RFECV_NUM_FOLDS, self.RFECV_SCORING, self.RFECV_STEP, self.TRAIN_SPLIT_PERCENTAGE = self.CONFIGS.list_variables()
        self.DB = data_utils.Database(self.INIT_DATA_PATH)

    def clean_train_and_target(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        multi_label_encoded_train = self.DB.multi_label_encode(self.DB.RAW_DATA, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS)
        one_hot_encoded_train = self.DB.one_hot_encode(multi_label_encoded_train, self.CLEAN_CATEGORICAL_FIELDS)
        train, target = self.DB.clean_raw_data(one_hot_encoded_train, self.CLEAN_DROP_FIELDS, self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE)
        return train, target

    def mask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = data_utils.DataUtils()
        mask = d.get_mask(self.CLEAN_MASK_PATH)
        masked_features = d.apply_mask(mask, df)
        return masked_features

    def get_estimator(self):
        est = predictor_utils.RandomForestClassifierWithCoef(
            n_estimators=self.MODEL_ESTIMATOR_COUNT,
            bootstrap=self.MODEL_BOOTSTRAP,
            min_samples_split=self.MODEL_MIN_SAMPLE_SPLIT,
            n_jobs=self.MODEL_NUM_JOBS,
            random_state=random.randint(1, 2 ** 8)
        )
        return est

    def train(self):
        train_cleaned, target_cleaned = self.clean_train_and_target()
        train_features, train_target, val_features, val_target = self.DB.split_data(self.TRAIN_SPLIT_PERCENTAGE, train_cleaned, target_cleaned)
        train_features_no_mask, train_accession_numbers = self.DB.save_accession_numbers(train_features)
        val_features_no_mask = self.DB.save_accession_numbers(val_features)[0]
        train_features = self.mask_features(train_features_no_mask)
        val_features = self.mask_features(val_features_no_mask)
        est = self.get_estimator()

        est.fit(train_features, train_target)
        probability_prediction = est.predict_proba(val_features)[:, 1]

        #validator.y_randomization_test(est, db) #run y_randomization_test
        val = validation_utils.validation_metrics(val_target, probability_prediction)
        classification_information = (probability_prediction, val_target, train_accession_numbers, val_features)
        feature_importances = dict(zip(list(train_features), est.feature_importances_))

        return val.well_rounded_validation(), feature_importances, classification_information

    def optimize(self, best_params_out: str):
        train_cleaned, target_cleaned = self.clean_train_and_target()
        train_no_mask = self.DB.save_accession_numbers(train_cleaned)[0]
        train = self.mask_features(train_no_mask)
        est = self.get_estimator()
        return predictor_utils.optimize(est, self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID, self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID, self.OPT_CRITERION_GRID, train, target_cleaned, best_params_out)

    def rfecv(self, mask_out: str):
        train_cleaned, target_cleaned = self.clean_train_and_target()
        train_no_mask = self.DB.save_accession_numbers(train_cleaned)[0]
        est = self.get_estimator()
        return predictor_utils.recursive_feature_elimination(est, self.RFECV_STEP, self.RFECV_NUM_FOLDS, self.RFECV_MIN_FEATURES, self.RFECV_SCORING, train_no_mask, target_cleaned, mask_out)
