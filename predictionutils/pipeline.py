from dataclasses import dataclass, field
import pandas as pd
from typing import List, Tuple, Union
from yamldataclassconfig.config import YamlDataClassConfig

from . import data_utils, predictor_utils, validation_utils


@dataclass
class Config(YamlDataClassConfig):
    CLEAN_ACCESSION_NUMBER_FIELDNAME: str = 'Accesion Number'
    CLEAN_CATEGORICAL_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_DROP_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_ENRICHMENT_SPLIT_VALUE: int = None
    CLEAN_FILL_NAN_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_MASK_PATH: str = ''
    CLEAN_MULTI_LABEL_ENCODE_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_TARGET_FIELDNAME: str = None
    INIT_DATA_PATH: str = None
    MODEL_BOOTSTRAP: bool = False
    MODEL_NUM_TREES: int = None
    MODEL_MIN_SAMPLE_SPLIT: int = None
    MODEL_MAX_FEATURES: str = None
    MODEL_MAX_DEPTH: int = None
    MODEL_CRITERION: str = None
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
    VALIDATION_PROBABLITY_THRESHOLD: float = 0.5

    def list_variables(self):
        return [self.CLEAN_ACCESSION_NUMBER_FIELDNAME, self.CLEAN_CATEGORICAL_FIELDS,
                self.CLEAN_DROP_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE,
                self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_MASK_PATH, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS,
                self.CLEAN_TARGET_FIELDNAME, self.INIT_DATA_PATH, self.MODEL_BOOTSTRAP, self.MODEL_NUM_TREES,
                self.MODEL_MIN_SAMPLE_SPLIT, self.MODEL_MAX_FEATURES, self.MODEL_MAX_DEPTH,
                self.MODEL_CRITERION, self.MODEL_NUM_JOBS, self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID,
                self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID,
                self.OPT_CRITERION_GRID, self.RFECV_MIN_FEATURES, self.RFECV_NUM_FOLDS,
                self.RFECV_SCORING, self.RFECV_STEP, self.TRAIN_SPLIT_PERCENTAGE,
                self.VALIDATION_PROBABLITY_THRESHOLD]


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
        self.CLEAN_ACCESSION_NUMBER_FIELDNAME, self.CLEAN_CATEGORICAL_FIELDS, self.CLEAN_DROP_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE, self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_MASK_PATH, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS, self.CLEAN_TARGET_FIELDNAME, self.INIT_DATA_PATH, self.MODEL_BOOTSTRAP, self.MODEL_NUM_TREES, self.MODEL_MIN_SAMPLE_SPLIT, self.MODEL_MAX_FEATURES, self.MODEL_MAX_DEPTH, self.MODEL_CRITERION, self.MODEL_NUM_JOBS, self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID, self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID, self.OPT_CRITERION_GRID, self.RFECV_MIN_FEATURES, self.RFECV_NUM_FOLDS, self.RFECV_SCORING, self.RFECV_STEP, self.TRAIN_SPLIT_PERCENTAGE, self.VALIDATION_PROBABLITY_THRESHOLD = self.CONFIGS.list_variables()
        self.DB = data_utils.Database(self.INIT_DATA_PATH)

    def clean_and_encode(self, df: pd.DataFrame) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        d = data_utils.DataUtils()
        multi_label_encoded_train = d.multi_label_encode(df, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS)
        one_hot_encoded_train = d.one_hot_encode(multi_label_encoded_train, self.CLEAN_CATEGORICAL_FIELDS)
        train, target = self.DB.clean_raw_data(one_hot_encoded_train, self.CLEAN_DROP_FIELDS, self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE, self.CLEAN_TARGET_FIELDNAME)
        return train, target

    def mask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = data_utils.DataUtils()
        mask = d.get_mask(self.CLEAN_MASK_PATH)
        masked_features = d.apply_mask(mask, df)
        return masked_features

    def get_estimator(self):
        est = predictor_utils.RandomForestClassifierWithCoef(
            n_estimators=self.MODEL_NUM_TREES,
            max_depth=self.MODEL_MAX_DEPTH,
            max_features=self.MODEL_MAX_FEATURES,
            criterion=self.MODEL_CRITERION,
            bootstrap=self.MODEL_BOOTSTRAP,
            min_samples_split=self.MODEL_MIN_SAMPLE_SPLIT,
            n_jobs=self.MODEL_NUM_JOBS
        )
        return est

    def train(self):
        d = data_utils.DataUtils()
        df = self.DB.RAW_DATA
        train_accession_numbers = d.save_accession_numbers(df, self.CLEAN_ACCESSION_NUMBER_FIELDNAME)
        train_cleaned, target_cleaned = self.clean_and_encode(df)
        train_features, train_target, val_features, val_target = d.split_data(self.TRAIN_SPLIT_PERCENTAGE, train_cleaned, target_cleaned)

        est = self.get_estimator()

        est.fit(train_features, train_target)
        probability_prediction = pd.Series(est.predict_proba(val_features)[:, 1])

        val = validation_utils.ValidationUtils(val_target, probability_prediction)
        classification_information = (probability_prediction, val_target, train_accession_numbers, val_features)
        feature_importances = dict(zip(list(train_features), est.feature_importances_))

        return val.well_rounded_validation(self.VALIDATION_PROBABLITY_THRESHOLD), feature_importances, classification_information

    def optimize(self, best_params_out: str):
        df = self.DB.RAW_DATA
        train_cleaned, target_cleaned = self.clean_and_encode(df)
        train = self.mask_features(train_cleaned)
        est = self.get_estimator()
        return est.optimize(self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID,
                            self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID,
                            self.OPT_MIN_SAMPLES_SPLIT_GRID, self.OPT_CRITERION_GRID,
                            train, target_cleaned, best_params_out)

    def rfecv(self, mask_out: str):
        df = self.DB.RAW_DATA
        train_cleaned, target_cleaned = self.clean_and_encode(df)
        est = self.get_estimator()
        return est.recursive_feature_elimination(self.RFECV_STEP, self.RFECV_NUM_FOLDS, self.RFECV_MIN_FEATURES, self.RFECV_SCORING, train_cleaned, target_cleaned, mask_out)
